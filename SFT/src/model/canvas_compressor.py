"""
    Learnable Canvas Compressor

    Uses learnable query tokens with cross-attention to extract information
    from SigLIP2 patch tokens, automatically focusing on key canvas regions.
    Supports both fixed-length and adaptive-length modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableCanvasCompressor(nn.Module):
    """
    Compresses variable-length patch tokens into a fixed/adaptive number of canvas tokens
    via cross-attention.

    Core idea:
    - Maintains a set of learnable query tokens (count = max_n_queries)
    - Extracts key information from patch tokens through multi-layer cross-attention
    - Optional adaptive length: predicts the required token count via a budget predictor
    """
    
    def __init__(
        self,
        patch_dim: int,          # SigLIP2 patch token dimension (e.g. 1152)
        n_queries: int = 27,     # default number of compressed query tokens
        n_heads: int = 8,        # number of multi-head attention heads
        n_layers: int = 2,       # number of cross-attention layers
        adaptive: bool = False,  # whether to enable adaptive length
        max_n_queries: int = 64, # max query count in adaptive mode
        torch_dtype=None,
    ):
        super().__init__()
        
        self.patch_dim = patch_dim
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.adaptive = adaptive
        self.max_n_queries = max_n_queries if adaptive else n_queries
        
        # Learnable query tokens, initialized with truncated normal distribution
        self.queries = nn.Parameter(
            torch.randn(self.max_n_queries, patch_dim, dtype=torch_dtype) * 0.02
        )
        
        # Multi-layer cross-attention decoder layers
        # Each layer: self-attention on queries + cross-attention to patch tokens + FFN
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=patch_dim,
                n_heads=n_heads,
                dim_feedforward=patch_dim * 4,
                torch_dtype=torch_dtype,
            )
            for _ in range(n_layers)
        ])
        
        # Layer norm
        self.final_norm = nn.LayerNorm(patch_dim, dtype=torch_dtype)
        
        # Adaptive length: budget predictor
        if self.adaptive:
            self.budget_predictor = CanvasBudgetPredictor(
                hidden_dim=patch_dim,
                max_tokens=max_n_queries,
                torch_dtype=torch_dtype,
            )

    def forward(self, patch_tokens: torch.Tensor, n_target_tokens: int = None):
        """
        Args:
            patch_tokens: [B, P, D] patch tokens from SigLIP2 (P=729)
            n_target_tokens: optional, specifies output token count (determined by GT during training)

        Returns:
            compressed_tokens: [B, N, D] compressed canvas tokens
            budget_info: dict, adaptive length info (only valid when adaptive=True)
        """
        B = patch_tokens.shape[0]
        budget_info = {}
        
        if self.adaptive and n_target_tokens is None:
            # Adaptive mode: predict required token count per sample via budget predictor
            # Uses global average pooling of patch tokens as input
            global_feat = patch_tokens.mean(dim=1)  # [B, D]
            predicted_budget, budget_logits = self.budget_predictor(global_feat)
            budget_info["predicted_budget"] = predicted_budget
            budget_info["budget_logits"] = budget_logits
            # At inference, use predicted budget; take batch max to unify length
            n_output = predicted_budget.max().item()
        else:
            n_output = n_target_tokens if n_target_tokens is not None else self.n_queries
        
        # Select the first n_output query tokens
        queries = self.queries[:n_output].unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        # Multi-layer cross-attention
        for layer in self.layers:
            queries = layer(queries, patch_tokens)
        
        # Final LayerNorm
        compressed_tokens = self.final_norm(queries)  # [B, N, D]
        
        return compressed_tokens, budget_info


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer:
    1. Self-Attention on queries (self-attention among query tokens)
    2. Cross-Attention from queries to patch tokens (extracts info from patch tokens)
    3. Feed-Forward Network

    Each sub-layer has residual connections and LayerNorm (Pre-Norm style)
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0, torch_dtype=None):
        super().__init__()
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True, dtype=torch_dtype
        )
        self.self_attn_norm = nn.LayerNorm(d_model, dtype=torch_dtype)
        
        # Cross-attention: queries attend to patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True, dtype=torch_dtype
        )
        self.cross_attn_norm = nn.LayerNorm(d_model, dtype=torch_dtype)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, dtype=torch_dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, dtype=torch_dtype),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model, dtype=torch_dtype)
    
    def forward(self, queries: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, N, D] query tokens
            patch_tokens: [B, P, D] patch tokens from SigLIP2
        Returns:
            queries: [B, N, D] updated query tokens
        """
        # 1. Self-Attention (Pre-Norm)
        residual = queries
        queries = self.self_attn_norm(queries)
        queries, _ = self.self_attn(queries, queries, queries)
        queries = residual + queries
        
        # 2. Cross-Attention (Pre-Norm)
        residual = queries
        queries = self.cross_attn_norm(queries)
        queries, _ = self.cross_attn(queries, patch_tokens, patch_tokens)
        queries = residual + queries
        
        # 3. FFN (Pre-Norm)
        residual = queries
        queries = self.ffn_norm(queries)
        queries = self.ffn(queries)
        queries = residual + queries
        
        return queries


class CanvasBudgetPredictor(nn.Module):
    """
    Predicts how many tokens the current canvas requires based on
    global features of patch tokens.

    Output is an integer (selected via argmax); uses Gumbel-Softmax during
    training to remain differentiable.
    """
    
    def __init__(self, hidden_dim: int, max_tokens: int = 64, torch_dtype=None):
        super().__init__()
        self.max_tokens = max_tokens
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, max_tokens, dtype=torch_dtype),
        )
    
    def forward(self, global_feat: torch.Tensor):
        """
        Args:
            global_feat: [B, D] - global average pooled feature of patch tokens
        Returns:
            budget: [B] - token budget per sample (integer, minimum 1)
            logits: [B, max_tokens] - raw logits for loss computation
        """
        logits = self.predictor(global_feat)  # [B, max_tokens]
        
        if self.training:
            # Use Gumbel-Softmax during training to keep differentiable
            soft_budget = F.gumbel_softmax(logits, tau=1.0, hard=True)
            budget = soft_budget.argmax(dim=-1) + 1  # [B], at least 1 token
        else:
            # At inference, directly take argmax
            budget = logits.argmax(dim=-1) + 1  # [B]
        
        return budget, logits
