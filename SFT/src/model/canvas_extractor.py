from transformers import CLIPVisionModel, CLIPProcessor, SiglipVisionModel
import torch
import torch.nn as nn
from .canvas_compressor import LearnableCanvasCompressor

class CanvasExtractor(nn.Module):
    def __init__(self, model_path, canvas_token_num, config, torch_dtype, attn_implementation, llm_hidden_dim=2048):
        super().__init__()
        # CLIP frozen
        self.clip_vision = CLIPVisionModel.from_pretrained(
            model_path,
            config=config.vision_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()

        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        self.canvas_token_num = canvas_token_num

        # Projector trainable
        self.canvas_proj = nn.Linear(config.vision_config.hidden_size, llm_hidden_dim, dtype=torch_dtype)

    @torch.no_grad()
    def encode_clip(self, pixel_values):
        """
        Only return CLIP output's patch tokens, w/o CLS token
        """
        # outputs = self.clip_vision(pixel_values, output_hidden_states=True).hidden_states[-2]

        # return outputs[:,1:,:]

        outputs = self.clip_vision(pixel_values)
        if self.canvas_token_num == 1:
            return outputs.last_hidden_state[:, :1, :]
        return outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_dim]

    def forward(self, pixel_values, n_canvas_tokens=576):
        """
        pixel_values: [B, 3, H, W] PIL->tensor
        n_canvas_tokens: map to canvas token
        """
        clip_tokens = self.encode_clip(pixel_values)  # [B, P, hidden_dim]
        
        canvas_tokens = self.canvas_proj(clip_tokens)  # [B, P, llm_hidden_dim]

        B, P, D = canvas_tokens.shape
      
        return canvas_tokens#.view(-1,D)


class CanvasExtractor_Siglip(nn.Module):
    def __init__(self, model_path, canvas_token_num, config, torch_dtype, attn_implementation, llm_hidden_dim=2048,
                 use_compressor=False, compressor_n_heads=8, compressor_n_layers=2,
                 adaptive_budget=False, max_n_queries=64):
        super().__init__()
        # Siglip frozen
        self.siglip2_vision = SiglipVisionModel.from_pretrained(
            model_path,
            config=config.vision_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation).eval()

        for param in self.siglip2_vision.parameters():
            param.requires_grad = False
        
        self.canvas_token_num = canvas_token_num
        self.use_compressor = use_compressor

        # Projector trainable
        self.canvas_proj = nn.Linear(config.vision_config.hidden_size, llm_hidden_dim, dtype=torch_dtype)

        # Learnable cross-attention compressor (replaces average pooling)
        if use_compressor:
            self.compressor = LearnableCanvasCompressor(
                patch_dim=config.vision_config.hidden_size,  # compress in SigLIP2 feature space
                n_queries=canvas_token_num,
                n_heads=compressor_n_heads,
                n_layers=compressor_n_layers,
                adaptive=adaptive_budget,
                max_n_queries=max_n_queries,
                torch_dtype=torch_dtype,
            )

    @torch.no_grad()
    def vision_encode(self, pixel_values):
        """
        Only return CLIP output's patch tokens, w/o CLS token
        """
        outputs = self.siglip2_vision(pixel_values)

        if self.canvas_token_num == 1 and not self.use_compressor:
            return outputs.pooler_output.unsqueeze(1)
   
        return outputs.last_hidden_state  # [B, num_patches, hidden_dim]

    def forward(self, pixel_values, n_canvas_tokens=576):
        """
        pixel_values: [B, 3, H, W] PIL->tensor
        n_canvas_tokens: target canvas token count
        
        Returns:
            canvas_tokens: [B, N, llm_hidden_dim] 
                With compressor: N = canvas_token_num (or adaptive length)
                Without compressor: N = P (original patch count, compressed later via average pooling)
            budget_info: dict, adaptive length info (only valid when use_compressor + adaptive)
        """
        clip_tokens = self.vision_encode(pixel_values)  # [B, P, hidden_dim]
        
        budget_info = {}
        
        if self.use_compressor:
            # Use learnable cross-attention compressor
            # Compress in SigLIP2 feature space, then project
            compressed_tokens, budget_info = self.compressor(
                clip_tokens, 
                n_target_tokens=self.canvas_token_num
            )  # [B, N, hidden_dim]
            
            # Project to LLM hidden dimension
            canvas_tokens = self.canvas_proj(compressed_tokens)  # [B, N, llm_hidden_dim]
        else:
            # Original approach: project first, compress later via average pooling in forward
            canvas_tokens = self.canvas_proj(clip_tokens)  # [B, P, llm_hidden_dim]

        return canvas_tokens