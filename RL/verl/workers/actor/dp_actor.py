# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""
import hylar_rl_patch
import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input
#from verl.workers.actor.fa_shim import index_first_axis, pad_input, unpad_input # implementation by AXZ


from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]

def collect_varlen_segment_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_id: int,
    end_id: int,
) -> torch.Tensor:
    """
    Collect indices (on the varlen/unpadded sequence) for token positions strictly inside
    matched (start, end) segments, skipping the first matched segment per sequence.

    Args:
        input_ids: LongTensor of shape (B, S).
        attention_mask: Bool/Long/Byte Tensor of shape (B, S); 1=valid, 0=pad/ignored.
        start_id: int, the start marker token id.
        end_id: int, the end marker token id.

    Returns:
        LongTensor of shape (K,), where each element is the index on the unpadded varlen
        sequence (i.e., the [1, L] after unpad+transpose) corresponding to a kept position.
    """
    assert input_ids.dim() == 2, "input_ids must be 2D (B, S)"
    assert attention_mask.shape == input_ids.shape, "attention_mask must match input_ids"

    device = input_ids.device
    B, S = input_ids.shape

    # Ensure mask is 0/1 long tensor
    mask = attention_mask.to(dtype=torch.long)

    # Flatten mask to compute varlen positions: prefix sum gives mapping to [0..T-1]
    # For any flat position p with mask[p]==1, its varlen index is prefix[p]-1.
    mask_flat = mask.reshape(-1)                                      # (B*S,)
    prefix = torch.cumsum(mask_flat, dim=0)                           # (B*S,)
    # We will only index prefix at places where mask==1.

    varlen_indices_per_batch = []
    varlen_indices_by_batch = []
    for b in range(B):
        varlen_indices = []
        row_ids = input_ids[b]                                        # (S,)
        row_mask = mask[b]                                            # (S,)

        # Find all start/end positions (on [0..S-1])
        starts = (row_ids == start_id).nonzero(as_tuple=False).squeeze(-1)  # (Ns,) or empty
        ends   = (row_ids == end_id).nonzero(as_tuple=False).squeeze(-1)    # (Ne,) or empty

        if starts.numel() == 0 or ends.numel() == 0:
            varlen_indices_by_batch.append(varlen_indices)
            continue

        # Two-pointer greedy matching: for each start, find the nearest end to its right.
        i_ptr, j_ptr = 0, 0
        matched = []  # list of (s, e), with e > s
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = starts[i_ptr].item()
            # Move j_ptr until we find an end strictly to the right of s_pos
            while j_ptr < ends.numel() and ends[j_ptr].item() <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = ends[j_ptr].item()
            matched.append((s_pos, e_pos))
            i_ptr += 1
            j_ptr += 1

        if len(matched) <= 0:
            # Nothing (or only the first segment which we must skip)
            varlen_indices_by_batch.append(varlen_indices)
            continue

        for (s_pos, e_pos) in matched[:]:
            # New latent emission logic in hylar_gpu_model_runner.py:
            # - When start_id is sampled, just_saw_start=True, token is emitted as-is.
            # - Next step: model processes start_id embedding, active=True,
            #   latent[0] is emitted (hidden state after seeing start_id).
            # - In latent mode, tokens are forced to hylar_id; their embeddings
            #   are overridden by the pending hidden state each step.
            # - When current_len >= latent_size, force emit end_id.
            #
            # For latent_size=N:
            # - Token sequence: [start_id, hylar_id, hylar_id, ..., hylar_id, end_id]
            #                    s_pos     s_pos+1   s_pos+2       s_pos+N   e_pos
            # - Latents emitted: N total (latent[0] at s_pos+1, ..., latent[N-1] at s_pos+N)
            # - latent count = e_pos - s_pos - 1 = N
            #
            # Position mapping (does NOT include start_id position):
            # - latent[0]: hidden[s_pos+1] (first hylar_id position)
            # - latent[1]: hidden[s_pos+2]
            # - ...
            # - latent[N-1]: hidden[s_pos+N] = hidden[e_pos-1]
            #
            # So we need range(s_pos+1, e_pos), which has length e_pos - s_pos - 1
            if e_pos > s_pos + 1:
                inner = torch.arange(s_pos + 1, e_pos, device=device, dtype=torch.long)
            else:
                inner = torch.empty(0, device=device, dtype=torch.long)
            
            # Filter by attention mask (positions not in varlen stream should be dropped)
            inner_valid = inner[row_mask[inner] == 1]
            if inner_valid.numel() == 0:
                continue

            # Map (b, pos) -> flat index -> varlen index
            flat_pos = b * S + inner_valid                               # (Lkeep,)
            # mask_flat[flat_pos] must be 1 here; varlen idx = prefix - 1
            var_idx = prefix[flat_pos] - 1                                # still on device, Long
            varlen_indices_per_batch.append(var_idx)
            varlen_indices.append(var_idx)
        varlen_indices_by_batch.append(varlen_indices)
    if len(varlen_indices_per_batch) == 0:
        return torch.empty(0, dtype=torch.long, device=device), varlen_indices_by_batch

    # Concatenate all batches; these indices correspond to positions on the
    # unpadded [1, total_nnz] sequence (i.e., after unpad + transpose).
    return varlen_indices_per_batch, varlen_indices_by_batch


def build_canvas_mask(
    input_ids: torch.Tensor,
    response_length: int,
    canvas_start_id: int,
    canvas_end_id: int,
) -> torch.Tensor:
    """
    Build canvas_mask identifying tokens inside <|canvas_start|> to <|canvas_end|> spans
    within the response portion.
    
    Args:
        input_ids: (bsz, seqlen) full input sequence (prompt + response)
        response_length: length of the response portion
        canvas_start_id: token ID for <|canvas_start|>
        canvas_end_id: token ID for <|canvas_end|>
    
    Returns:
        canvas_mask: (bsz, response_length) bool tensor.
                     True indicates a token inside a sketch span (excluding canvas_start
                     and canvas_end themselves); these positions' log_probs should not
                     participate in the DAPO probability ratio.
                     
                     Note: <|canvas_start|> and <|canvas_end|> are autonomously generated
                     boundary markers whose log_probs must be kept for policy updates,
                     so the model learns when to enter/exit sketch reasoning.
    """
    bsz, seqlen = input_ids.shape
    device = input_ids.device
    
    # Only look at the response portion of input_ids
    response_ids = input_ids[:, -response_length:]  # (bsz, response_length)
    canvas_mask = torch.zeros(bsz, response_length, dtype=torch.bool, device=device)
    
    for b in range(bsz):
        row = response_ids[b]  # (response_length,)
        starts = (row == canvas_start_id).nonzero(as_tuple=False).squeeze(-1)
        ends = (row == canvas_end_id).nonzero(as_tuple=False).squeeze(-1)
        
        if starts.numel() == 0 or ends.numel() == 0:
            continue
        
        # Greedy matching: for each start, find the nearest end to its right
        i_ptr, j_ptr = 0, 0
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = starts[i_ptr].item()
            while j_ptr < ends.numel() and ends[j_ptr].item() <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = ends[j_ptr].item()
            # Mask the open interval (s_pos, e_pos): only mask inner tokens, not start/end markers.
            # <|canvas_start|> and <|canvas_end|> are model-generated; keep their log_probs.
            if e_pos - s_pos > 1:
                canvas_mask[b, s_pos + 1 : e_pos] = True
            i_ptr += 1
            j_ptr += 1
    
    return canvas_mask


def build_latent_mask(
    input_ids: torch.Tensor,
    response_length: int,
    canvas_start_id: int,
    canvas_end_id: int,
) -> torch.Tensor:
    """
    D-1: Build latent_mask identifying latent token positions in the response.
    
    Under the deferred-activation mechanism, the latent span is (s_pos, e_pos),
    i.e. [s_pos+1, e_pos):
    - s_pos is the <|canvas_start|> position, not a latent (model sees its embedding first)
    - s_pos+1 is the first latent token (<|hylar|>, embedding overridden by hidden_state)
    - e_pos is the <|canvas_end|> position, not a latent
    
    Consistent with build_canvas_mask: masks the open interval (s_pos+1, e_pos)
    Consistent with collect_varlen_segment_indices: range(s_pos+1, e_pos)
    
    Args:
        input_ids: (bsz, seqlen) full input sequence (prompt + response)
        response_length: length of the response portion
        canvas_start_id: token ID for <|canvas_start|>
        canvas_end_id: token ID for <|canvas_end|>
    
    Returns:
        latent_mask: (bsz, response_length) bool tensor.
                     True indicates a latent token position (used for D-1 decoupled PPO loss).
    """
    bsz, seqlen = input_ids.shape
    device = input_ids.device
    
    response_ids = input_ids[:, -response_length:]  # (bsz, response_length)
    latent_mask = torch.zeros(bsz, response_length, dtype=torch.bool, device=device)
    
    for b in range(bsz):
        row = response_ids[b]
        starts = (row == canvas_start_id).nonzero(as_tuple=False).squeeze(-1)
        ends = (row == canvas_end_id).nonzero(as_tuple=False).squeeze(-1)
        
        if starts.numel() == 0 or ends.numel() == 0:
            continue
        
        i_ptr, j_ptr = 0, 0
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = starts[i_ptr].item()
            while j_ptr < ends.numel() and ends[j_ptr].item() <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = ends[j_ptr].item()
            # Mask (s_pos, e_pos) i.e. [s_pos+1, e_pos): excludes start_id and end_id positions
            if e_pos > s_pos + 1:
                latent_mask[b, s_pos + 1 : e_pos] = True
            i_ptr += 1
            j_ptr += 1
    
    return latent_mask


def compute_latent_log_probs(latent_poss, latents, last_hidden_state, kappa=0.01):
    """
    Compute log-prob under a vMF (von Mises-Fisher) distribution:
        log p(z | μ, κ) = κ · cos(z, μ) + const
    WITH L2 normalization — μ and z are normalized to unit vectors before computing cosine similarity.
    This ensures log_prob ∈ [-κ, κ], making the PPO ratio numerically stable.
    The normalization constant C_D(κ) is omitted as it cancels in the PPO ratio.

    Args:
        latent_poss: 1D LongTensor/list of positions for latent tokens (length L).
        latents:     Tensor of shape [L, D], rollout latents (z) at those positions.
        last_hidden_state: Tensor of shape [B, T, D] or [total_nnz, D], hidden states.
        kappa:       float, concentration parameter of the vMF distribution.

    Returns:
        latent_log_probs: Tensor of shape [L], per-position vMF log-probs.
    """
    if last_hidden_state.dim() == 2:
        mu = last_hidden_state[latent_poss, :]       # padding_free: (total_nnz, D)
    else:
        mu = last_hidden_state[0, latent_poss, :]    # non-padding_free: (B, T, D)
    latents = latents.to(mu)

    # L2 normalization: ensures cos_sim in [-1, 1], thus log_prob in [-kappa, kappa]
    # mu_norm = F.normalize(mu.float(), dim=-1)
    # z_norm = F.normalize(latents.float(), dim=-1)
    # cos_sim = (mu_norm * z_norm).sum(dim=-1)           # [L], ∈ [-1, 1]
    # latent_log_probs = kappa * cos_sim                  # [L], ∈ [-κ, κ]
    cos_sim = (mu.float() * latents.float()).sum(dim=-1)
    latent_log_probs = kappa * cos_sim

    return latent_log_probs.to(mu.dtype)


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float, return_latent_mu: bool = False) -> torch.Tensor:
        """
        Args:
            micro_batch: Dict with input tensors.
            temperature: sampling temperature.
            return_latent_mu: if True, also return hidden state mu at latent positions (for D-2 closed-form KL).

        Returns:
            If return_latent_mu=False: log_probs (bs, response_len)
            If return_latent_mu=True:  (log_probs, latent_mu, latents_aligned)
                latent_mu: (L, D) actor hidden state, may be None
                latents_aligned: (L, D) rollout z vectors aligned with latent_mu, may be None
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        latent_poss = None
        latents = None
        _hylar_mode = bool(os.environ.get("HYLAR_ID", ""))
        if self.config.sampling_strategy == 'hylar' and not self.config.ablate_latent:
            try:
                latent_poss = []
                start_id = int(os.getenv("LATENT_START_ID"))
                end_id = int(os.getenv("LATENT_END_ID"))
                _, varlen_by_batch = collect_varlen_segment_indices(
                    input_ids=micro_batch["input_ids"], # (micro_batch["input_ids"][1]==151666).nonzero() (micro_batch["input_ids"][1]==151667).nonzero()
                    attention_mask=micro_batch["attention_mask"],
                    start_id=start_id, end_id=end_id,
                )

                latents_list, per_sample = [], []
                for i, lat in enumerate(micro_batch['latents']):
                    if lat is not None:
                        t = torch.tensor(lat)  # (steps, D)
                        segments = varlen_by_batch[i] if i < len(varlen_by_batch) else []
                        poss_cnt = sum(v.numel() for v in segments)
                        if t.shape[0]!=poss_cnt:
                            # Get sample details for debugging
                            sample_input_ids = micro_batch['input_ids'][i].tolist() if isinstance(micro_batch['input_ids'], torch.Tensor) else micro_batch['input_ids'][i]
                            # Decode input_ids to text (if tokenizer is available)
                            try:
                                from transformers import AutoTokenizer
                                tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_PATH", ""), trust_remote_code=True)
                                sample_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                            except Exception:
                                sample_text = "[cannot decode, tokenizer unavailable]"
                            
                            # Compute position info for each segment
                            segment_info = []
                            for seg_idx, seg in enumerate(segments):
                                segment_info.append({
                                    "segment_idx": seg_idx,
                                    "positions": seg.tolist() if isinstance(seg, torch.Tensor) else seg,
                                    "length": seg.numel() if isinstance(seg, torch.Tensor) else len(seg)
                                })
                            
                            # Get uid to trace back to the sample in training.json
                            sample_uid = ""
                            if "uid" in micro_batch:
                                sample_uid = str(micro_batch["uid"][i]) if micro_batch["uid"] is not None else ""
                            
                            # Get prompt for quick sample identification
                            sample_prompt = ""
                            if "problem" in micro_batch:
                                sample_prompt = str(micro_batch["problem"][i]) if micro_batch["problem"] is not None else ""
                            
                            # Get ground_truth
                            sample_ground_truth = ""
                            if "ground_truth" in micro_batch:
                                sample_ground_truth = str(micro_batch["ground_truth"][i]) if micro_batch["ground_truth"] is not None else ""
                            
                            mismatch_info = {
                                "uid": sample_uid,
                                "prompt": sample_prompt,
                                "ground_truth": sample_ground_truth,
                                "sample_idx_in_batch": i,
                                "latent_count": t.shape[0],
                                "position_count": poss_cnt,
                                "num_segments": len(segments),
                                "segment_details": segment_info,
                                "decoded_text": sample_text,
                                "latent_shape": list(t.shape),
                                # No longer saving raw input_ids since decoded_text already contains readable text
                            }
                            
                            print(f"[WARNING] A latent segment in a sample has different numbers of latent {t.shape[0]} and latent pad {poss_cnt}. Skip this sample for latent policy gradient computing.")
                            print(f"[DEBUG] Mismatch details: uid={sample_uid}, num_segments={len(segments)}, segment_lengths={[s['length'] for s in segment_info]}")
                            
                            # Save to JSON file
                            import json
                            import time
                            mismatch_log_path = os.path.join(os.getcwd(), "latent_mismatch_samples.json")
                            mismatch_info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            try:
                                # Read existing data
                                if os.path.exists(mismatch_log_path):
                                    with open(mismatch_log_path, "r", encoding="utf-8") as f:
                                        existing_data = json.load(f)
                                else:
                                    existing_data = []
                                # Append new data
                                existing_data.append(mismatch_info)
                                # Write back to file
                                with open(mismatch_log_path, "w", encoding="utf-8") as f:
                                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                                print(f"[INFO] Mismatch sample saved to {mismatch_log_path}")
                            except Exception as e:
                                print(f"[WARNING] Failed to save mismatch sample to JSON: {e}")
                            
                            continue
                        latents_list.append(t)
                        latent_poss.extend(varlen_by_batch[i])
                        per_sample.append((i, t.shape[0], poss_cnt))

                if len(latents_list) > 0 and len(latent_poss) > 0:
                    latent_poss = torch.cat(latent_poss, dim=0)
                    latents = torch.cat(latents_list, dim=0).to(input_ids.device)

                    if latents.shape[0] != latent_poss.shape[0]:
                        print(f"[WARNING] latents.shape[0] != latent_poss.shape[0], per-sample (idx, lat, poss)={per_sample}, total lat={latents.shape[0]}, poss={int(latent_poss.numel())}. Skip this mirco batch for latent policy gradient computing", flush=True)
                        output_hidden_states = False
                        latent_poss = None
                        latents = None
                else:
                    latent_poss = None
                
                if latents is not None and latent_poss is not None:
                    output_hidden_states = True
                else:
                    output_hidden_states = False
            except Exception:
                print(f"[WARNING] Unexpected error before the latent importance sampling. Fall back to vanilla prob computation for this mirco batch.")
                output_hidden_states = False
                pass
        else:
            output_hidden_states = False

        response_length = responses.size(-1)
        latent_mu = None
        latents_aligned = None
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        #breakpoint()
        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            # breakpoint()
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                latent_poss=latent_poss,
                latents=latents,
                output_hidden_states=output_hidden_states, # AXZ
                #return_dict=True # AXZ
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            #breakpoint()
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            if self.config.sampling_strategy == 'hylar' and not self.config.ablate_latent:
                if latents is not None:
                    latent_log_probs = compute_latent_log_probs(latent_poss, latents, output.hidden_states[-1], kappa=self.config.hylar_rl_kappa)
                    log_probs[latent_poss] = latent_log_probs.to(log_probs.dtype)
                    # If latent_mu is requested, extract hidden state.
                    # Note: in update_policy (with grad context), do not detach,
                    # so that D-2 closed-form KL can backpropagate to update model params.
                    if return_latent_mu:
                        # In padding_free mode, hidden_states is 2D (total_nnz, D)
                        _hs = output.hidden_states[-1]
                        if _hs.dim() == 2:
                            _latent_mu_raw = _hs[latent_poss, :]          # padding_free
                        else:
                            _latent_mu_raw = _hs[0, latent_poss, :]      # non-padding_free
                        if torch.is_grad_enabled():
                            latent_mu = _latent_mu_raw  # (L, D) keep gradients
                        else:
                            latent_mu = _latent_mu_raw.detach().clone()  # (L, D)
                        # Also return the filtered latents (aligned with latent_mu) for D-2 closed-form KL
                        latents_aligned = latents.detach().clone()  # (L, D)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            #breakpoint()
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                latent_poss=latent_poss,
                latents=latents,
                output_hidden_states=output_hidden_states,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        # With Decoupled Hybrid PPO + vMF probability modeling + closed-form vMF KL enabled,
        # tokens in sketch spans participate in policy gradient; log_probs are no longer zeroed.

        if return_latent_mu:
            return log_probs, latent_mu, latents_aligned
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto, return_latent_mu: bool = False):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            return_latent_mu: if True, also return a list of latent_mu per micro_batch (for D-2 closed-form KL)

        Returns:
            If return_latent_mu=False: torch.Tensor (log_probs)
            If return_latent_mu=True:  (torch.Tensor, list[Tensor|None]) i.e. (log_probs, latent_mu_list)
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        if self.config.sampling_strategy == "hylar":
            non_tensor_select_keys.append('latents')
        
        # Include uid and problem for mismatch sample tracing
        if "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('uid')
        if "problem" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('problem')
        if "ground_truth" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('ground_truth')
        #
        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        latent_mu_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=5 if self.config.sampling_strategy == "mc" else 2)
        #breakpoint()
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if return_latent_mu:
                log_probs, latent_mu, _latents_aligned = self._forward_micro_batch(model_inputs, temperature=temperature, return_latent_mu=True)
                latent_mu_lst.append(latent_mu)
            else:
                log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        if return_latent_mu:
            # Merge all non-None latent_mu tensors
            valid_mus = [mu for mu in latent_mu_lst if mu is not None]
            if len(valid_mus) > 0:
                all_latent_mu = torch.cat(valid_mus, dim=0)  # (total_L, D)
            else:
                all_latent_mu = None
            return log_probs, all_latent_mu
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []
        
        if self.config.sampling_strategy == "hylar":
            non_tensor_select_keys.append('latents')
        
        # Include uid and problem for mismatch sample tracing
        if "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('uid')
        if "problem" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('problem')
        if "ground_truth" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append('ground_truth')

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        #breakpoint()
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        #breakpoint()
        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=6 if self.config.sampling_strategy == "mc" else 2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=7 if self.config.sampling_strategy == "mc" else 3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # ===== Forward =====
                    # D-2 needs latent_mu_actor (actor's current forward hidden state) for closed-form vMF KL
                    _is_hylar = (self.config.sampling_strategy == 'hylar' 
                                 and not self.config.ablate_latent)
                    _need_latent_mu = (_is_hylar and self.config.enable_latent_vmf_kl)
                    
                    if _need_latent_mu:
                        log_probs, latent_mu_actor, latents_aligned_for_kl = self._forward_micro_batch(
                            model_inputs, temperature=temperature, return_latent_mu=True
                        )
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        latent_mu_actor = None
                        latents_aligned_for_kl = None
                    
                    # With Decoupled Hybrid PPO enabled, tokens in sketch spans participate in
                    # policy gradient; sketch positions in response_mask are no longer zeroed.

                    # Entropy (for monitoring only, not part of loss)
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)

                    # ===== D-1: Decoupled Hybrid PPO (separate token/latent surrogate loss + dual clip) =====
                    if _is_hylar and self.config.enable_decoupled_hybrid_ppo:
                        # Build latent_mask: identify latent positions along the response dimension
                        canvas_start_id = int(os.environ.get("LATENT_START_ID", "151666"))
                        canvas_end_id = int(os.environ.get("LATENT_END_ID", "151667"))
                        latent_mask = build_latent_mask(
                            model_inputs["input_ids"], response_length, canvas_start_id, canvas_end_id
                        )  # (bsz, response_length) bool tensor, True = latent position
                        
                        # Build token_mask and lat_mask
                        latent_mask_float = latent_mask.to(response_mask.dtype)
                        token_mask = response_mask * (1.0 - latent_mask_float)  # non-latent positions
                        lat_mask = response_mask * latent_mask_float             # latent positions
                        
                        # Token PPO loss (using token clip parameters)
                        pg_loss_tok, pg_clipfrac_h_tok, pg_clipfrac_l_tok, ppo_kl_tok = core_algos.compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=token_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                        
                        # Latent PPO loss (using latent-specific clip parameters, typically tighter)
                        if lat_mask.sum() > 0:
                            pg_loss_lat, pg_clipfrac_h_lat, pg_clipfrac_l_lat, ppo_kl_lat = core_algos.compute_policy_loss(
                                old_log_probs=old_log_probs,
                                log_probs=log_probs,
                                advantages=advantages,
                                response_mask=lat_mask,
                                clip_ratio_low=self.config.latent_clip_ratio_low,
                                clip_ratio_high=self.config.latent_clip_ratio_high,
                                clip_ratio_dual=self.config.latent_clip_ratio_dual,
                            )
                        else:
                            # No latent positions (no sketch spans in this micro_batch)
                            pg_loss_lat = torch.tensor(0.0, device=log_probs.device)
                            pg_clipfrac_h_lat = torch.tensor(0.0, device=log_probs.device)
                            pg_clipfrac_l_lat = torch.tensor(0.0, device=log_probs.device)
                            ppo_kl_lat = torch.tensor(0.0, device=log_probs.device)
                        
                        # Combine: pg_loss = pg_loss_tok + alpha * pg_loss_lat
                        pg_loss = pg_loss_tok + self.config.latent_loss_alpha * pg_loss_lat
                        pg_clipfrac_higher = pg_clipfrac_h_tok  # use token part for metrics
                        pg_clipfrac_lower = pg_clipfrac_l_tok
                        ppo_kl = ppo_kl_tok
                        
                        # Record D-1 decoupled metrics
                        append_to_dict(metrics, {
                            "actor/pg_loss_tok": pg_loss_tok.detach().item(),
                            "actor/pg_loss_lat": pg_loss_lat.detach().item(),
                            "actor/clipfrac_h_tok": pg_clipfrac_h_tok.detach().item(),
                            "actor/clipfrac_h_lat": pg_clipfrac_h_lat.detach().item(),
                            "actor/clipfrac_l_tok": pg_clipfrac_l_tok.detach().item(),
                            "actor/clipfrac_l_lat": pg_clipfrac_l_lat.detach().item(),
                            "actor/ppo_kl_tok": ppo_kl_tok.detach().item(),
                            "actor/ppo_kl_lat": ppo_kl_lat.detach().item(),
                            "actor/latent_ratio": lat_mask.sum().item() / max(response_mask.sum().item(), 1.0),
                        })
                    else:
                        # Original logic: unified token + latent PPO loss
                        pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                    
                    # ===== KL Penalty =====
                    # Standard PPO KL penalty (optional, controlled by disable_kl flag)
                    if not self.config.disable_kl and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        
                        if _is_hylar and self.config.enable_decoupled_hybrid_ppo:
                            # D-1 mode: only compute sample-based KL for token positions (latent uses closed-form KL or none)
                            kld = core_algos.compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_loss_tok = VF.masked_mean(kld, token_mask)
                            pg_loss = pg_loss + kl_loss_tok * self.config.kl_coef
                            append_to_dict(metrics, {
                                "actor/kl_loss_tok": kl_loss_tok.detach().item(),
                                "actor/kl_coef": self.config.kl_coef,
                            })
                        else:
                            # Original logic: unified KL computation
                            kld = core_algos.compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_loss = VF.masked_mean(kld, response_mask)
                            pg_loss = pg_loss + kl_loss * self.config.kl_coef
                            append_to_dict(metrics, {
                                "actor/kl_loss": kl_loss.detach().item(),
                                "actor/kl_coef": self.config.kl_coef
                            })
                    
                    # ===== D-2: Closed-form vMF KL constraint =====
                    # Uses kappa * (1 - cos(mu_actor_new, z_rollout)) as a latent distribution drift constraint.
                    # z_rollout is the unit vector generated from old policy vMF(mu_old, kappa) during rollout.
                    if _is_hylar and self.config.enable_latent_vmf_kl and latent_mu_actor is not None and latents_aligned_for_kl is not None:
                        # latent_mu_actor: (L, D) hidden state from current forward (has gradients)
                        # latents_aligned_for_kl: (L, D) rollout z vectors (unit vectors) aligned with latent_mu_actor
                        # Both have been filtered through the same mismatch skip logic in _forward_micro_batch
                        z_rollout = latents_aligned_for_kl.to(latent_mu_actor.device).to(latent_mu_actor.dtype)
                        if z_rollout.shape[0] == latent_mu_actor.shape[0] and z_rollout.shape[0] > 0:
                            latent_vmf_kl = core_algos.compute_latent_vmf_kl(
                                mu_actor=latent_mu_actor,
                                mu_ref=z_rollout.detach(),
                                kappa=self.config.hylar_rl_kappa,
                            )
                            pg_loss = pg_loss + latent_vmf_kl * self.config.latent_kl_coef
                            append_to_dict(metrics, {
                                "actor/latent_vmf_kl": latent_vmf_kl.detach().item(),
                                "actor/latent_kl_coef": self.config.latent_kl_coef,
                            })

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
