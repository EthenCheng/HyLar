import os
import uuid
import json
import datetime
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type, Set

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager, FunctionRuleBasedJudgeManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from tools.api_judge import api_batch_judge
from tools.custom_api import build_deepseek_client, build_gemini_client

import random
from torch.utils.data import Dataset    
from ..utils.dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
#from tools.compute_embeds import compute_embeds_fn
from tools.actors import StepHashServer, SampleHashServer
from tools.actors import EmbedServer
import matplotlib.pyplot as plt
import re

def replace_sketch_token_content(s: str) -> str:
    pattern = re.compile(r'(<\|canvas_start\|>)(.*?)(<\|canvas_end\|>)', flags=re.DOTALL)
    return pattern.sub(r'\1<canvas>\3', s)

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    DAPO = "dapo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(config: PPOConfig, data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, sampling_strategy: str = "greedy") -> DataProto:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if sampling_strategy in ["greedy"]:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
        elif sampling_strategy in ["hylar"]:
            advantages, returns = core_algos.compute_grpo_latent_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.DAPO:
        if sampling_strategy in ["greedy"]:
            advantages, returns = core_algos.compute_dapo_outcome_advantage(token_level_rewards, response_mask, index)
        elif sampling_strategy in ["hylar"]:
            advantages, returns = core_algos.compute_dapo_latent_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
        rule_based_judge: Optional[FunctionRuleBasedJudgeManager] = None,
        #embed_model: Optional[torch.nn.Module] = None,
        #embed_tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.rule_based_judge = rule_based_judge
        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")


        if self.config.data.pr_batch_size != -1:
            if config.data.pr_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.pr_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )
            
        else:
            if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by actor global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
                )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO, AdvantageEstimator.DAPO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO, RLOO and DAPO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")
        
        self.base_dataset = self.train_dataloader.dataset

        if self.config.worker.rollout.sampling_strategy in ["hylar"]:
            self.sample_hash_server_main = SampleHashServer.options(
                name="sample_hash_server_main"
            ).remote()
            #print("type of self.sample_hash_server_main:", type(self.sample_hash_server_main))
            self.config.worker.rollout.hylar.hash_server_name = "sample_hash_server_main"
            ray.get(self.sample_hash_server_main.ping.remote())

        if "api" in self.config.worker.rule_based_judge.judge_function_name:
            if self.config.worker.rule_based_judge.api_name in ['deepseek-chat', 'deepseek']:
                self.client = build_deepseek_client()
            elif self.config.worker.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.client = build_gemini_client()
            elif self.config.worker.rule_based_judge.api_name == 'gpt-4o-mini':
                self.client = None  # gpt-4o-mini uses a unified HTTP API, no separate client needed
            elif self.config.worker.rule_based_judge.api_name == 'gpt-5':
                self.client = None  # gpt-5 uses a remote proxy HTTP API, no separate client needed
            elif self.config.worker.rule_based_judge.api_name == 'Qwen3_32B_local':
                self.client = None  # Qwen3_32B_local uses a locally deployed vLLM OpenAI-compatible API
            else:
                self.client = None
                raise ValueError(f"API {self.config.worker.rule_based_judge.api_name} not supported.")
        
        # Sample saving configuration
        self.save_samples = config.trainer.save_samples
        self.samples_save_dir = config.trainer.samples_save_dir
        self.samples_save_interval = config.trainer.samples_save_interval
        
        # Create sample saving directory
        if self.save_samples:
            os.makedirs(self.samples_save_dir, exist_ok=True)
            print(f"Rollout samples will be saved to: {self.samples_save_dir}")
            
            # Create validation sample saving directory
            self.validation_samples_dir = os.path.join(os.path.dirname(self.samples_save_dir), "validation_samples")
            os.makedirs(self.validation_samples_dir, exist_ok=True)
            print(f"Validation samples will be saved to: {self.validation_samples_dir}")
        
        # Rollout round counter (to distinguish each rollout round)
        self._current_rollout_round = 0
        # Accumulate samples from all rounds in each step
        self._step_round_samples = []

    # Strict response format regex:
    # <think>...</think> + one or more (<|canvas_start|><canvas><|canvas_end|><think>...</think>) + <answer>...</answer>
    # Whitespace (newlines, spaces, etc.) is allowed between tags
    _VALID_RESPONSE_FORMAT = re.compile(
        r'^\s*<think>.*?</think>'
        r'(\s*<\|canvas_start\|><canvas><\|canvas_end\|>\s*<think>.*?</think>)+'
        r'\s*<answer>.*?</answer>\s*$',
        re.DOTALL
    )

    def _filter_by_answer_tag(self, batch: DataProto) -> DataProto:
        """Filter out groups containing any response with an invalid format.
        
        Valid format:
          <think>...</think>
          (<|canvas_start|><canvas><|canvas_end|><think>...</think>)+
          <answer>...</answer>
        i.e., starts with <think>, ends with <answer>, with one or more sketch+think cycles in between.
        
        If any response in a group does not match this format, the entire group is discarded.
        
        Args:
            batch: DataProto after accuracy filtering
            
        Returns:
            Filtered DataProto
        """
        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        response_length = response_mask.sum(dim=-1)
        uids = batch.non_tensor_batch["uid"]
        
        invalid_uids = set()
        total_invalid_responses = 0
        
        for i in range(len(batch)):
            valid_response_ids = response_ids[i][:int(response_length[i].item())]
            response_text = replace_sketch_token_content(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            ).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            
            if not self._VALID_RESPONSE_FORMAT.match(response_text):
                invalid_uids.add(uids[i])
                total_invalid_responses += 1
        
        if len(invalid_uids) == 0:
            print(f"Format filter: all groups passed, no filtering needed.")
            return batch
        
        unique_uids = set(uids)
        num_groups_before = len(unique_uids)
        
        kept_idxs = [idx for idx, uid in enumerate(uids) if uid not in invalid_uids]
        num_groups_after = num_groups_before - len(invalid_uids)
        
        print(f"Format filter: {total_invalid_responses} invalid responses in {len(invalid_uids)} groups. "
              f"Kept {num_groups_after}/{num_groups_before} groups ({len(kept_idxs)}/{len(batch)} samples).")
        
        if len(kept_idxs) == 0:
            print("Warning: Format filter removed ALL groups. Keeping original batch to avoid empty data.")
            return batch
        
        return batch[kept_idxs]

    def _extract_sample_data(self, batch: DataProto, index: int) -> dict:
        """Extract data for a single sample from the batch.
        
        Args:
            batch: DataProto containing sample data
            index: Index of the sample in the batch
            
        Returns:
            Dictionary containing sample information
        """
        sample_data = {}
        
        response_ids = batch.batch.get("responses", None)
        prompt_ids = batch.batch.get("input_ids", None)
        response_mask = batch.batch.get("response_mask", None)
        non_tensor_batch = batch.non_tensor_batch if hasattr(batch, 'non_tensor_batch') else {}
        
        # Save uid
        if "uid" in non_tensor_batch:
            sample_data["uid"] = str(non_tensor_batch["uid"][index])
        else:
            sample_data["uid"] = ""
        
        # Save prompt
        if "problem" in non_tensor_batch:
            sample_data["prompt"] = non_tensor_batch["problem"][index]
        elif prompt_ids is not None:
            prompt_text = self.tokenizer.decode(prompt_ids[index], skip_special_tokens=False)
            sample_data["prompt"] = prompt_text
        else:
            sample_data["prompt"] = ""
        
        # Save response
        if response_ids is not None and response_mask is not None:
            response_length = response_mask[index].sum().item()
            valid_response_ids = response_ids[index][:int(response_length)]
            response_text = replace_sketch_token_content(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            )
            response_text = response_text.replace("<|endoftext|>", "").replace("<|im_end|>", "")
            sample_data["response"] = response_text
            sample_data["token_ids"] = valid_response_ids.cpu().tolist()
        else:
            sample_data["response"] = ""
            sample_data["token_ids"] = []
        
        # Save ground truth
        if "ground_truth" in non_tensor_batch:
            sample_data["ground_truth"] = non_tensor_batch["ground_truth"][index]
        else:
            sample_data["ground_truth"] = ""
        
        return sample_data

    def _save_round_samples(self, batch: DataProto, step: int, round_num: int) -> None:
        """Save samples from a single rollout round to a JSON file, organized by group.
        
        Samples from each rollout round are saved to step{X}_round_{Y}.json,
        grouped by uid with all samples in each group together.
        
        Args:
            batch: DataProto containing sample data (with scoring info)
            step: Current training step
            round_num: Current rollout round number
        """
        if not self.save_samples or step % self.samples_save_interval != 0:
            return
            
        try:
            non_tensor_batch = batch.non_tensor_batch if hasattr(batch, 'non_tensor_batch') else {}
            
            # Group samples by uid
            uid_to_samples = defaultdict(list)
            for i in range(len(batch)):
                sample_data = self._extract_sample_data(batch, i)
                
                # Add scoring info
                if "correctness" in non_tensor_batch:
                    sample_data["correctness"] = float(non_tensor_batch["correctness"][i])
                
                # Add token_level_scores if available
                if "token_level_scores" in batch.batch:
                    scores = batch.batch["token_level_scores"]
                    sample_data["reward_score"] = float(scores[i].sum().item())
                
                uid = sample_data["uid"]
                uid_to_samples[uid].append(sample_data)
            
            # Build group-organized data structure
            groups = []
            for uid, samples in uid_to_samples.items():
                group_data = {
                    "uid": uid,
                    "prompt": samples[0]["prompt"] if samples else "",
                    "ground_truth": samples[0]["ground_truth"] if samples else "",
                    "samples": samples
                }
                # Compute average score per group
                if samples and "correctness" in samples[0]:
                    group_data["avg_correctness"] = sum(s.get("correctness", 0) for s in samples) / len(samples)
                if samples and "reward_score" in samples[0]:
                    group_data["avg_reward_score"] = sum(s.get("reward_score", 0) for s in samples) / len(samples)
                groups.append(group_data)
            
            # Prepare data for saving
            samples_data = {
                "step": step,
                "round": round_num,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_samples": len(batch),
                "total_groups": len(groups),
                "groups": groups
            }
            
            # Save to file
            filename = f"step{step}_round_{round_num}.json"
            filepath = os.path.join(self.samples_save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(batch)} samples ({len(groups)} groups) to {filepath}")
            
        except Exception as e:
            print(f"Error saving round samples: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_training_samples(self, batch: DataProto, step: int) -> None:
        """Save final training samples to a JSON file.
        
        The final training groups are saved to step{X}_training.json.
        
        Args:
            batch: DataProto of samples used for training
            step: Current training step
        """
        if not self.save_samples or step % self.samples_save_interval != 0:
            return
            
        try:
            non_tensor_batch = batch.non_tensor_batch if hasattr(batch, 'non_tensor_batch') else {}
            
            # Group samples by uid
            uid_to_samples = defaultdict(list)
            for i in range(len(batch)):
                sample_data = self._extract_sample_data(batch, i)
                
                # Add scoring info
                if "correctness" in non_tensor_batch:
                    sample_data["correctness"] = float(non_tensor_batch["correctness"][i])
                
                # Add token_level_scores
                if "token_level_scores" in batch.batch:
                    scores = batch.batch["token_level_scores"]
                    sample_data["reward_score"] = float(scores[i].sum().item())
                
                # Add advantages if available
                if "advantages" in batch.batch:
                    advantages = batch.batch["advantages"]
                    response_mask = batch.batch.get("response_mask", None)
                    if response_mask is not None:
                        # Compute masked mean advantage
                        mask = response_mask[i]
                        adv = advantages[i]
                        valid_adv = adv[mask.bool()]
                        sample_data["avg_advantage"] = float(valid_adv.mean().item()) if len(valid_adv) > 0 else 0.0
                    else:
                        sample_data["avg_advantage"] = float(advantages[i].mean().item())
                
                uid = sample_data["uid"]
                uid_to_samples[uid].append(sample_data)
            
            # Build group-organized data structure
            groups = []
            for uid, samples in uid_to_samples.items():
                group_data = {
                    "uid": uid,
                    "prompt": samples[0]["prompt"] if samples else "",
                    "ground_truth": samples[0]["ground_truth"] if samples else "",
                    "samples": samples
                }
                # Compute group-level statistics
                if samples and "correctness" in samples[0]:
                    group_data["avg_correctness"] = sum(s.get("correctness", 0) for s in samples) / len(samples)
                if samples and "reward_score" in samples[0]:
                    group_data["avg_reward_score"] = sum(s.get("reward_score", 0) for s in samples) / len(samples)
                if samples and "avg_advantage" in samples[0]:
                    group_data["avg_advantage"] = sum(s.get("avg_advantage", 0) for s in samples) / len(samples)
                groups.append(group_data)
            
            # Prepare data for saving
            samples_data = {
                "step": step,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_samples": len(batch),
                "total_groups": len(groups),
                "groups": groups
            }
            
            # Save to file
            filename = f"step{step}_training.json"
            filepath = os.path.join(self.samples_save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved training samples: {len(batch)} samples ({len(groups)} groups) to {filepath}")
            
        except Exception as e:
            print(f"Error saving training samples: {e}")
            import traceback
            traceback.print_exc()

    def _save_rollout_samples(self, batch: DataProto, step: int, stage: str = "rollout") -> None:
        """[Deprecated] Legacy saving method kept for backward compatibility.
        
        Use the following methods instead:
        - _save_round_samples: save samples from each rollout round
        - _save_training_samples: save final training samples
        """
        pass

    def _save_validation_samples(
        self, 
        step: int,
        inputs: List[str], 
        outputs: List[str], 
        labels: List[str], 
        scores: List[float],
        correctness: List[float] = None,
        reward_metrics: Dict[str, List[float]] = None,
        token_ids: List[List[int]] = None
    ) -> None:
        """Save validation samples to a JSON file.
        
        Results from each validation round are saved to step{X}_validation.json.
        
        Args:
            step: Current training step
            inputs: List of input texts
            outputs: List of model output texts
            labels: List of ground truth answers
            scores: List of reward scores
            correctness: List of correctness judgments (0 or 1)
            reward_metrics: Dictionary of reward metric values
            token_ids: List of output token id sequences per sample
        """
        if not self.save_samples:
            return
            
        try:
            import json
            import datetime
            
            # Build sample data
            samples = []
            for i in range(len(inputs)):
                sample_data = {
                    "index": i,
                    "input": inputs[i],
                    "output": outputs[i],
                    "ground_truth": labels[i],
                    "reward_score": scores[i] if i < len(scores) else None,
                }
                if correctness is not None and i < len(correctness):
                    sample_data["correctness"] = correctness[i]
                # Add token_ids
                if token_ids is not None and i < len(token_ids):
                    sample_data["token_ids"] = token_ids[i]
                samples.append(sample_data)
            
            # Compute statistics
            stats = {
                "total_samples": len(samples),
                "avg_reward_score": sum(scores) / len(scores) if scores else 0,
            }
            if correctness:
                stats["accuracy"] = sum(correctness) / len(correctness) if correctness else 0
                stats["correct_count"] = sum(correctness)
                stats["incorrect_count"] = len(correctness) - sum(correctness)
            
            # Prepare data for saving
            validation_data = {
                "step": step,
                "timestamp": datetime.datetime.now().isoformat(),
                "stats": stats,
                "samples": samples
            }
            
            # Save to file
            filename = f"step{step}_validation.json"
            filepath = os.path.join(self.validation_samples_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved validation samples: {len(samples)} samples to {filepath}")
            
        except Exception as e:
            print(f"Error saving validation samples: {e}")
            import traceback
            traceback.print_exc()

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # If val_generations_to_log is -1, log all samples; otherwise log the specified number
        if self.config.trainer.val_generations_to_log > 0:
            samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores, sample_token_ids = [], [], [], [], []
        reward_metrics_lst = defaultdict(list)
        
        # Track Ray object references
        ray_object_refs = []
        
        # Track correctness for accuracy computation
        correctness_list = []
        
        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # Use skip_special_tokens=True to filter out padding and image_pad tokens
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "global_index"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch.meta_info["mode"] = "test"
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]

            # Decode all tokens (including latent regions), then replace garbled content via replace_sketch_token_content
            output_texts = [replace_sketch_token_content(self.tokenizer.decode(ids, skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "") for ids in output_ids]

            # Collect token ids (use response_mask to extract valid tokens, consistent with rollout)
            response_mask = test_output_gen_batch.batch.get("response_mask", None)
            for idx_i, ids in enumerate(output_ids):
                if response_mask is not None:
                    valid_len = int(response_mask[idx_i].sum().item())
                    sample_token_ids.append(ids[:valid_len].tolist())
                else:
                    # fallback: strip trailing pad tokens
                    ids_list = ids.tolist()
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 151643
                    while ids_list and ids_list[-1] == pad_id:
                        ids_list.pop()
                    sample_token_ids.append(ids_list)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # Add ref_resp_lengths if not present (needed for reward function)
            if "ref_resp_lengths" not in test_batch.non_tensor_batch:
                # Set to zeros as default (no length penalty in validation)
                batch_size = len(test_batch.batch)
                test_batch.non_tensor_batch["ref_resp_lengths"] = np.zeros(batch_size, dtype=np.float32)

            # evaluate using reward_function
            if 'api' in self.config.worker.rule_based_judge.judge_function_name:
                #breakpoint()
                correctness_batch = api_batch_judge(
                    questions=test_batch.non_tensor_batch["problem"].tolist(),
                    preds=output_texts,
                    gts=test_batch.non_tensor_batch["ground_truth"].tolist(),
                    api_name=self.config.worker.rule_based_judge.api_name,
                    api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                    client=self.client,
                    repetition_penalty=self.config.worker.reward.repetition_penalty,
                    api_url=self.config.worker.rule_based_judge.api_url,
                    api_key=self.config.worker.rule_based_judge.api_key,
                )
                #correctness_list = ray.get(self.rule_based_judge.judge.remote(output_texts, test_batch.non_tensor_batch["ground_truth"].tolist()))
                test_batch.non_tensor_batch["correctness"] = correctness_batch
                # Collect correctness for accuracy computation
                correctness_list.extend(correctness_batch)
            
            # Get reward and process immediately without retaining Ray object references
            reward_ref = self.val_reward_fn.compute_reward.remote(test_batch)
            ray_object_refs.append(reward_ref)
            reward_tensor, reward_metrics = ray.get(reward_ref)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor.cpu())  # Move to CPU to free GPU memory
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            
            # Clean up variables to free memory
            del test_batch, test_gen_batch, test_output_gen_batch, reward_tensor, output_ids, reward_ref
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Periodically clean up Ray object store
            if (batch_idx + 1) % 2 == 0:
                import gc
                gc.collect()
                # Clear completed Ray object references
                ray_object_refs.clear()

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        
        # Save validation samples to JSON file
        self._save_validation_samples(
            step=self.global_step,
            inputs=sample_inputs,
            outputs=sample_outputs,
            labels=sample_labels,
            scores=sample_scores,
            correctness=correctness_list if len(correctness_list) > 0 else None,
            reward_metrics=reward_metrics_lst,
            token_ids=sample_token_ids
        )
        
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}

        # Compute accuracy: proportion of correct answers
        if len(correctness_list) > 0:
            accuracy = sum(correctness_list) / len(correctness_list)
            val_reward_metrics["val/accuracy"] = accuracy
            print(f"Validation accuracy: {accuracy:.4f} ({sum(correctness_list)}/{len(correctness_list)})")
        else:
            print("Warning: No correctness data available for accuracy calculation")

        # Clean up accumulated validation data
        del reward_tensor_lst, sample_inputs, sample_outputs, sample_labels, sample_scores, reward_metrics_lst, ray_object_refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()

        return {**val_reward_metrics, "val/reward_score": reward_score}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        print('start building worker group')
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
        print('done building worker group')
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

        if self.config.worker.rollout.sampling_strategy in ["hylar"]:
            self.sample_hash_server_main.save_info.remote(filepath=folder_path, overwrite=True)
        

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        self.step_hash_server_main.load_info.remote(self.config.trainer.load_checkpoint_path)

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        """
        Generate a batch of data with online filtering and automatic batch padding.
        This method implements PAPO-style online filtering and automatic batch replenishment.
        """
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        # Reset rollout round counter for this step
        self._current_rollout_round = 0
        print("Start generating batch...")
        
        while True:
            num_try_make_batch += 1
            # Increment rollout round counter for this step
            self._current_rollout_round += 1
            current_round = self._current_rollout_round
            
            # Get next batch from dataloader
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                # Reset iterator when exhausted
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)
            
            # Create DataProto from batch dict
            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            # Build generation batch
            new_batch.meta_info["mode"] = "train_rl_gen"
            gen_batch = self.build_gen_batch(new_batch)
            
            # Generate sequences
            gen_batch.meta_info["mode"] = "train_rl_gen"
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            
            # Handle REMAX baseline if needed
            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                
                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                
                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output
            
            # Assign unique IDs to each prompt
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            
            # Repeat to align with rollout.n responses per prompt
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            
            # Compute rule-based judge to get correctness before reward computation
            if hasattr(self.actor_rollout_wg, 'compute_rule_based_judge'):
                judge_output = self.actor_rollout_wg.compute_rule_based_judge(new_batch)
                # Directly add correctness to non_tensor_batch instead of using union
                if judge_output.non_tensor_batch is not None:
                    for key, value in judge_output.non_tensor_batch.items():
                        new_batch.non_tensor_batch[key] = value
            
            # Online filtering based on reward
            if self.config.algorithm.online_filtering:
                # Compute rewards for filtering
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                # Unlock the TensorDict if it's locked before modifying
                if new_batch.batch.is_locked:
                    new_batch.batch.unlock_()
                new_batch.batch["token_level_scores"] = reward_tensor
                
                # Save current round samples (with judge and reward scores)
                self._save_round_samples(new_batch, self.global_step, current_round)
                
                # Accumulate metrics
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)
                
                # Get filter scores
                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                assert len(filter_scores) != 0, "Filter scores should not be empty."
                
                # Group by uid (same prompt)
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)
                
                # Compute mean score for each group
                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                
                # Filter: keep groups with mean score in [filter_low, filter_high]
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                
                # Get indices of kept samples
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                
                # Safety: if all filtered out, keep all
                if len(kept_sample_idxs) == 0:
                    print("Warning: All samples filtered out, keeping all samples.")
                    kept_sample_idxs = list(range(len(uids)))
                
                # Filter the batch
                new_batch = new_batch[kept_sample_idxs]
                print(f"Accuracy filtered: kept {len(kept_uids)} groups out of {len(uid2mean)} groups")
            else:
                # Save current round samples even without online_filtering (judge scores only)
                self._save_round_samples(new_batch, self.global_step, current_round)
            
            # Format filtering: discard entire group if any response has invalid format
            # Valid format: <think>...</think> + 1+ (sketch+think) cycles + <answer>...</answer>
            if self.config.algorithm.answer_tag_filtering and len(new_batch) > 0:
                new_batch = self._filter_by_answer_tag(new_batch)
            
            # Concatenate with existing batch
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            
            # Check if we have enough samples
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            
            if current_batch_size < rollout_batch_size:
                # Need more samples
                if len(batch) == 0:
                    print("Warning: Generated batch is empty, continuing to generate more data...")
                    continue
                
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                    # Continue to next iteration to generate more
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                # Have enough samples
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                
                # Update metrics if online filtering was used
                if self.config.algorithm.online_filtering:
                    from .metrics import reduce_metrics
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                
                # Truncate to exact required sample count
                final_batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                
                # Save final training samples
                self._save_training_samples(final_batch, self.global_step)
                
                # Return exactly rollout_batch_size * rollout.n samples
                return final_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        # Initialize data iterator for _make_batch_data
        self.data_iterator = iter(self.train_dataloader)
        
        # Main training loop
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            
            with timer("step", timing_raw):
                # Generate a batch with online filtering and auto-replenishment
                with timer("gen", timing_raw):
                    batch = self._make_batch_data(metrics=metrics)

                # Balance the number of valid tokens on each dp rank
                self._balance_batch(batch, metrics=metrics)

                # Compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Update model with the generated batch
                self.post_generate_update(metrics, timing_raw, batch)

            # Collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()


        
    def post_generate_process(self, metrics, timing_raw, batch, gen_batch, gen_batch_output):

        if self.config.algorithm.adv_estimator == "remax":
            with timer("gen_max", timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
        batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)

        batch = batch.union(gen_batch_output)
        batch.non_tensor_batch.pop("multi_modal_data", None)

        # balance the number of valid tokens on each dp rank.
        # Note that this breaks the order of data inside the batch.
        # Please take care when you implement group based adv computation such as GRPO and rloo
        self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()    

        return batch

    def post_generate_update(self, metrics, timing_raw, batch):
        with timer("reward", timing_raw):
            # batch.non_tensor_batch should have "correctness" here
            reward_ref = self.reward_fn.compute_reward.remote(batch)

        # recompute old_log_probs
        #breakpoint()
        with timer("old", timing_raw):
            old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
            batch = batch.union(old_log_probs)

        # compute ref_log_probs
        if self.use_reference_policy:
            with timer("ref", timing_raw):
                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                batch = batch.union(ref_log_probs)

        # compute values
        if self.use_critic:
            with timer("values", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        #breakpoint()
        with timer("adv", timing_raw):
            # get token level scores
            reward_tensor, reward_metrics = ray.get(reward_ref)
            batch.batch["token_level_scores"] = reward_tensor
            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
            metrics.update(reward_metrics)

            # apply kl penalty if available
            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                # apply kl penalty to reward
                batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process
            batch = compute_advantage(
                self.config,
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                sampling_strategy=self.config.worker.rollout.sampling_strategy
            )

        # update critic
        if self.use_critic:
            with timer("update_critic", timing_raw):
                critic_output = self.critic_wg.update_critic(batch)

            critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
            metrics.update(critic_metrics)

        # update actor
        #breakpoint()
        if self.config.trainer.critic_warmup <= self.global_step:
            with timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)

            actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
            metrics.update(actor_metrics)

        # validate
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and self.global_step % self.config.trainer.val_freq == 0
        ):
            with timer("validation", timing_raw):
                val_metrics = self._validate()

            metrics.update(val_metrics)

        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
            with timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

    def build_gen_batch(self, batch: DataProto) -> None:
        if "multi_modal_data" in batch.non_tensor_batch.keys():
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "global_index", "ground_truth", "problem"]
            )
        else:
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "global_index", "ground_truth", "problem"],
            )
        return gen_batch

    @staticmethod
    def split_solution_into_steps(solution: str, delim: str = "### Step") -> List[List[str]]:
        steps = solution.split(delim)
        steps = [re.sub(r"^ \d+(\.\d+)?: ", "", step).strip() for step in steps]
        return steps
