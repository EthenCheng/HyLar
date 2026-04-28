#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

export PYTHONUNBUFFERED=1
# Set your own WandB API key (or remove to disable WandB logging)
# export WANDB_API_KEY=your_wandb_api_key
export hylar_DEBUG=1

unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=16
export RAY_NUM_GPUS=8
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
export RAY_METRICS_EXPORT_PORT=0
export RAY_LOG_TO_STDERR=0
export RAY_LOCAL_MODE=0
export RAY_task_exit_on_oom=1
export RAY_SPILL_DIR=/tmp/ray_spill
export RAY_TMPDIR=/tmp/ray_tmp
mkdir -p /tmp/ray_spill /tmp/ray_tmp
export RAY_OBJECT_STORE_MEMORY=107374182400
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300

# API endpoint for the judge model (GPT-5)
export OPENAI_BASE_URL="https://api.openai.com/v1"
# Set your OpenAI API key here or export it beforehand
# export OPENAI_API_KEY="sk-..."

HYLAR_RL_PATCH=1  # overwrite the transformers and vllm forward module
MODEL_PATH=path/to/your/hylar/checkpoint

latent_size=32
export LATENT_SIZE=${latent_size}
export HYLAR_ID=151665

ROLLOUT_N=8
TEMPERATURE=0.9
GPU_UTILIZATION=0.9
KL_COEF=0.05
ROLLOUT_BATCH_SIZE=64
TRAIN_MAX_SAMPLES=49664
VAL_MAX_SAMPLES=800
N_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=1
HYLAR_RL_KAPPA=0.01
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=2048

## Online Filtering Parameters
ONLINE_FILTERING=true
FILTER_KEY=accuracy
FILTER_LOW=0.1
FILTER_HIGH=0.9
ANSWER_TAG_FILTERING=false

python -m verl.trainer.main \
    config=examples/config_hylar.yaml \
    data.train_files=path/to/your/train/data \
    data.val_files=path/to/your/val/data.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=HyLar_RL_training \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.sampling_strategy=hylar \
    worker.rollout.max_num_seqs=128 \
    worker.rollout.max_num_batched_tokens=65536 \
    worker.reward.reward_function=./examples/reward_function/hylar_reward_function.py:compute_score_w_prev_correctness \
    worker.reward.repetition_penalty=false \
    worker.rule_based_judge.judge_function=./examples/reward_function/hylar_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name=gpt-5 \
    worker.rule_based_judge.api_url=https://api.openai.com/v1 \
    worker.rule_based_judge.api_key=${OPENAI_API_KEY} \
    worker.actor.hylar_rl_kappa=${HYLAR_RL_KAPPA} \
    worker.ref.hylar_rl_kappa=${HYLAR_RL_KAPPA} \
    algorithm.kl_coef=${KL_COEF} \
    algorithm.online_filtering=${ONLINE_FILTERING} \
    algorithm.filter_key=${FILTER_KEY} \
    algorithm.filter_low=${FILTER_LOW} \
    algorithm.filter_high=${FILTER_HIGH} \
    algorithm.answer_tag_filtering=${ANSWER_TAG_FILTERING} \
    algorithm.enable_decoupled_hybrid_ppo=true \
    algorithm.latent_clip_ratio_low=0.1 \
    algorithm.latent_clip_ratio_high=0.1 \
    algorithm.latent_clip_ratio_dual=3.0 \
    algorithm.latent_loss_alpha=0.5 \
    algorithm.enable_latent_vmf_kl=true \
    algorithm.latent_kl_coef=1e-2 \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.dataloader_num_workers=32 \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH}
