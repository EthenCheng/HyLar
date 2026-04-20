#!/bin/bash

# Set environment variables
export PYTHONPATH=$(pwd)/SFT
# Set visible GPUs (adjust based on your hardware)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model configuration
MODEL_PATH="path/to/your/hylar/checkpoint"

# Dataset name
DATASET_NAME="V_star"
#DATASET_NAME="HRBench_4K"
#DATASET_NAME="HRBench_8K"
#DATASET_NAME="MMStar"
#DATASET_NAME="MMVP"

# Image token configuration
MIN_TOKEN=128
MAX_TOKEN=16384

# Maximum number of latent reasoning steps during inference
MAX_LATENT_STEPS=32

# Output directory for evaluation results
OUTPUT_PATH="./eval_results/HyLar_latent_step_${MAX_LATENT_STEPS}/${DATASET_NAME}"

# Number of GPUs for parallel inference
NUM_GPUS=8

# Run evaluation
python Evaluate/eval_HyLar.py \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_path "$OUTPUT_PATH" \
    --max_new_tokens 1024 \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --num_gpus $NUM_GPUS \
    --max_latent_steps $MAX_LATENT_STEPS \
    --num_samples -1
