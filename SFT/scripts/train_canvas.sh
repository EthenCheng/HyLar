#!/bin/bash

export WANDB_PROJECT="Canvas-7B-SFT"
export WANDB_DISABLED=false
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# Model configs
MODEL_SIZE='7B'
MODEL_NAME="path/to/Qwen2.5-VL-7B-Instruct"

CANVAS_ENCODER="path/to/siglip2-so400m-patch14-384"

PATTERN="vl"
GRAD_CHECK=True

RANDOM_SEED=42
DATA_PATH="path/to/your/sft/data"

GLOBAL_BATCH_SIZE=128      
BATCH_PER_DEVICE=1           
NUM_DEVICES=8
# GRAD_ACCUM_STEPS=16
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

LR=1e-5
CANVAS_LR=1e-4

CANVAS_LOSS=mse
CANVAS_LAMBDA=0.5
CANVAS_TOKEN_NUM=16

USE_CANVAS_COMPRESSOR=True
COMPRESSOR_N_HEADS=8
COMPRESSOR_N_LAYERS=2

MAX_TOKEN=8192
MIN_TOKEN=128

RUN_NAME="canvas/siglipvit_${PATTERN}_canvas${CANVAS_TOKEN_NUM}_${CANVAS_LAMBDA}${CANVAS_LOSS}_${MODEL_SIZE}_lr${LR}_bsz${GLOBAL_BATCH_SIZE}_maxImgToken${MAX_TOKEN}"
OUTPUT_DIR="canvas/siglipvit_${PATTERN}_canvas${CANVAS_TOKEN_NUM}_${CANVAS_LAMBDA}${CANVAS_LOSS}_${MODEL_SIZE}_lr${LR}_bsz${GLOBAL_BATCH_SIZE}_maxImgToken${MAX_TOKEN}"


export PYTHONPATH=$(pwd)
deepspeed src/train/train_canvas.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --pattern $PATTERN \
    --model_id $MODEL_NAME \
    --canvas_encoder $CANVAS_ENCODER \
    --data_path "$DATA_PATH" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --max_grad_norm 1.0 \
    --learning_rate $LR \
    --canvas_lr $CANVAS_LR \
    --canvas_loss $CANVAS_LOSS\
    --canvas_lambda $CANVAS_LAMBDA \
    --use_canvas_compressor $USE_CANVAS_COMPRESSOR \
    --compressor_n_heads $COMPRESSOR_N_HEADS \
    --compressor_n_layers $COMPRESSOR_N_LAYERS \
    --canvas_token_num $CANVAS_TOKEN_NUM \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 28 * 28)) \
    --image_max_pixels $((MAX_TOKEN * 28 * 28)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --tf32 False \
    --gradient_checkpointing $GRAD_CHECK \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 32 \
    --random_seed $RANDOM_SEED \
    --report_to wandb
