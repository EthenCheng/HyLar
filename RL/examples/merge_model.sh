#!/bin/bash
# Merge FSDP sharded checkpoints into a single HuggingFace model.
# Set CKPT_PATH to the actor checkpoint directory from your training run.

CKPT_PATH=path/to/your/checkpoint/actor
python -m scripts.model_merger --local_dir=${CKPT_PATH}
