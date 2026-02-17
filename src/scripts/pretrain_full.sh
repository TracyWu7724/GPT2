#!/bin/bash

# GPT Full Pretraining Script
# This script trains the full GPT model on GPU
# Designed for multi-epoch pretraining on the FineWeb-Edu dataset

echo "Starting full GPT pretraining..."
echo "=============================================="

# cd to src/ so that `models` package is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Full training hyperparameters
python scripts/pretrain_gpt.py \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --max_epochs 2 \
    --target_tokens 1200000000 \
    --emb_dim 512 \
    --n_layers 12 \
    --n_heads 8 \
    --context_length 1024 \
    --drop_rate 0.1 \
    --weight_decay 0.1 \
    --stride 512 \
    --save_every 1000 \
    --eval_every 500 \
    --num_workers 8 \
    --device auto \
    --data_format arrow \
    --data_path ./data/fineweb_train \
    --eval_data_format arrow \
    --eval_data_path ./data/fineweb_eval \
    --output_dir ./models/pretrained-models/ \
    --wandb_project gpt-pretraining \
    --wandb_run_name "gpt-full-$(date +%Y%m%d-%H%M%S)"

echo "Full pretraining completed!"
echo "Check ./models/pretrained-models/ for the saved model."
