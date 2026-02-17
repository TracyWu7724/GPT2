#!/bin/bash

# EECS 595 HW3: Tiny GPT Training Script
# This script trains a very small GPT model for local testing
# Designed to run quickly on CPU for student verification

echo "Starting tiny GPT training for local testing..."
echo "=============================================="

# cd to src/ so that `models` package is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Training hyperparameters for tiny model (CPU-friendly)
python scripts/pretrain_gpt.py \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --max_epochs 1 \
    --emb_dim 32 \
    --n_layers 2 \
    --n_heads 4 \
    --context_length 64 \
    --drop_rate 0.0 \
    --weight_decay 0.01 \
    --warmup_steps 10 \
    --max_docs 100 \
    --stride 64 \
    --save_every 50 \
    --eval_every 25 \
    --device cpu \
    --no_compile \
    --data_path ./data/fineweb_tiny_train.jsonl \
    --eval_data_path ./data/fineweb_tiny_eval.jsonl \
    --output_dir ./models/tiny/ \
    --wandb_run_name "gpt-tiny-test-$(date +%Y%m%d-%H%M%S)"

echo "Tiny training completed!"
echo "This should run in a few minutes on CPU."
echo "Check ./models/tiny/ for the saved model."
