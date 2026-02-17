"""
The script will:
1. Load data from the specified dataset
2. Create train/validation splits
3. Initialize the GPT model
4. Train the model with mixed precision
5. Save checkpoints and log to wandb

Usage: python -m scripts.pretrain_gpt --data_path <path> [options]
"""

import os
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, DataLoader
import json
import glob
import gzip
import bz2
import datetime

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Progress and timing
from tqdm.auto import tqdm, trange
import time
import wandb

# Import our GPT implementation
import models.gpt2 as gpt
import data.pretrain_dataset as gptData
from sklearn.model_selection import train_test_split

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                       default='./Data/fineweb-edu-sample-1M.jsonl.gz',
                       help='Path to the training data (JSONL.gz file or Arrow dataset directory)')
    parser.add_argument('--data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing, only applies to raw text)')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected if not specified)')
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=2,
                       help='Maximum number of epochs')
    parser.add_argument('--target_tokens', type=int, default=1_200_000_000,
                       help='Target number of tokens to train on')
    parser.add_argument('--warmup_steps', type=int, default=None,
                       help='Number of warmup steps (default: min(400, 2%% of total steps))')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window tokenization (default: context_length // 2)')

    # Validation arguments
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to validation data (if not provided, splits training data 95/5)')
    parser.add_argument('--eval_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='./models/pretrained-models/',
                       help='Output directory for saving models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                       help='Evaluate model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-pretraining',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
                       default=f"gpt-pretraining-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                       help='Wandb run name')
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_arg

def get_amp_dtype(device):
    '''Get the appropriate AMP dtype for mixed precision training on the device.'''

    if device.startswith('cuda'):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == 'mps':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  # or disable autocast on CPU
    return amp_dtype

def load_data(data_path, max_docs=None, data_format='jsonl'):
    """
    Load data from JSONL file or Arrow dataset.

    Args:
        data_path: Path to the data file or Arrow dataset directory
        max_docs: Maximum number of documents to load (only for raw text)
        data_format: Format of the data ('jsonl' or 'arrow')
    Returns:
        List of text documents (for raw text) or None (for Arrow datasets)
    """
    if data_format == 'arrow':
        print(f"Using Arrow dataset from {data_path}")
        # For Arrow datasets, we don't need to load the data here
        # The GPTArrowDataset in gpt.py will handle loading
        return None
    else:
        print(f"Loading data from {data_path}")

        ofunc = gzip.open if data_path.endswith('gz') else open
        docs = []

        with ofunc(data_path, 'rt') as f:
            for i, line in enumerate(tqdm(f, desc="Reading data from file")):
                if max_docs and i >= max_docs:
                    break
                docs.append(json.loads(line)['text'])

        print(f"Loaded {len(docs)} documents")
        return docs

def create_dataloaders(docs, tokenizer, config, args):
    """Create train and validation dataloaders."""
    print("Creating dataloaders...")

    if args.data_format == "arrow":
        print(f"Using Arrow training dataset from {args.data_path}")

        train_loader = gptData.create_dataloader(
            arrow_dataset_path=args.data_path,
            batch_size=args.batch_size,
            max_length=config["context_length"],
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers
        )
        train_docs = None

    else:
        print(f"Using JSONL training data from {args.data_path}")
        train_docs = docs #TODO: no need for loader??
    
    val_loader = None

    if args.eval_data_path:
        print(f"Validation data path detected: {args.eval_data_path}")

        if args.eval_data_format == "arrow":
            val_loader = gptData.create_dataloader(
                arrow_dataset_path=args.eval_data_path, 
                batch_size=args.eval_batch_size or args.batch_size,
                max_length=config['context_length'],
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )
        else:
            print(f"Using JSONL validation dataset from {args.eval_data_path}")
            val_docs = load_data(args.eval_data_path, args.eval_max_docs, args.eval_data_format)

            val_dataset = gptData.GPTDataset(
                docs=val_docs,
                tokenizer=tokenizer,
                max_length=config['context_length'],
                stride=config.get("stride", config["context_length"] // 2)
            )

            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.eval_batch_size or args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers
            )
        print(f"Training batches:   {len(train_loader)}")
        print(f"Validation batches: {len(val_loader) if val_loader else 0}")
        print("✅ Dataloaders created successfully.\n")

        return train_loader, val_loader
            
    elif train_docs is not None:
        print("No separate validation path provided — splitting training data 95/5.")

        # TODO: not understand this, critical safeguard: ensure test size ios at least one 
        total_train_docs = len(train_docs)
        min_test_size = 1
        test_size = max(min_test_size, int(total_train_docs * 0.05))

        train_docs_split, val_docs_split = train_test_split(train_docs, test_size=test_size, random_state=args.seed)

        train_dataset = gptData.GPTDataset(
            docs=train_docs_split,
            tokenizer=tokenizer,
            max_length=config['context_length'],
            stride=config.get("stride", config["context_length"] // 2)
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True, drop_last=True, num_workers=args.num_workers
        )

        val_dataset = gptData.GPTDataset(
            docs=val_docs_split,
            tokenizer=tokenizer,
            max_length=config['context_length'],
            stride=config.get("stride", config["context_length"] // 2)
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )
        print("✅ Dataloaders created")
        print(f"📊 Training documents: {len(train_loader)}")
        print(f"📊 Validation documents: {len(val_loader)}")
        return train_loader, val_loader

def save_model(model, optimizer, args, global_step, config, is_epoch_end=False):

    if is_epoch_end:
        filename = f"model_epoch_final.pth"
    else:
        filename = f"model_step_{global_step}.pth"
    
    save_path = os.path.join(args.output_dir, args.wandb_run_name, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'global_step': global_step,
    }, save_path)
    print(f"✅ Model saved to {save_path}")


def evaluate_validation_loss(model, val_loader, loss_fn, device, amp_dtype, max_batches=None):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device, dtype=amp_dtype):
                logits, _ = model(input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss_val = loss_fn(logits_flat, labels_flat)

            total_loss += loss_val.item()
            num_batches += 1

            if max_batches is not None and num_batches >= max_batches:
                break

    loss_avg = total_loss / max(num_batches, 1)
    model.train()

    return loss_avg

def train_model(model, train_loader, val_loader, config, args):
    device = get_device(args.device)
    amp_dtype = get_amp_dtype(device)
    print(f"Using device: {device}")

    # Move model to device
    model.to(device)

    # Set up training components
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    tokens_per_step = config['context_length'] * args.batch_size
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(400, int(0.02 * total_steps))

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5 # half-cosine
    )

    # Initialize wandb
    wandb_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "position_embedding": "rope",
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "context_length": config["context_length"],
        "drop_rate": config["drop_rate"],
    }
    wandb.init(project=args.wandb_project,
               config=wandb_config,
               name=args.wandb_run_name,
               )
    
    # Training loop
    model.train()
    opt_step = 0 # TODO: confused with opt_step and global_step
    global_step = 0
    losses = []

    last_eval_step = -1
    last_save_step = -1

    # To avoid running out of memory, we use gradient accumulation
    target_global_batch = 256
    micro_batch = args.batch_size
    accum = max(1, target_global_batch // micro_batch)

    print(f"Starting training...")
    print(f"Gradient accumulation steps: {accum}")

    # GradScaler is only needed for float16; bfloat16 doesn't need loss scaling
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    current_step_loss = 0.0
    tokens_trained = 0
    training_done = False

    for epoch in trange(args.max_epochs, desc="Epoch"):
        if training_done:
            break

        for step, (input_ids, labels) in enumerate(tqdm(train_loader, position=1, leave=True, desc="Step")):
            # Move input_ids and labels to the correct device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            with autocast(device_type=device, dtype=amp_dtype):
                logits, _ = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / accum  # scale for gradient accumulation

            # Backward pass
            scaler.scale(loss).backward()
            current_step_loss += loss.item()
            global_step += 1

            # Optimizer step after accumulation
            if global_step % accum == 0:
                # Unscale, Clip and Step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Clear gradients and update LR
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1

                # Reconstruct the un-scaled loss for this optimizer step
                step_loss = current_step_loss  # already accumulated (loss/accum) * accum = true loss
                current_step_loss = 0.0
                tokens_trained = opt_step * tokens_per_step * accum

                # Log to wandb
                wandb.log({
                    "train/loss_step": step_loss,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/tokens": tokens_trained,
                    "train/opt_step": opt_step,
                    "train/epoch": epoch,
                }, step=opt_step)

                # Step evaluation
                if val_loader is not None and opt_step % args.eval_every == 0 and opt_step != last_eval_step:
                    last_eval_step = opt_step
                    val_loss = evaluate_validation_loss(
                        model, val_loader, loss_fn, device, amp_dtype,
                        max_batches=args.eval_max_docs_step,
                    )
                    print(f"\n[Step {opt_step}] Val loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.2f}")
                    wandb.log({
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(min(val_loss, 20)),
                    }, step=opt_step)

                # Step checkpoint
                if opt_step % args.save_every == 0 and opt_step != last_save_step:
                    last_save_step = opt_step
                    save_model(model, optimizer, args, opt_step, config)

                # Check if we've hit the target token count
                if tokens_trained >= args.target_tokens:
                    print(f"\nReached target tokens ({tokens_trained:,} >= {args.target_tokens:,}). Stopping.")
                    training_done = True
                    break

        # End-of-epoch evaluation and checkpoint
        if val_loader is not None:
            val_loss = evaluate_validation_loss(model, val_loader, loss_fn, device, amp_dtype)
            print(f"\n[Epoch {epoch}] Val loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.2f}")
            wandb.log({"val/loss": val_loss, "val/perplexity": math.exp(min(val_loss, 20))}, step=opt_step)

        save_model(model, optimizer, args, opt_step, config, is_epoch_end=True)

    print(f"Training completed! Total optimizer steps: {opt_step}, tokens: {tokens_trained:,}")
    wandb.finish()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up tokenizer
    print("Setting up tokenizer....")
    tokenizer = gptData.setup_tokenizer()

    # Determine vocabulary size
    if args.vocab_size is None:
        special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
        max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
        vocab_size = max_token_id + 1
    
    else:
        vocab_size = args.vocab_size
    
    print(f"Using vocabulary size: {vocab_size}")

    # Create model config 
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False,
        "stride": args.stride if args.stride is not None else args.context_length // 2,
    }

    docs = load_data(args.data_path, args.max_docs, args.data_format)

    train_loader, val_loader = create_dataloaders(docs, tokenizer, config, args)
    print("\n" + "="*80)
    print("📘 GPT Pretraining Configuration Summary")
    print("="*80)
    print(f"🗂 Data path:        {args.data_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print("-"*80)
    print(f"🧠 Model structure:")
    print(f"   vocab_size={config['vocab_size']}, context_length={config['context_length']}")
    print(f"   emb_dim={config['emb_dim']}, n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    print(f"   dropout={config['drop_rate']}")
    print("-"*80)
    print(f"⚙️ Training settings:")
    print(f"   batch_size={args.batch_size}, learning_rate={args.learning_rate}")
    print(f"   weight_decay={args.weight_decay}, max_epochs={args.max_epochs}")
    print(f"   save_every={args.save_every}, eval_every={args.eval_every}")
    print("-"*80)
    print(f"💻 Device:           {get_device(args.device)}")
    print(f"🌱 Random seed:      {args.seed}")
    print(f"🧵 Num workers:      {args.num_workers}")
    print("-"*80)
    print(f"📊 WandB Project:    {args.wandb_project}")
    print(f"📈 Run Name:         {args.wandb_run_name}")
    print("="*80 + "\n")

    device = get_device(args.device)
    print(device)

    model = gpt.GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    if not args.no_compile:
        model = torch.compile(model, mode="default")

    train_model(model, train_loader=train_loader, val_loader=val_loader, config=config, args=args)



if __name__ == "__main__":
    main()