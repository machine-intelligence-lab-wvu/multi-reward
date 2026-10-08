#!/bin/bash
# ============================================================
# Multi-Reward Unsupervised GRPO + KL-Cov
# Single GPU (RTX 4090, 24GB) — Qwen3-1.7B + LoRA
# ============================================================
# Usage:
#   bash scripts/train_4090.sh
#   bash scripts/train_4090.sh --dataset math --max_steps 500
#   bash scripts/train_4090.sh --enable_thinking --max_completion_length 2048 --num_generations 4
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="multi-reward-trl-testrun"
export WANDB_DISABLE_WEAVE=true

python train_trl.py \
    --model_name Qwen/Qwen3-1.7B \
    --lora_rank 64 \
    --lora_alpha 32 \
    --dataset gsm8k \
    --num_generations 8 \
    --num_iterations 1 \
    --max_completion_length 1024 \
    --temperature 1.5 \
    --w_cluster 0.5 \
    --w_format 0.1 \
    --w_cluster_format 0.5 \
    --w_selfcert 0.5 \
    --num_train_epochs 1 \
    --max_steps 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --beta_cov 0.1 \
    --top_k_fraction 0.02 \
    --beta_kl 0.005 \
    --output_dir ./outputs/multi_reward_grpo_trl \
    --logging_steps 1 \
    --save_steps 100 \
    --run_name wanddb_run3_gsm8k \
    --print_rollouts_every 0 \
    --use_wandb \
    "$@"
