#!/bin/bash
# ============================================================
# INTUITOR — Self-Certainty GRPO (Paper Reproduction)
# Single GPU (RTX 4090, 24GB) — Qwen2.5-1.5B + LoRA
# ============================================================
# Usage:
#   bash scripts/train_intuitor.sh
#   bash scripts/train_intuitor.sh --max_steps 1000
#   bash scripts/train_intuitor.sh --resume_from ./outputs/intuitor/Qwen2.5-1.5B_run1_math/run_0/checkpoint-100
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="intuitor"
export WANDB_DISABLE_WEAVE=true
export VLLM_USE_V1=0

python train_intuitor.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --lora_rank 64 \
    --lora_alpha 128 \
    --dataset math \
    --num_generations 7 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --beta_kl 0.005 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 500 \
    --output_dir ./outputs/intuitor/Qwen2.5-1.5B_run1_math \
    --logging_steps 1 \
    --save_steps 10 \
    --print_rollouts_every 10 \
    --run_name intuitor_Qwen2.5-1.5B_MATH \
    --gpu_memory_utilization .5 \
    --use_wandb \
    --seed 7 \
    "$@"
