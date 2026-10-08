#!/bin/bash
# ============================================================
# Multi-Reward GRPO — Thinking Mode (Qwen3)
# Self-certainty on REASONING tokens only (<think>...</think>)
# Persistent vLLM — requires >= 40GB VRAM (RTX PRO 6000, A100)
# ============================================================
# Usage:
#   bash scripts/train_thinking.sh
#   bash scripts/train_thinking.sh --model_name Qwen/Qwen3-4B
#   bash scripts/train_thinking.sh --resume_from ./outputs/thinking/run_0/checkpoint-100
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="multi-reward-thinking"
export WANDB_DISABLE_WEAVE=true
export VLLM_USE_V1=1

python train_thinking.py \
    --model_name Qwen/Qwen3-1.7B \
    --lora_rank 64 \
    --lora_alpha 128 \
    --dataset math-lighteval \
    --num_generations 7 \
    --max_prompt_length 512 \
    --max_completion_length 4096 \
    --temperature 0.6 \
    --w_cluster 0.5 \
    --w_format 0.1 \
    --w_cluster_format 0.5 \
    --w_selfcert 0.5 \
    --beta_cov 0.0 \
    --top_k_fraction 0.02 \
    --beta_kl 0.005 \
    --entropy_bonus 0.0 \
    --min_completion_tokens 50 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-6 \
    --lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 500 \
    --output_dir ./outputs/thinking \
    --logging_steps 1 \
    --save_steps 10 \
    --print_rollouts_every 20 \
    --run_name thinking_Qwen3-1.7B_MATH \
    --gpu_memory_utilization 0.25 \
    --use_wandb \
    "$@"
