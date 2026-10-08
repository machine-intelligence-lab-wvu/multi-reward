#!/bin/bash
# ============================================================
# Multi-Reward Unsupervised GRPO — From Scratch (vLLM + PyTorch)
# Single GPU (RTX 4090, 24GB) — Qwen2.5-1.5B + LoRA
# ============================================================
# Usage:
#   bash scripts/train_multi-reward_scratch.sh
#   bash scripts/train_multi-reward_scratch.sh --dataset gsm8k
#   bash scripts/train_multi-reward_scratch.sh --resume_from ./outputs/.../checkpoint-200
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="multi-reward-trl-testrun"
export WANDB_DISABLE_WEAVE=true
# vLLM V1 has a memory profiling race condition when sharing GPU with PyTorch training.
# V0 handles dynamic memory gracefully.
export VLLM_USE_V1=0


##use beta_kl = 0.005 for the next run
python train_from_scratch.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --lora_rank 64 \
    --lora_alpha 128 \
    --dataset math-lighteval \
    --num_generations 7 \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --w_cluster 0.5 \
    --w_format 0.1 \
    --w_cluster_format 0.5 \
    --w_selfcert 0.5 \
    --beta_cov 0.0 \
    --top_k_fraction 0.02 \
    --beta_kl 0.005 \
    --entropy_bonus 0.0 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 500 \
    --output_dir outputs/multi_reward_scratch_4090/Qwen2.5-1.5B_run_4_math-lighteval_reasoninglength \
    --logging_steps 1 \
    --save_steps 5 \
    --print_rollouts_every 10 \
    --run_name scratch_run_math-lighteval__reasoninglength_Qwen2.5-1.5B \
    --gpu_memory_utilization .5 \
    --use_wandb \
    "$@"
