#!/bin/bash
# ============================================================
# Traditional GRPO Baseline (for comparison)
# Single GPU (RTX 4090, 24GB) — Qwen3-1.7B + LoRA
#
# Uses TRL's GRPOTrainer directly — no multi-reward, no KL-Cov.
# Same model, dataset, LoRA config, and hyperparameters as
# train4090.sh so the comparison is fair.
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_grpo_baseline.py \
    --model_name Qwen/Qwen3-1.7B \
    --lora_rank 64 \
    --lora_alpha 32 \
    --dataset gsm8k \
    --num_generations 8 \
    --max_steps 500 \
    --max_completion_length 1024 \
    --temperature 1.5 \
    --w_correctness 1.0 \
    --w_format 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --beta_kl 0.04 \
    --output_dir ./outputs/grpo_baseline_4090 \
    --logging_steps 1 \
    --save_steps 100 \
    --run_name grpo_baseline_4090 \
    "$@"
