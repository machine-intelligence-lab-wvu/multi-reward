#!/bin/bash
# ============================================================
# Multi-Reward GRPO — Dual GPU (No Memory Leak)
# Uses 2 GPUs: one for vLLM server, one for training.
# Works on any multi-GPU setup. On 4+ GPUs, run parallel experiments.
# ============================================================
# Usage:
#   1. Start vLLM server (in a separate terminal/tmux):
#      bash scripts/start_vllm_server.sh
#
#   2. Run training:
#      bash scripts/train_dual_gpu.sh
#
#   3. Resume from checkpoint:
#      bash scripts/train_dual_gpu.sh --resume_from ./outputs/dual_gpu/run_0/checkpoint-100
#
#   4. Run 2 experiments on a 4-GPU cluster:
#      # Experiment 1: GPU 0 (vLLM) + GPU 1 (train)
#      CUDA_VISIBLE_DEVICES=0 bash scripts/start_vllm_server.sh Qwen/Qwen2.5-1.5B 8000
#      CUDA_VISIBLE_DEVICES=1 bash scripts/train_dual_gpu.sh --vllm_base_url http://localhost:8000
#
#      # Experiment 2: GPU 2 (vLLM) + GPU 3 (train)
#      CUDA_VISIBLE_DEVICES=2 bash scripts/start_vllm_server.sh Qwen/Qwen2.5-3B 8001
#      CUDA_VISIBLE_DEVICES=3 bash scripts/train_dual_gpu.sh --vllm_base_url http://localhost:8001
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="multi-reward-dual-gpu"
export WANDB_DISABLE_WEAVE=true

# Training runs on GPU 1 only
export CUDA_VISIBLE_DEVICES=1

python train_dual_gpu.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --lora_rank 64 \
    --lora_alpha 128 \
    --dataset math-lighteval \
    --num_generations 7 \
    --max_prompt_length 512 \
    --max_completion_length 3072 \
    --temperature 1.0 \
    --vllm_base_url http://localhost:8000 \
    --w_cluster 0.5 \
    --w_format 0.1 \
    --w_cluster_format 0.5 \
    --w_selfcert 0.5 \
    --beta_cov 0.0 \
    --beta_kl 0.005 \
    --entropy_bonus 0.0 \
    --min_completion_tokens 50 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-6 \
    --lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 500 \
    --train_device cuda:0 \
    --output_dir ./outputs/dual_gpu \
    --logging_steps 1 \
    --save_steps 10 \
    --print_rollouts_every 10 \
    --run_name dual_gpu_Qwen2.5-1.5B_MATH \
    --use_wandb \
    "$@"
