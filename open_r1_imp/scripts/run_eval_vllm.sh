#!/bin/bash
# ============================================================
# Fast vLLM Evaluation for LoRA Checkpoints
#
# Uses vLLM with native LoRA support for ~100-200x faster evaluation
# compared to the old HuggingFace sequential eval (eval_checkpoint.py).
#
# Defaults:
#   - LoRA mode:     native (no merge, exact weights)
#   - System prompt:  training (the prompt used during GRPO training)
#   - Dataset:       GSM8K (full 1319 samples)
#   - Decoding:      greedy (temperature=0)
#   - GPU:           GPU 2 (A100 80GB)
#
# Environment variables required for native LoRA mode:
#   VLLM_WORKER_MULTIPROC_METHOD=spawn
#   VLLM_ENABLE_V1_MULTIPROCESSING=0
#
# Usage:
#   # Run with defaults (training prompt, GSM8K, native LoRA)
#   bash open_r1_imp/scripts/run_eval_vllm.sh
#
#   # Override checkpoint
#   CHECKPOINT=outputs/intuitor_klcov_lora_local/checkpoint-700 \
#       bash open_r1_imp/scripts/run_eval_vllm.sh
#
#   # Override system prompt
#   SYSTEM_PROMPT=eval_v2 bash open_r1_imp/scripts/run_eval_vllm.sh
#
#   # Override dataset
#   DATASET=math500 bash open_r1_imp/scripts/run_eval_vllm.sh
#
#   # Override GPU
#   GPU=0 bash open_r1_imp/scripts/run_eval_vllm.sh
#
#   # Quick test with 100 samples
#   MAX_SAMPLES="--max_samples 100" bash open_r1_imp/scripts/run_eval_vllm.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ── vLLM environment (required for native LoRA) ──
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false

# ── Configurable defaults (override via environment) ──
CHECKPOINT="${CHECKPOINT:-outputs/multi_reward_lora_local_klcov005/checkpoint-500}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-training}"
DATASET="${DATASET:-gsm8k}"
GPU="${GPU:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "============================================================"
echo "  vLLM Evaluation (native LoRA)"
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  System prompt: ${SYSTEM_PROMPT}"
echo "  Dataset:       ${DATASET}"
echo "  GPU:           ${GPU}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python evaluation/eval_vllm.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset ${DATASET} \
    --system_prompt ${SYSTEM_PROMPT} \
    --lora_mode native \
    --max_lora_rank 64 \
    --max_new_tokens 1024 \
    --min_new_tokens 64 \
    --max_model_len 4096 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.7 \
    --results_dir eval_results \
    ${MAX_SAMPLES}

echo "============================================================"
echo "  Done! Results saved in eval_results/"
echo "============================================================"
