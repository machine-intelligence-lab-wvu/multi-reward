#!/bin/bash
# ============================================================
# Local Launch Script — INTUITOR (Self-Certainty GRPO, LoRA)
# Hardware: GPU 0 (RTX 4500) = vLLM, GPUs 1-2 (RTX 4500 + A100) = training
#
# Usage:
#   bash open_r1_imp/scripts/run_intuitor_local.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

VLLM_GPU=0
TRAIN_GPUS="2"
NUM_TRAIN_GPUS=1

CONFIG="open_r1_imp/recipes/config_intuitor_lora_local.yaml"
ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero2_local.yaml"

# ─── Activate environment ────────────────────────────────────
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── PyTorch / CUDA ─────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# ─── WandB ───────────────────────────────────────────────────
export WANDB_PROJECT="multi-reward-multigpu"
export WANDB_DISABLE_WEAVE=true

# ─── vLLM ────────────────────────────────────────────────────
export VLLM_USE_V1=0

# ─── Print info ─────────────────────────────────────────────
echo "============================================================"
echo "  INTUITOR — LoRA (Local)"
echo "  Config:        ${CONFIG}"
echo "  Accelerate:    ${ACCEL_CONFIG}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: Start vLLM server on GPU 0
# ════════════════════════════════════════════════════════════════
VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')

echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
    --model "${VLLM_MODEL}" \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.8 &
VLLM_PID=$!
echo "[$(date)] vLLM server PID: ${VLLM_PID}"

# Wait for vLLM to be ready
echo "[$(date)] Waiting for vLLM server..."
VLLM_READY=false
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM server ready!"
        VLLM_READY=true
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died!"
        exit 1
    fi
    sleep 5
done
if [ "$VLLM_READY" = false ]; then
    echo "ERROR: vLLM server did not become healthy within 300s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ════════════════════════════════════════════════════════════════
# PHASE 2: Launch INTUITOR training on GPUs 1-2
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching INTUITOR training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/intuitor_train.py \
    --config "${CONFIG}"

TRAIN_EXIT=$?

# ════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "[$(date)] INTUITOR done."
exit $TRAIN_EXIT
