#!/bin/bash
# ============================================================
# SLURM Submission Script — Standard GRPO with Ground-Truth Reward
# Cluster: CSCS Clariden (GH200 120GB GPUs)
#
# Architecture (1 node, 4 GPUs):
#   GPU 0: vLLM inference server (trl vllm-serve)
#   GPUs 1-3: accelerate + DeepSpeed training
#
# Usage (from project root):
#   sbatch open_r1_imp/scripts/submit_grpo_gt.sh
#   sbatch open_r1_imp/scripts/submit_grpo_gt.sh --export=ALL,MODEL=Qwen/Qwen2.5-1.5B
# ============================================================

#SBATCH --job-name=grpo-gt
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/grpo-gt-%j.out
#SBATCH --error=logs/grpo-gt-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────
PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# ─── Configurable defaults (override via --export) ───────────
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-}"
ACCEL_CONFIG_OVERRIDE="${ACCEL_CONFIG_OVERRIDE:-}"
MODEL="${MODEL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ─── Default config ─────────────────────────────────────────
CONFIG="${CONFIG_OVERRIDE:-open_r1_imp/recipes/config_grpo_gt_clariden.yaml}"

# Auto-detect LoRA from config to pick accelerate strategy (unless overridden)
if [ -n "$ACCEL_CONFIG_OVERRIDE" ]; then
    ACCEL_CONFIG="$ACCEL_CONFIG_OVERRIDE"
    if ! grep -q "use_peft: true" "$CONFIG" 2>/dev/null; then
        EXTRA_ARGS="${EXTRA_ARGS} --use_peft false"
    else
        EXTRA_ARGS="${EXTRA_ARGS} --use_peft true"
    fi
elif grep -q "use_peft: true" "$CONFIG" 2>/dev/null; then
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero2.yaml"
    EXTRA_ARGS="${EXTRA_ARGS} --use_peft true"
else
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3.yaml"
    EXTRA_ARGS="${EXTRA_ARGS} --use_peft false"
fi

# ─── Create directories ─────────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"

# ─── Activate virtual environment ────────────────────────────
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── NCCL & Networking (GH200 / Clariden optimised) ──────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_NVLS_ENABLE=0

# ─── PyTorch / CUDA ─────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=18
export TOKENIZERS_PARALLELISM=false

# ─── HF ───────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your shell before submitting, e.g. export HF_TOKEN=hf_xxx}"

# ─── WandB ───────────────────────────────────────────────────
export WANDB_PROJECT="multi-reward-multigpu"
export WANDB_DISABLE_WEAVE=true

# ─── vLLM ────────────────────────────────────────────────────
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoid "Cannot re-initialize CUDA in forked subprocess" if V0 falls back to V1's mp executor

# ─── Build GPU device lists ─────────────────────────────────
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 $NUM_TRAIN_GPUS); do
    if [ -n "$TRAIN_GPUS" ]; then
        TRAIN_GPUS="${TRAIN_GPUS},$i"
    else
        TRAIN_GPUS="$i"
    fi
done

# Model override
if [ -n "$MODEL" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_name_or_path ${MODEL}"
fi

# ─── Print job info ──────────────────────────────────────────
echo "============================================================"
echo "  Standard GRPO — Ground-Truth Reward"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPUs:          ${SLURM_GPUS_ON_NODE:-4}"
echo "  Config:        ${CONFIG}"
echo "  Accelerate:    ${ACCEL_CONFIG}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: Start vLLM server on GPU 0 (background)
# ════════════════════════════════════════════════════════════════
VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')
if [ -n "$MODEL" ]; then
    VLLM_MODEL="$MODEL"
fi

echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
    --model "${VLLM_MODEL}" \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.8 \
    --enforce_eager true &
VLLM_PID=$!
echo "[$(date)] vLLM server PID: ${VLLM_PID}"

# Wait for vLLM to be ready
echo "[$(date)] Waiting for vLLM server to start..."
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
# PHASE 2: Launch GRPO training via accelerate on GPUs 1-N
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching standard GRPO (ground-truth) training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/grpo_gt_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

# ════════════════════════════════════════════════════════════════
# Cleanup: kill vLLM server
# ════════════════════════════════════════════════════════════════
if [ -n "${VLLM_PID:-}" ]; then
    echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

echo "[$(date)] Standard GRPO (ground-truth) job complete."
exit $TRAIN_EXIT
