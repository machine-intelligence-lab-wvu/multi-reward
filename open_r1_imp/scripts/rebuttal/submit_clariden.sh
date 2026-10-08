#!/bin/bash
# ============================================================
# SLURM Submission Script — Rebuttal recipes on CSCS Clariden (GH200 120GB)
#
# Generic version of ../submit.sh / ../submit_grpo_gt.sh / ../submit_intuitor.sh
# that also parameterizes the training entry script, since the rebuttal
# recipes route to three different ones (multi_reward_train.py /
# intuitor_train.py / grpo_gt_train.py) depending on method. Not meant to be
# called directly — use launch_clariden.sh <config_id>, which fills in
# CONFIG_OVERRIDE / TRAIN_SCRIPT / ACCEL_CONFIG_OVERRIDE for you.
#
# Architecture (1 node, 4 GPUs — the full node, unlike Bridges-2's
# GPU-shared queue that launch.sh/submit_bridges2.sh had to ration):
#   GPU 0: vLLM inference server (trl vllm-serve)
#   GPUs 1-3: accelerate + DeepSpeed training (NUM_TRAIN_GPUS=3)
# ============================================================

#SBATCH --job-name=rebuttal
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/rebuttal-%j.out
#SBATCH --error=logs/rebuttal-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

# ─── Required inputs (set via launch_clariden.sh's --export) ────
CONFIG="${CONFIG_OVERRIDE:?CONFIG_OVERRIDE not set — use launch_clariden.sh <config_id>}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:?TRAIN_SCRIPT not set — use launch_clariden.sh <config_id>}"
ACCEL_CONFIG="${ACCEL_CONFIG_OVERRIDE:?ACCEL_CONFIG_OVERRIDE not set — use launch_clariden.sh <config_id>}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ─── Paths ───────────────────────────────────────────────────
PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

mkdir -p "${PROJECT_DIR}/logs"
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
export WANDB_PROJECT="multi-reward-rebuttal"
export WANDB_DISABLE_WEAVE=true

# ─── vLLM ────────────────────────────────────────────────────
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # avoid "Cannot re-initialize CUDA in forked subprocess" if V0 falls back to V1's mp executor

# ─── Build GPU device lists ─────────────────────────────────
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 "$NUM_TRAIN_GPUS"); do
    TRAIN_GPUS="${TRAIN_GPUS:+${TRAIN_GPUS},}$i"
done

VLLM_MODEL=$(grep "^model_name_or_path:" "$CONFIG" | head -1 | awk '{print $2}')

echo "============================================================"
echo "  Rebuttal run"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  Config:        ${CONFIG}"
echo "  Train script:  ${TRAIN_SCRIPT}"
echo "  Accelerate:    ${ACCEL_CONFIG}"
echo "  vLLM GPU:      ${VLLM_GPU} (${VLLM_MODEL}) | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: Start vLLM server on GPU 0 (background)
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
    --model "${VLLM_MODEL}" \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.8 \
    --enforce_eager true &
VLLM_PID=$!
echo "[$(date)] vLLM server PID: ${VLLM_PID}"

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
# PHASE 2: Launch training via accelerate on GPUs 1-N
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching ${TRAIN_SCRIPT} on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

if [ -n "${VLLM_PID:-}" ]; then
    echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
fi

echo "[$(date)] Rebuttal job complete."
exit $TRAIN_EXIT
