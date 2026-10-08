#!/bin/bash -lt
# ============================================================
# SLURM Submission Script - Multi-Reward GRPO 7B on A40 nodes (DollySods)
# Cluster: WVU DollySods, A40 46GB nodes (dscog031-034)
#
# Architecture (1 node, 4 GPUs):
#   GPU 0    : vLLM inference server (trl vllm-serve)
#   GPUs 1-3 : accelerate + DeepSpeed ZeRO-3 training (3 train GPUs)
#
# 7B + Zero3 AdamW state (~52 GB/GPU on-GPU) does NOT fit on A40 46GB.
# This variant CPU-offloads the optimizer state via zero3_grad42_cpu_offload.yaml,
# bringing per-GPU memory to ~24 GB. Trade-off: 3-5x slower per step.
#
# Estimated wall time for 500 steps: 30-60 h (uses gpu_7day partition; 7-day walltime).
# Effective batch = per_device(7) x grad_accum(42) x 3 = 882 (matches the A100 run).
#
# Use this in addition to the A100 sbatch: whichever job starts first wins,
# scancel the loser when one starts running.
# ============================================================

#SBATCH --job-name=mrew-7b-klcov0005-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --partition=gpu_7day
#SBATCH --nodelist=dscog031,dscog032,dscog033,dscog034   # all 4 A40 nodes
#SBATCH --time=3-00:00:00                                # 72h, well under gpu_7day's 7-day cap
#SBATCH --output=slurm_logs/mrew-7b-klcov0005-a40-%j.out
#SBATCH --error=slurm_logs/mrew-7b-klcov0005-a40-%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/unsupervised-rl"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-urlenv}"

NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-open_r1_imp/recipes/final/config_multi_reward_klcov_0.0005_qwen25_7b.yaml}"
ACCEL_CONFIG_OVERRIDE="${ACCEL_CONFIG_OVERRIDE:-open_r1_imp/accelerate_configs/zero3_grad42_cpu_offload.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NO_VLLM="${NO_VLLM:-false}"

module load cuda/12.3 || true
export CUDA_HOME="${CUDA_HOME:-/shared/software/nvidia/cuda/12.3}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${HF_HUB_CACHE}"

export WANDB_PROJECT="${WANDB_PROJECT:-multi-reward-7b}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
mkdir -p "${WANDB_DIR}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=0

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"

VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 ${NUM_TRAIN_GPUS}); do
    if [ -n "${TRAIN_GPUS}" ]; then
        TRAIN_GPUS="${TRAIN_GPUS},$i"
    else
        TRAIN_GPUS="$i"
    fi
done

echo "============================================================"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPUs:          ${SLURM_GPUS_ON_NODE:-4}  (A40 46GB)"
echo "  Conda env:     ${CONDA_ENV_NAME}"
echo "  Config:        ${CONFIG_OVERRIDE}"
echo "  Accelerate:    ${ACCEL_CONFIG_OVERRIDE}  (CPU optimizer offload)"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "  HF cache:      ${HF_HUB_CACHE} (offline=${HF_HUB_OFFLINE})"
echo "  Wandb:         ${WANDB_PROJECT} (mode=${WANDB_MODE})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "============================================================"

if [ "${NO_VLLM}" = false ]; then
    VLLM_MODEL=$(grep "model_name_or_path" "${CONFIG_OVERRIDE}" | head -1 | awk '{print $2}')

    echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 &
    VLLM_PID=$!
    echo "[$(date)] vLLM server PID: ${VLLM_PID}"

    echo "[$(date)] Waiting for vLLM server to start (up to 600s)..."
    VLLM_READY=false
    for i in $(seq 1 120); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "[$(date)] vLLM server ready!"
            VLLM_READY=true
            break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "ERROR: vLLM server died during startup; check the slurm .err log."
            exit 1
        fi
        sleep 5
    done
    if [ "${VLLM_READY}" = false ]; then
        echo "ERROR: vLLM server did not become healthy within 600s"
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
fi

echo "[$(date)] Launching training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG_OVERRIDE}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG_OVERRIDE}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

if [ "${NO_VLLM}" = false ] && [ -n "${VLLM_PID:-}" ]; then
    echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM server..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

echo "[$(date)] Job complete."
exit ${TRAIN_EXIT}
