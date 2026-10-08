#!/bin/bash -lt
# ============================================================
# SLURM Submission Script - Multi-Reward GRPO 7B on DollySods
# Cluster: WVU DollySods, A100 80GB nodes (dscog035 / dscog036)
#
# Architecture (1 node, 4 GPUs - partial slice of an A100 node):
#   GPU 0    : vLLM inference server (trl vllm-serve)
#   GPUs 1-3 : accelerate + DeepSpeed ZeRO-3 training (3 train GPUs)
#
# Sized for the gpu_2day partition (48h walltime) with a 4-GPU allocation,
# matching the kind of slice you can typically get without waiting for a
# full node to free up. Effective batch (882) still matches the 1.5B/3B
# final runs for fair comparison.
#
# Estimated wall time for 500 steps: 12-20h (well within 48h).
#
# Default config: 7B + multi-reward + KL-Cov 0.0005
# Override via --export, e.g.:
#   sbatch open_r1_imp/scripts/submit_dollysods_7b.sh
#   sbatch --export=ALL,CONFIG_OVERRIDE=<other>.yaml open_r1_imp/scripts/submit_dollysods_7b.sh
# ============================================================

#SBATCH --job-name=mrew-7b-klcov0005
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --partition=gpu_2day               # 48h walltime; enough for 500 steps of 7B
#SBATCH --nodelist=dscog036                # dscog035 has a broken CUDA driver state as of 2026-04-29; pin to dscog036
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/mrew-7b-klcov0005-%j.out
#SBATCH --error=slurm_logs/mrew-7b-klcov0005-%j.err

set -euo pipefail

# --- Paths --------------------------------------------------
PROJECT_DIR="${HOME}/unsupervised-rl"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-urlenv}"

# --- Configurable defaults (override via --export) ----------
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"          # vLLM gets GPU 0, the rest train
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-open_r1_imp/recipes/final/config_multi_reward_klcov_0.0005_qwen25_7b.yaml}"
ACCEL_CONFIG_OVERRIDE="${ACCEL_CONFIG_OVERRIDE:-open_r1_imp/accelerate_configs/zero3_grad42.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NO_VLLM="${NO_VLLM:-false}"

# --- Conda + CUDA toolchain ---------------------------------
# DollySods compute nodes need module-loaded CUDA so DeepSpeed's
# import-time op-compat check does not raise MissingCUDAException.
module load cuda/12.3 || true
export CUDA_HOME="${CUDA_HOME:-/shared/software/nvidia/cuda/12.3}"

# Activate conda env (Miniconda installed under ~ per HPC docs).
source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# --- HF / Wandb / vLLM env ---------------------------------
# Compute nodes assumed to have NO internet -> use offline modes.
# Pre-download the model on dsis001 once with:
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B')"
# and then set HF_HUB_OFFLINE=1 below. WANDB runs are synced afterwards from a
# login node with `wandb sync slurm_logs/wandb/offline-run-*`.
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${HF_HUB_CACHE}"

# Set HF_TOKEN here (or `huggingface-cli login` once on dsis001) if you
# need to pull a gated model. Qwen2.5-7B is open-weight so this is optional.
# export HF_TOKEN="hf_xxx"

export WANDB_PROJECT="${WANDB_PROJECT:-multi-reward-7b}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/wandb}"
mkdir -p "${WANDB_DIR}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=0      # V0 is the safer path for trl vllm-serve

# NCCL on DollySods (Mellanox/IB; tweak if your nodes differ)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"

# --- Build GPU device lists --------------------------------
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 ${NUM_TRAIN_GPUS}); do
    if [ -n "${TRAIN_GPUS}" ]; then
        TRAIN_GPUS="${TRAIN_GPUS},$i"
    else
        TRAIN_GPUS="$i"
    fi
done

# --- Print job info ----------------------------------------
echo "============================================================"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPUs:          ${SLURM_GPUS_ON_NODE:-8}"
echo "  Conda env:     ${CONDA_ENV_NAME}"
echo "  Config:        ${CONFIG_OVERRIDE}"
echo "  Accelerate:    ${ACCEL_CONFIG_OVERRIDE}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "  HF cache:      ${HF_HUB_CACHE} (offline=${HF_HUB_OFFLINE})"
echo "  Wandb:         ${WANDB_PROJECT} (mode=${WANDB_MODE})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "============================================================"

# ============================================================
# PHASE 1: Start vLLM server on GPU 0 (background)
# ============================================================
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

    echo "[$(date)] Waiting for vLLM server to start (up to 600s; 7B load is slower than 1.5B)..."
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

# ============================================================
# PHASE 2: Launch training via accelerate on GPUs 1-N
# ============================================================
echo "[$(date)] Launching training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG_OVERRIDE}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG_OVERRIDE}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

# ============================================================
# Cleanup: kill vLLM server
# ============================================================
if [ "${NO_VLLM}" = false ] && [ -n "${VLLM_PID:-}" ]; then
    echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM server..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

echo "[$(date)] Job complete."
exit ${TRAIN_EXIT}
