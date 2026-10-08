#!/bin/bash
# ============================================================
# SLURM Submission Script — Multi-Reward GRPO LoRA (param-space gradcorr)
# Cluster: CSCS Clariden (GH200 120GB GPUs)
#
# Architecture (1 node, 4 GPUs):
#   GPU 0    : vLLM inference server (trl vllm-serve)
#   GPUs 1-3 : accelerate + DeepSpeed ZeRO-2 training (3 train GPUs)
#
# Purpose:
#   Run the *parameter-space* gradient-correlation probe (cluster vs
#   self-certainty) under LoRA. Full-FT param-space probing OOM'd at 7B;
#   LoRA shrinks trainable params to ~tens of M so the 2-3 separate
#   torch.autograd.grad calls (with retain_graph=True) fit in memory.
#
# Differences from submit_logps_gradcorr.sh:
#   - Trains via open_r1_imp/multi_reward_train.py (param-space probe in
#     multi_reward_trainer.py:467-525), NOT the logps variant.
#   - Default config is the LoRA klcov_0.05 1.5B recipe.
#   - Uses zero2_grad14.yaml (matches yaml grad_accum=14 — the bare zero2.yaml
#     hard-codes 32, which would fail DeepSpeed's batch assertion).
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_lora.sh
#   sbatch open_r1_imp/scripts/submit_lora.sh \
#       --export=ALL,GRAD_CORR_EVERY=10
#   sbatch open_r1_imp/scripts/submit_lora.sh \
#       --export=ALL,CONFIG_OVERRIDE=path/to/other_lora_config.yaml
# ============================================================

#SBATCH --job-name=mrew-lora
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/mrew-lora-%j.out
#SBATCH --error=logs/mrew-lora-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────
PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# ─── Configurable defaults (override via --export) ───────────
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-open_r1_imp/recipes/final/config_multi_reward_klcov_0.05_qwen25_1.5b_lora.yaml}"
ACCEL_CONFIG_OVERRIDE="${ACCEL_CONFIG_OVERRIDE:-open_r1_imp/accelerate_configs/ddp.yaml}"
MODEL="${MODEL:-}"
NO_VLLM="${NO_VLLM:-false}"
GRAD_CORR_EVERY="${GRAD_CORR_EVERY:-1}"
GRAD_CORR_NUM_TENSORS="${GRAD_CORR_NUM_TENSORS:-32}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Force gradcorr probe on (the new probe is the whole point of this job)
EXTRA_ARGS="${EXTRA_ARGS} --grad_corr_enabled true --grad_corr_every_steps ${GRAD_CORR_EVERY} --grad_corr_num_tensors ${GRAD_CORR_NUM_TENSORS}"

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

# Keep Triton autotune cache off the (quota-limited) home fs
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MR_CACHE_DIR}/triton}"
mkdir -p "${TRITON_CACHE_DIR}"

# ─── HF / Wandb ──────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your shell before submitting, e.g. export HF_TOKEN=hf_xxx}"
export WANDB_PROJECT="${WANDB_PROJECT:-multi-reward-multigpu}"
export WANDB_DISABLE_WEAVE=true
# Keep wandb run dir off the (quota-limited) home fs
export WANDB_DIR="${WANDB_DIR:-${MR_OUTPUT_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

# ─── WandB run continuity (deterministic ID from run_name) ──
# Re-submitting the same CONFIG resumes the same wandb run.
if [ -z "${WANDB_RUN_ID:-}" ]; then
    RAW_RUN_NAME=$(grep "^run_name:" "${CONFIG_OVERRIDE}" | head -1 | awk '{print $2}' | tr -d '"')
    if [ -n "${RAW_RUN_NAME}" ]; then
        export WANDB_RUN_ID=$(echo "${RAW_RUN_NAME}" | tr -c 'A-Za-z0-9_-' '_')
    fi
fi
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# ─── vLLM ────────────────────────────────────────────────────
export VLLM_USE_V1=0

# Pass --use_peft true through to the train script (matches yaml)
EXTRA_ARGS="${EXTRA_ARGS} --use_peft true"

# Model override
if [ -n "$MODEL" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model_name_or_path ${MODEL}"
fi

# ─── Build GPU device lists ─────────────────────────────────
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 ${NUM_TRAIN_GPUS}); do
    if [ -n "${TRAIN_GPUS}" ]; then
        TRAIN_GPUS="${TRAIN_GPUS},$i"
    else
        TRAIN_GPUS="$i"
    fi
done

# ─── Print job info ──────────────────────────────────────────
echo "============================================================"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "  Node:          $(hostname)"
echo "  GPUs:          ${SLURM_GPUS_ON_NODE:-4}"
echo "  Config:        ${CONFIG_OVERRIDE}"
echo "  Accelerate:    ${ACCEL_CONFIG_OVERRIDE}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "  Probe every:   ${GRAD_CORR_EVERY} steps  (param-space, last ${GRAD_CORR_NUM_TENSORS} LoRA tensors)"
echo "  Train script:  open_r1_imp/multi_reward_train.py"
echo "  EXTRA_ARGS:    ${EXTRA_ARGS}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: Start vLLM server on GPU 0 (background)
# ════════════════════════════════════════════════════════════════
if [ "${NO_VLLM}" = false ]; then
    VLLM_MODEL=$(grep "model_name_or_path" "${CONFIG_OVERRIDE}" | head -1 | awk '{print $2}')
    if [ -n "${MODEL}" ]; then
        VLLM_MODEL="${MODEL}"
    fi

    echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.8 \
        --enforce_eager true &
    VLLM_PID=$!
    echo "[$(date)] vLLM server PID: ${VLLM_PID}"

    echo "[$(date)] Waiting for vLLM server to start..."
    VLLM_READY=false
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "[$(date)] vLLM server ready!"
            VLLM_READY=true
            break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "ERROR: vLLM server died!"
            exit 1
        fi
        sleep 5
    done
    if [ "${VLLM_READY}" = false ]; then
        echo "ERROR: vLLM server did not become healthy within 300s"
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
fi

# ════════════════════════════════════════════════════════════════
# PHASE 2: Launch training on GPUs 1-N
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching training on GPUs: ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG_OVERRIDE}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG_OVERRIDE}" \
    ${EXTRA_ARGS}

TRAIN_EXIT=$?

# ════════════════════════════════════════════════════════════════
# Cleanup: kill vLLM server
# ════════════════════════════════════════════════════════════════
if [ "${NO_VLLM}" = false ] && [ -n "${VLLM_PID:-}" ]; then
    echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT}). Stopping vLLM server..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

echo "[$(date)] Job complete."
exit ${TRAIN_EXIT}
