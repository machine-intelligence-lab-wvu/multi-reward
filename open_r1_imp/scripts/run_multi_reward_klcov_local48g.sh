#!/bin/bash
# ============================================================
# Local Launch Script — Multi-Reward GRPO (INTUITOR self-certainty
# + cluster-voting) with KL-Cov, Qwen2.5-1.5B, beta_cov = 0.02
#
# Host: WVU workstation, 4 x RTX PRO 5000 Blackwell 48GB (sm_120)
#   GPU 0    : vLLM rollout server (trl vllm-serve)
#   GPUs 1-3 : accelerate + DeepSpeed ZeRO-3 training
#
# No SLURM — this runs directly. For a long unattended run:
#   nohup bash open_r1_imp/scripts/run_multi_reward_klcov_local48g.sh \
#       > logs/klcov002-1.5b.out 2>&1 &
#
# Knobs (env vars):
#   CONFIG          config yaml               (default: the local48g klcov 0.02 1.5B)
#   ACCEL_CONFIG    accelerate/deepspeed yaml (default: zero3_grad42.yaml — must
#                                              match gradient_accumulation_steps)
#   NUM_TRAIN_GPUS  training GPUs             (default: 3)
#   ATTN            attn implementation       (default: from config; set to `sdpa`
#                                              if flash-attn has no sm_120 kernels)
#   SMOKE           1 = 6-step dry run into outputs/smoke_* (default: 0)
#   RESUME          path to a checkpoint-N dir to resume from
#   NO_VLLM         true = skip the vLLM server (colocated generation)
#   ENFORCE_EAGER   true = vLLM without cudagraphs (lower mem, slower)
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

CONFIG="${CONFIG:-open_r1_imp/recipes/final/config_multi_reward_klcov_0.02_qwen25_1.5b_local48g.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-open_r1_imp/accelerate_configs/zero3_grad42.yaml}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
NO_VLLM="${NO_VLLM:-false}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
SMOKE="${SMOKE:-0}"
ATTN="${ATTN:-}"
RESUME="${RESUME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs

# ─── Environment ─────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ─── CUDA toolkit ───────────────────────────────────────────
# This host has no system CUDA toolkit; a matching 12.8 one (nvcc + cudart +
# cccl + nvrtc, from NVIDIA's redist archive) lives alongside the venv. DeepSpeed
# refuses to import without a usable CUDA_HOME. Kept outside venv/ so that
# rebuilding the venv does not delete it.
export CUDA_HOME="${CUDA_HOME:-${PROJECT_DIR}/.local-cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# ─── PyTorch / CUDA ─────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ─── NCCL (single node, 4 local GPUs, no InfiniBand) ────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# ─── WandB (offline; sync later with `wandb sync wandb/offline-run-*`) ──
export WANDB_PROJECT="${WANDB_PROJECT:-multi-reward-local48g}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb}"
export WANDB_DISABLE_WEAVE=true
mkdir -p "${WANDB_DIR}"

# Deterministic wandb run id so a resume continues the same run.
if [ -z "${WANDB_RUN_ID:-}" ]; then
    RAW_RUN_NAME=$(grep "^run_name:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
    [ -n "$RAW_RUN_NAME" ] && export WANDB_RUN_ID=$(echo "$RAW_RUN_NAME" | tr -c 'A-Za-z0-9_-' '_')
fi
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# NOTE: deliberately NOT setting VLLM_USE_V1=0 — the V0 engine was removed in
# vLLM 0.11, and the cluster scripts' `export VLLM_USE_V1=0` would break startup.

# ─── Optional overrides ─────────────────────────────────────
[ -n "$ATTN" ]   && EXTRA_ARGS="${EXTRA_ARGS} --attn_implementation ${ATTN}"
[ -n "$RESUME" ] && EXTRA_ARGS="${EXTRA_ARGS} --resume_from_checkpoint ${RESUME}"
if [ "$SMOKE" = "1" ]; then
    # Keep per_device=7 and max_completion_length=3072 so peak GPU memory matches
    # the real run; only shorten the accumulation loop. grad_accum must match the
    # deepspeed config, hence the paired zero3_grad1.yaml.
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3_grad1.yaml"
    EXTRA_ARGS="${EXTRA_ARGS} --gradient_accumulation_steps 1 --max_steps 3 --save_steps 3 --save_total_limit 1"
    EXTRA_ARGS="${EXTRA_ARGS} --output_dir ${PROJECT_DIR}/outputs/smoke_klcov002_1.5b"
    export WANDB_RUN_ID="smoke_klcov002_1_5b"
fi

# ─── GPU assignment ─────────────────────────────────────────
VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 "$NUM_TRAIN_GPUS"); do
    [ -n "$TRAIN_GPUS" ] && TRAIN_GPUS="${TRAIN_GPUS},$i" || TRAIN_GPUS="$i"
done

VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')

echo "============================================================"
echo "  Multi-Reward (self-certainty + cluster-voting) + KL-Cov"
echo "  Host:          $(hostname)   $(date)"
echo "  Config:        ${CONFIG}"
echo "  Accelerate:    ${ACCEL_CONFIG}"
echo "  Model:         ${VLLM_MODEL}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "  Smoke:         ${SMOKE} | Extra args: ${EXTRA_ARGS:-<none>}"
echo "  Wandb:         ${WANDB_PROJECT} (mode=${WANDB_MODE}, id=${WANDB_RUN_ID:-auto})"
echo "  HF cache:      ${HF_HUB_CACHE}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: vLLM rollout server on GPU 0
# ════════════════════════════════════════════════════════════════
VLLM_PID=""
cleanup() {
    [ -z "${VLLM_PID}" ] && return 0
    # vllm-serve forks a VLLM::EngineCore child that holds the GPU memory and
    # does NOT die with its parent. Signal the whole process group (vLLM is
    # started under setsid, so VLLM_PID is its process-group leader) or ~42GB
    # stays pinned on GPU 0 after the run.
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date)] Stopping vLLM server (pgid ${VLLM_PID})..."
        kill -TERM -"${VLLM_PID}" 2>/dev/null || kill -TERM "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"${VLLM_PID}" 2>/dev/null || kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
    # Backstop: anything still holding GPU 0 that we own.
    local leftover
    leftover=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    for p in ${leftover}; do
        if [ -n "$p" ] && [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
            case "$(ps -o cmd= -p "$p" 2>/dev/null)" in
                *VLLM*|*vllm*) echo "[$(date)] Killing leftover vLLM pid $p"; kill -9 "$p" 2>/dev/null || true ;;
            esac
        fi
    done
}
trap cleanup EXIT INT TERM

if [ "$NO_VLLM" = false ]; then
    EAGER_FLAG=""
    [ "$ENFORCE_EAGER" = true ] && EAGER_FLAG="--enforce_eager true"

    echo "[$(date)] Starting vLLM server on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
    # setsid: give vLLM its own process group so cleanup() can signal the
    # EngineCore child too (see cleanup above).
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} setsid trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 \
        ${EAGER_FLAG} > logs/vllm-serve.log 2>&1 &
    VLLM_PID=$!
    echo "[$(date)] vLLM server PID: ${VLLM_PID} (log: logs/vllm-serve.log)"

    echo "[$(date)] Waiting for vLLM server (up to 600s)..."
    VLLM_READY=false
    for i in $(seq 1 120); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "[$(date)] vLLM server ready after $((i*5))s."
            VLLM_READY=true
            break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "ERROR: vLLM server died during startup. Last 40 lines:"
            tail -40 logs/vllm-serve.log
            exit 1
        fi
        sleep 5
    done
    if [ "$VLLM_READY" = false ]; then
        echo "ERROR: vLLM server did not become healthy within 600s. Last 40 lines:"
        tail -40 logs/vllm-serve.log
        exit 1
    fi
fi

# ════════════════════════════════════════════════════════════════
# PHASE 2: training on GPUs 1..N
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching training on GPUs: ${TRAIN_GPUS}..."
set +e
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    open_r1_imp/multi_reward_train.py \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}
TRAIN_EXIT=$?
set -e

echo "[$(date)] Training finished (exit code: ${TRAIN_EXIT})."
exit ${TRAIN_EXIT}
