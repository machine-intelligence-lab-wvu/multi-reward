#!/bin/bash
# ============================================================
# Run ONE rebuttal config locally (no SLURM) on the WVU box:
#   4 x RTX PRO 5000 Blackwell 48GB — GPU 0 = vLLM, GPUs 1-3 = ZeRO-3.
#
#   bash open_r1_imp/scripts/rebuttal/run_local48g.sh A2_qwen3_17b_full
#
# Local analogue of scripts/rebuttal/launch.sh: same entry-script routing
# (multi / intuitor / grpo), but reads configs from recipes/rebuttal_local48g/
# (per_device 7 x grad_accum 42 instead of 21 x 14 — same effective batch 882,
# because the 48GB cards cannot hold the (B,S,V) logits tensor at B=21).
#
# Env knobs: MAX_STEPS, EXTRA_ARGS, NUM_TRAIN_GPUS, RESUME, ATTN, NO_VLLM.
# ============================================================
set -euo pipefail

ID="${1:?usage: run_local48g.sh <config_id>}"; shift || true

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"
CONFIG="open_r1_imp/recipes/rebuttal_local48g/${ID}.yaml"

cd "${PROJECT_DIR}"
[ -f "${CONFIG}" ] || { echo "ERROR: no config ${CONFIG}"; exit 1; }

# ─── Entry script by method (mirrors launch.sh; intuitor checked first) ──
case "$ID" in
  *intuitor*)                TRAIN=open_r1_imp/intuitor_train.py ;;
  *full*|*multireward*|B7_*) TRAIN=open_r1_imp/multi_reward_train.py ;;
  *)                         TRAIN=open_r1_imp/grpo_gt_train.py ;;
esac

# grad_accum in the accelerate config MUST match the yaml (42 for these).
ACCEL_CONFIG="${ACCEL_CONFIG:-open_r1_imp/accelerate_configs/zero3_grad42.yaml}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"
NO_VLLM="${NO_VLLM:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ATTN="${ATTN:-}"
RESUME="${RESUME:-}"

mkdir -p logs outputs/rebuttal

# ─── Environment ─────────────────────────────────────────────
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# No system CUDA toolkit on this host; DeepSpeed needs a usable CUDA_HOME.
export CUDA_HOME="${CUDA_HOME:-${PROJECT_DIR}/.local-cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# Same project as the Bridges-2 pipeline (submit_bridges2.sh) so local and cluster
# runs can be compared side by side. But submit_bridges2.sh derives its run id from
# the config's `run_name`, so without a suffix a local run and a cluster run would
# both display as e.g. "a2-qwen3-17b-full". RUN_SUFFIX keeps them distinguishable —
# it also records that these local runs are the mu=2 / per_device-7 variant.
export WANDB_PROJECT="${WANDB_PROJECT:-rebuttal-multireward}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/wandb}"
export WANDB_DISABLE_WEAVE=true
RUN_SUFFIX="${RUN_SUFFIX:--local48g-mu2}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(echo "${ID}${RUN_SUFFIX}" | tr -c 'A-Za-z0-9_-' '_')}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
mkdir -p "${WANDB_DIR}"

# Mirror the suffix into the HF/TRL run_name so the wandb display name matches.
CFG_RUN_NAME=$(grep "^run_name:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
[ -n "${CFG_RUN_NAME}" ] && EXTRA_ARGS="${EXTRA_ARGS} --run_name ${CFG_RUN_NAME}${RUN_SUFFIX}"

# NOT setting VLLM_USE_V1=0 — the V0 engine was removed in vLLM 0.11.

[ -n "$ATTN" ]       && EXTRA_ARGS="${EXTRA_ARGS} --attn_implementation ${ATTN}"
[ -n "$RESUME" ]     && EXTRA_ARGS="${EXTRA_ARGS} --resume_from_checkpoint ${RESUME}"
[ -n "${MAX_STEPS:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --max_steps ${MAX_STEPS}"

VLLM_GPU=0
TRAIN_GPUS=""
for i in $(seq 1 "$NUM_TRAIN_GPUS"); do
    [ -n "$TRAIN_GPUS" ] && TRAIN_GPUS="${TRAIN_GPUS},$i" || TRAIN_GPUS="$i"
done

VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')
OUT_DIR=$(grep "^output_dir:" "$CONFIG" | head -1 | awk '{print $2}')

echo "============================================================"
echo "  Rebuttal run (local): ${ID}"
echo "  Started:       $(date)"
echo "  Config:        ${CONFIG}"
echo "  Entry:         ${TRAIN}"
echo "  Accelerate:    ${ACCEL_CONFIG}"
echo "  Model:         ${VLLM_MODEL}"
echo "  Output:        ${OUT_DIR}"
echo "  vLLM GPU:      ${VLLM_GPU} | Training GPUs: ${TRAIN_GPUS}"
echo "  Extra args:    ${EXTRA_ARGS:-<none>}"
echo "  Wandb:         ${WANDB_PROJECT} (${WANDB_MODE}, id=${WANDB_RUN_ID})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "============================================================"

# ════════════════════════════════════════════════════════════════
# PHASE 1: vLLM rollout server on GPU 0
# ════════════════════════════════════════════════════════════════
VLLM_PID=""
cleanup() {
    [ -z "${VLLM_PID}" ] && return 0
    # vllm-serve forks a VLLM::EngineCore child that does NOT die with its
    # parent; signal the whole process group (started under setsid) or ~42GB
    # stays pinned on GPU 0 and the next run in the queue cannot start.
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date)] Stopping vLLM (pgid ${VLLM_PID})..."
        kill -TERM -"${VLLM_PID}" 2>/dev/null || kill -TERM "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 "${VLLM_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL -"${VLLM_PID}" 2>/dev/null || kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
            case "$(ps -o cmd= -p "$p" 2>/dev/null)" in
                *VLLM*|*vllm*) echo "[$(date)] Killing leftover vLLM pid $p"; kill -9 "$p" 2>/dev/null || true ;;
            esac
        fi
    done
}
trap cleanup EXIT INT TERM

if [ "$NO_VLLM" = false ]; then
    echo "[$(date)] Starting vLLM on GPU ${VLLM_GPU} for ${VLLM_MODEL}..."
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} setsid trl vllm-serve \
        --model "${VLLM_MODEL}" \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 > "logs/vllm-${ID}.log" 2>&1 &
    VLLM_PID=$!
    echo "[$(date)] vLLM PID ${VLLM_PID} (log: logs/vllm-${ID}.log)"

    VLLM_READY=false
    for i in $(seq 1 120); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "[$(date)] vLLM ready after $((i*5))s."; VLLM_READY=true; break
        fi
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "ERROR: vLLM died during startup. Last 40 lines:"; tail -40 "logs/vllm-${ID}.log"; exit 1
        fi
        sleep 5
    done
    if [ "$VLLM_READY" = false ]; then
        echo "ERROR: vLLM not healthy within 600s."; tail -40 "logs/vllm-${ID}.log"; exit 1
    fi
fi

# ════════════════════════════════════════════════════════════════
# PHASE 2: training on GPUs 1..N
# ════════════════════════════════════════════════════════════════
echo "[$(date)] Launching ${TRAIN} on GPUs ${TRAIN_GPUS}..."
set +e
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    "${TRAIN}" \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}
TRAIN_EXIT=$?
set -e

echo "[$(date)] ${ID} finished (exit ${TRAIN_EXIT})."
exit ${TRAIN_EXIT}
