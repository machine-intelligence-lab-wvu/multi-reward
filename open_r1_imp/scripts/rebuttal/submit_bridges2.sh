#!/bin/bash
# ============================================================
# SLURM Submission Script — Multi-Reward GRPO (Open-R1 Pipeline)
# Cluster: PSC BRIDGES-2  (ported from the CSCS Clariden submit.sh)
#
# Architecture (1 node, 4 GPUs on GPU-shared):
#   GPU 0:   vLLM inference server (trl vllm-serve)
#   GPUs 1-3: accelerate + DeepSpeed ZeRO-3 training
#
# Must use H100 (flash-attn 2 needs Ampere+; V100 will NOT work).
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_bridges2.sh                       # default multi-reward
#   sbatch --time=1:00:00 --export=ALL,CONFIG_OVERRIDE=...,MAX_STEPS=20 open_r1_imp/scripts/submit_bridges2.sh   # calibration
#   sbatch --export=ALL,TRAIN_SCRIPT=open_r1_imp/intuitor_train.py,CONFIG_OVERRIDE=... open_r1_imp/scripts/submit_bridges2.sh
#   sbatch --export=ALL,TRAIN_SCRIPT=open_r1_imp/grpo_gt_train.py,CONFIG_OVERRIDE=...  open_r1_imp/scripts/submit_bridges2.sh
# ============================================================

#SBATCH --job-name=grpo-openr1
#SBATCH --account=cis260218p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# ─── Paths (Bridges-2) ───────────────────────────────────────
PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${MR_SCRATCH}"
SCRATCH="${MR_SCRATCH}"

# ─── Configurable defaults (override via --export) ───────────
MODE="${MODE:-full}"                        # lora | full
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-3}"       # GPUs for training (vLLM gets GPU 0)
THINKING="${THINKING:-false}"
NO_VLLM="${NO_VLLM:-false}"
MODEL="${MODEL:-}"                          # empty = use config default
CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-}"
ACCEL_CONFIG_OVERRIDE="${ACCEL_CONFIG_OVERRIDE:-}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-open_r1_imp/multi_reward_train.py}"   # swap for intuitor/grpo_gt
MAX_STEPS="${MAX_STEPS:-}"                   # override max_steps (e.g. 20 for calibration, 200 for rebuttal)
OUTPUT_DIR="${OUTPUT_DIR:-}"                 # override config output_dir (else config default)
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ─── Modules + venv (Bridges-2 replaces `uenv`) ──────────────
source /usr/share/lmod/lmod/init/bash
module load anaconda3/2024.10-1 cuda/12.6.1
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── NCCL (single-node H100; drop Clariden Slingshot 'hsn' settings) ──
export NCCL_DEBUG=WARN
# intra-node NVLink/PCIe only for a 1-node job — leave interface auto-detected.

# ─── PyTorch / CUDA ─────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=5
export TOKENIZERS_PARALLELISM=false

# ─── HF / caches on /ocean (home is only 25G) ───────────────
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"             # set in your shell for gated models (Llama); do NOT hardcode
mkdir -p "${HF_HOME}"

# ─── WandB ───────────────────────────────────────────────────
# export WANDB_PROJECT="${WANDB_PROJECT:-multi-reward-multigpu}"
export WANDB_PROJECT="multi-reward-multigpu"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH}/wandb}"
export WANDB_DISABLE_WEAVE=true
mkdir -p "${WANDB_DIR}"

# ─── vLLM ────────────────────────────────────────────────────
export VLLM_USE_V1=0

# ─── Select config ──────────────────────────────────────────
if [ -n "$CONFIG_OVERRIDE" ]; then
    CONFIG="$CONFIG_OVERRIDE"
else
    echo "ERROR: set CONFIG_OVERRIDE=<path to yaml>"; exit 2
fi

# ─── Assemble EXTRA_ARGS overrides ──────────────────────────
[ "$MODE" = "full" ] && EXTRA_ARGS="${EXTRA_ARGS} --use_peft false"
[ -n "$MODEL" ]      && EXTRA_ARGS="${EXTRA_ARGS} --model_name_or_path ${MODEL}"
[ "$THINKING" = true ] && EXTRA_ARGS="${EXTRA_ARGS} --thinking true"
[ -n "$MAX_STEPS" ]  && EXTRA_ARGS="${EXTRA_ARGS} --max_steps ${MAX_STEPS}"
[ -n "$OUTPUT_DIR" ] && EXTRA_ARGS="${EXTRA_ARGS} --output_dir ${OUTPUT_DIR}"

# ─── Accelerate config ──────────────────────────────────────
if [ -n "$ACCEL_CONFIG_OVERRIDE" ]; then
    ACCEL_CONFIG="$ACCEL_CONFIG_OVERRIDE"
else
    ACCEL_CONFIG="open_r1_imp/accelerate_configs/zero3.yaml"
fi

# ─── wandb run continuity ───────────────────────────────────
if [ -z "${WANDB_RUN_ID:-}" ]; then
    RAW_RUN_NAME=$(grep "^run_name:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
    [ -n "$RAW_RUN_NAME" ] && export WANDB_RUN_ID=$(echo "$RAW_RUN_NAME" | tr -c 'A-Za-z0-9_-' '_')
fi
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# ─── GPU device lists ───────────────────────────────────────
# Derive the split from the GPUs SLURM actually allocated (robust on GPU-shared,
# where the job may not get device ordinals 0..3). GPU[0] -> vLLM, rest -> training.
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#_GPUS[@]}" -lt $((NUM_TRAIN_GPUS + 1)) ]; then
    echo "ERROR: need $((NUM_TRAIN_GPUS + 1)) GPUs but SLURM gave ${#_GPUS[@]} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"; exit 3
fi
VLLM_GPU="${_GPUS[0]}"
TRAIN_GPUS=$(IFS=,; echo "${_GPUS[*]:1:NUM_TRAIN_GPUS}")
# Unset the job-wide mask so our per-process CUDA_VISIBLE_DEVICES assignments take effect.
unset CUDA_VISIBLE_DEVICES

echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}   Node: $(hostname)"
echo "  Config:      ${CONFIG}"
echo "  Train script:${TRAIN_SCRIPT}"
echo "  Accelerate:  ${ACCEL_CONFIG}"
echo "  vLLM GPU:    ${VLLM_GPU} | Train GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "  Extra args:  ${EXTRA_ARGS}"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── PHASE 1: vLLM server on GPU 0 ──
if [ "$NO_VLLM" = false ]; then
    VLLM_MODEL=$(grep "model_name_or_path" "$CONFIG" | head -1 | awk '{print $2}')
    [ -n "$MODEL" ] && VLLM_MODEL="$MODEL"
    echo "[$(date)] Starting vLLM for ${VLLM_MODEL} on GPU ${VLLM_GPU}..."
    # --max_model_len is REQUIRED: without it vLLM 0.9.x defaults to the model's full
    # 131k context, making KV-cache profiling/init take >10 min (observed job failures).
    CUDA_VISIBLE_DEVICES=${VLLM_GPU} trl vllm-serve \
        --model "${VLLM_MODEL}" --dtype bfloat16 \
        --max_model_len 4096 \
        --gpu_memory_utilization 0.85 --enforce_eager true &
    VLLM_PID=$!
    VLLM_READY=false
    for i in $(seq 1 180); do
        curl -s http://localhost:8000/health > /dev/null 2>&1 && { echo "[$(date)] vLLM ready"; VLLM_READY=true; break; }
        kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vLLM died"; exit 1; }
        sleep 5
    done
    if [ "$VLLM_READY" = false ]; then
        echo "ERROR: vLLM not healthy within 900s — aborting instead of launching training against a dead server."
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
fi

# ── PHASE 2: training on GPUs 1..N ──
echo "[$(date)] Launching training (${TRAIN_SCRIPT}) on GPUs ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_TRAIN_GPUS}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}
TRAIN_EXIT=$?

[ "$NO_VLLM" = false ] && [ -n "${VLLM_PID:-}" ] && { kill $VLLM_PID 2>/dev/null || true; }
echo "[$(date)] Job complete (exit ${TRAIN_EXIT})."
exit $TRAIN_EXIT
