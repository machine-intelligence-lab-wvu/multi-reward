#!/bin/bash
# ============================================================
# SLURM Submission Script — Checkpoint Evaluation
# Cluster: CSCS Clariden (GH200 120GB GPUs)
#
# Only needs 1 GPU — runs eval_checkpoint.py with greedy or
# majority voting on GSM8K and MATH500.
#
# Usage:
#   # Single checkpoint eval:
#   sbatch submit_eval.sh
#   sbatch --export=ALL,CHECKPOINT=/path/to/checkpoint-200 submit_eval.sh
#
#   # ALL checkpoints in a dir (mode collapse tracking):
#   sbatch --export=ALL,EVAL_ALL=true,OUTPUT_DIR=/path/to/outputs,MODEL=Qwen/Qwen2.5-1.5B,THINKING=false,DATASETS="gsm8k,math500",WANDB_RUN_NAME="my-run" submit_eval.sh
# ============================================================

#SBATCH --job-name=grpo-eval
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=low
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────
PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# ─── Configurable defaults (override via --export) ───────────
CHECKPOINT="${CHECKPOINT:-${MR_OUTPUT_DIR}/open_r1_full_qwen3/checkpoint-300}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
BASE_ONLY="${BASE_ONLY:-false}"
THINKING="${THINKING:-true}"
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DATASETS="${DATASETS:-gsm8k,math500}"
WANDB="${WANDB:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-multi-reward-eval}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# ─── Mode: single checkpoint or all checkpoints ─────────────
# Set EVAL_ALL=true to evaluate ALL checkpoints in OUTPUT_DIR
EVAL_ALL="${EVAL_ALL:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ─── Create directories ─────────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/eval_results_final"

# ─── Activate virtual environment ────────────────────────────
source "${VENV_DIR}/bin/activate"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ─── WandB ───────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_DISABLE_WEAVE=true

# ════════════════════════════════════════════════════════════════
# MODE: Evaluate ALL checkpoints in a directory
# ════════════════════════════════════════════════════════════════
if [ "${EVAL_ALL}" = "true" ]; then
    echo "============================================================"
    echo "  EVAL ALL CHECKPOINTS"
    echo "  Output dir:    ${OUTPUT_DIR}"
    echo "  Model:         ${MODEL}"
    echo "  Datasets:      ${DATASETS}"
    echo "  Max samples:   ${MAX_SAMPLES:-all}"
    echo "  W&B run name:  ${WANDB_RUN_NAME:-auto}"
    echo "============================================================"

    # Convert comma-separated to space-separated for --dataset nargs
    DATASET_ARGS=$(echo "${DATASETS}" | tr ',' ' ')

    CMD="python evaluation/eval_during_training.py \
        --output_dir ${OUTPUT_DIR} \
        --model_name ${MODEL} \
        --dataset ${DATASET_ARGS} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --no_watch"

    if [ -n "${MAX_SAMPLES}" ]; then
        CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    fi

    if [ "${THINKING}" = "true" ]; then
        CMD="${CMD} --thinking"
    fi

    if [ "${WANDB}" = "true" ]; then
        CMD="${CMD} --wandb_project ${WANDB_PROJECT_NAME}"
        if [ -n "${WANDB_RUN_NAME}" ]; then
            CMD="${CMD} --wandb_run_name ${WANDB_RUN_NAME}"
        fi
    fi

    echo "[$(date)] Running: ${CMD}"
    eval ${CMD}
    echo "[$(date)] All checkpoint evaluations complete."
    exit 0
fi

# ════════════════════════════════════════════════════════════════
# MODE: Evaluate a SINGLE checkpoint
# ════════════════════════════════════════════════════════════════

# ─── Build common args ───────────────────────────────────────
COMMON_ARGS="--model_name ${MODEL} --n_samples ${N_SAMPLES} --temperature ${TEMPERATURE} --max_new_tokens ${MAX_NEW_TOKENS} --output_dir eval_results_final"

if [ "${THINKING}" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --thinking"
fi

if [ "${BASE_ONLY}" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --base_only"
fi

if [ -n "${CHECKPOINT}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --checkpoint ${CHECKPOINT}"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --max_samples ${MAX_SAMPLES}"
fi

if [ "${WANDB}" = "true" ]; then
    COMMON_ARGS="${COMMON_ARGS} --wandb --wandb_project ${WANDB_PROJECT_NAME}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        COMMON_ARGS="${COMMON_ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
fi

# ─── Run eval on each dataset (in parallel) ─────────────────
IFS=',' read -ra DS_LIST <<< "${DATASETS}"

PIDS=()
for DS in "${DS_LIST[@]}"; do
    echo ""
    echo "============================================================"
    echo "Launching eval on ${DS} (background)"
    echo "============================================================"
    python evaluation/eval_checkpoint.py --dataset "${DS}" ${COMMON_ARGS} &
    PIDS+=($!)
done

# Wait for all parallel evals to finish
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

echo ""
echo "============================================================"
echo "All evaluations complete. Results in eval_results_final/"
echo "============================================================"
ls -la eval_results_final/

if [ "$FAIL" -ne 0 ]; then
    echo "WARNING: One or more evaluations failed."
    exit 1
fi
