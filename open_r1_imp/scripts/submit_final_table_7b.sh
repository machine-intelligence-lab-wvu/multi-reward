#!/bin/bash
# ============================================================
# SLURM — Final-checkpoint comparison eval (Qwen2.5-7B)
# Cluster: CSCS Clariden (GH200 120GB GPUs)
#
# Runs 4 methods × 4 tasks = 16 (method, task) jobs, distributed
# round-robin across 4 GPUs.
# Per-task N: gsm8k=8, math500=16, aime24=32, aime25=32 @ T=1.
#
# Output: eval_results_final/final_table/7b/
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_final_table_7b.sh
# ============================================================

#SBATCH --job-name=ftab-7b
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/ftab-7b-%j.out
#SBATCH --error=logs/ftab-7b-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

METHODS="${METHODS:-}"
TASKS="${TASKS:-}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-final_table_7b_v1}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=18

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_DISABLE_WEAVE=true
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_GROUP="${WANDB_GROUP}"

echo "============================================================"
echo "  Final-table eval — Qwen2.5-7B (4 GPUs parallel)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Methods:      ${METHODS:-<all>}"
echo "  Tasks:        ${TASKS:-<all>}"
echo "  Overwrite:    ${OVERWRITE}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(--scale 7b --num_gpus 4)
[ -n "${METHODS}" ] && ARGS+=(--methods "${METHODS}")
[ -n "${TASKS}" ]   && ARGS+=(--tasks "${TASKS}")
[ "${OVERWRITE}" = "1" ] && ARGS+=(--overwrite)
[ "${WANDB}" = "1" ] && ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")

python -u evaluation/final_table.py "${ARGS[@]}"

echo ""
echo "=== Building markdown + pass@k plot ==="
python -u evaluation/summarize_final_table.py --scale 7b || true

echo "[$(date)] final-table 7B done."
