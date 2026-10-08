#!/bin/bash
# ============================================================
# SLURM — Trajectory Evaluation for cluster_only (Qwen2.5-1.5B)
# Cluster: CSCS Clariden (GH200 GPUs)
#
# Evaluates checkpoints 20, 40, ..., 280 of the cluster-only run
# (cluster_only_noklcov_qwen25_1.5b) on GSM8K / MATH500 / MMLU-Pro.
# 14 ckpts × 3 tasks = 42 evaluations, distributed across 4 GPUs.
#
# Output goes straight into eval_results_final/single_multi-reward/
# so the build/plot scripts can consume it without an extra copy.
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_trajectory_cluster_only.sh
#
#   # Smoke test (first 20 samples per task):
#   sbatch --export=ALL,MAX_SAMPLES=20 \
#          open_r1_imp/scripts/submit_trajectory_cluster_only.sh
#
#   # Different step cap / stride:
#   sbatch --export=ALL,MAX_STEP=200,STEP_STRIDE=40 \
#          open_r1_imp/scripts/submit_trajectory_cluster_only.sh
# ============================================================

#SBATCH --job-name=traj-cluster-only
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/traj-cluster-only-%j.out
#SBATCH --error=logs/traj-cluster-only-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# Configurable (override via --export)
OUTPUT_DIR="${OUTPUT_DIR:-eval_results_final/single_multi-reward}"
TASKS="${TASKS:-gsm8k,math500,mmlu_pro}"
ONLY="${ONLY:-cluster_only}"
MAX_STEP="${MAX_STEP:-280}"
STEP_STRIDE="${STEP_STRIDE:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-unsupervised-rl-eval}"
WANDB_GROUP="${WANDB_GROUP:-trajectory_single_multi_qwen15b}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Ensure math_verify for paper-exact math scoring
python -c "import math_verify, latex2sympy2_extended" 2>/dev/null || \
    pip install --quiet math-verify latex2sympy2-extended || true

# vLLM / LoRA-safe env
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
echo "  Trajectory eval — cluster_only (Qwen2.5-1.5B, 4 GPUs)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Tasks:        ${TASKS}"
echo "  Only methods: ${ONLY}"
echo "  Step range:   stride=${STEP_STRIDE}, max=${MAX_STEP}"
echo "  Max samples:  ${MAX_SAMPLES:-<full>}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(
    --trajectory
    --num_gpus 4
    --output_dir "${OUTPUT_DIR}"
    --tasks "${TASKS}"
    --only "${ONLY}"
    --max_step "${MAX_STEP}"
    --step_stride "${STEP_STRIDE}"
    --enforce_eager
)
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
    ARGS+=(--overwrite)
fi
if [ "${WANDB}" = "1" ]; then
    ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")
fi

python -u evaluation/orchestrate.py "${ARGS[@]}"

echo ""
echo "[$(date)] Eval complete — per-checkpoint dirs:"
find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "cluster_only_ckpt*" | sort | while read -r d; do
    n=$(find "$d" -maxdepth 1 -name '*.json' | wc -l)
    echo "  ${d}  (${n} JSONs)"
done

echo "[$(date)] Done."
