#!/bin/bash
# ============================================================
# SLURM Evaluation — Code-reasoning ablation, ckpt-160 only
# 8 ckpts (4 methods × {1.5B, 3B}) × 2 tasks (lcb, cruxeval_o)
# = 16 task-jobs distributed across 4 GH200 GPUs.
#
# Usage:
#   sbatch open_r1_imp/scripts/submit_code_ablation.sh
# ============================================================

#SBATCH --job-name=eval-code
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --partition=low
#SBATCH --output=logs/eval-code-%j.out
#SBATCH --error=logs/eval-code-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"

# Hardcoded for the code-ablation sweep.
OUTPUT_DIR="${MR_OUTPUT_DIR}/eval_results_final/code_eval"
TASKS="lcb,cruxeval_o"
ONLY="grpo_gt,intuitor,multi_reward_final,multi_reward_final_klcov05,grpo_gt_3b,intuitor_3b,multi_reward_final_3b,multi_reward_final_klcov05_3b"
TRAJECTORY=1
MAX_STEP=160
STEP_STRIDE=160

WANDB=1
WANDB_PROJECT="unsupervised-rl-eval"
WANDB_GROUP="code_ablation"
N_SAMPLES=1
TEMPERATURE=0.0
TOP_P=1.0

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# math_verify is harmless to have for math tasks; idempotent install.
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
echo "  Code-ablation evaluation (4 GPUs parallel)"
echo "  Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "  Node:         $(hostname)"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Tasks:        ${TASKS}"
echo "  Only runs:    ${ONLY}"
echo "  n_samples:    ${N_SAMPLES}  temperature:${TEMPERATURE}  top_p:${TOP_P}"
echo "  Trajectory:   ${TRAJECTORY}  max_step:${MAX_STEP}  step_stride:${STEP_STRIDE}"
echo "  Wandb:        ${WANDB} (project=${WANDB_PROJECT}, group=${WANDB_GROUP})"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================================"

ARGS=(
    --num_gpus 4
    --output_dir "${OUTPUT_DIR}"
    --tasks "${TASKS}"
    --only "${ONLY}"
    --enforce_eager
    --trajectory
    --max_step "${MAX_STEP}"
    --step_stride "${STEP_STRIDE}"
)
if [ "${WANDB}" = "1" ]; then
    ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}")
fi

python -u evaluation/orchestrate.py "${ARGS[@]}"

echo ""
echo "=== Aggregating results ==="
python -u evaluation/summarize.py --output_dir "${OUTPUT_DIR}"

echo "[$(date)] code-ablation evaluation done."
