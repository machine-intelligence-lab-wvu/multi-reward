#!/bin/bash
# ============================================================
# SLURM Evaluation — rebuttal checkpoint trajectory, CSCS Clariden
#
# Generic wrapper around evaluation/run_task.py (the single-checkpoint atomic
# unit) for runs that aren't in evaluation/orchestrate.py's hard-coded JOBS
# list — i.e. every new rebuttal run (B6, T2, ...). Loops every checkpoint-N/
# under RUN_DIR and evaluates each on TASKS, one GPU, sequentially — matching
# ICLR-TODO.md's "greedy eval every N steps" trajectory requirement.
#
# Usage:
#   sbatch --export=ALL,RUN_DIR=${MR_OUTPUT_DIR}/rebuttal/B6_llama32_3b_intuitor,RUN_ID=b6_llama32_3b_intuitor \
#       open_r1_imp/scripts/rebuttal/submit_eval_clariden.sh
#
#   # TriviaQA arm (add the actual probe task):
#   sbatch --export=ALL,RUN_DIR=...,RUN_ID=t2_triviaqa_full,TASKS=gsm8k,math500,triviaqa \
#       open_r1_imp/scripts/rebuttal/submit_eval_clariden.sh
# ============================================================

#SBATCH --job-name=rebuttal-eval
#SBATCH --account=a0264
#SBATCH --qos=slowdown
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=low
#SBATCH --output=logs/rebuttal-eval-%j.out
#SBATCH --error=logs/rebuttal-eval-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the training output_dir (contains checkpoint-N/ subdirs)}"
RUN_ID="${RUN_ID:?Set RUN_ID, e.g. b6_llama32_3b_intuitor (used as the trajectory method name)}"
TASKS="${TASKS:-gsm8k,math500}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results_final/rebuttal/${RUN_ID}}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-multi-reward-rebuttal-eval}"

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${PROJECT_DIR}/venv"
mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_DISABLE_WEAVE=true

echo "============================================================"
echo "  Rebuttal checkpoint trajectory eval"
echo "  RUN_DIR:    ${RUN_DIR}"
echo "  RUN_ID:     ${RUN_ID}"
echo "  Tasks:      ${TASKS}"
echo "============================================================"

shopt -s nullglob
CKPTS=("${RUN_DIR}"/checkpoint-*)
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint-* directories found under ${RUN_DIR}" >&2
    exit 1
fi

# Sort numerically by step
mapfile -t CKPTS < <(for c in "${CKPTS[@]}"; do echo "$c"; done | sort -t- -k2 -n)

for CKPT in "${CKPTS[@]}"; do
    STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
    RN="${RUN_ID}_ckpt${STEP}"
    echo "[$(date)] Evaluating step ${STEP} (${CKPT})..."
    ARGS=(--checkpoint "$CKPT" --run_name "$RN" --tasks "$TASKS"
          --output_dir "$OUTPUT_DIR" --trajectory --enforce_eager
          --wandb --wandb_project "$WANDB_PROJECT_NAME" --wandb_group "$RUN_ID")
    [ -n "$MAX_SAMPLES" ] && ARGS+=(--max_samples "$MAX_SAMPLES")
    python -u evaluation/run_task.py "${ARGS[@]}"
done

echo "[$(date)] Trajectory eval complete for ${RUN_ID} → ${OUTPUT_DIR}"
