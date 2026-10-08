#!/bin/bash
# ============================================================
# Checkpoint-sweep evaluation on Bridges-2 (port of run_eval_local48g.sh).
#
# One GPU, inference-only. Sweeps every saved checkpoint (step 20..MAX_STEP)
# of each run in RUNS with evaluation/run_task.py (greedy, bf16,
# max_model_len 16384 — identical settings to the paper's eval suite), plus
# an optional base-model row. Idempotent: existing
# {OUT}/{name}/{name}__{task}.json are skipped unless OVERWRITE=1.
#
# Usage (defaults = the B6 Llama suite):
#   sbatch open_r1_imp/scripts/rebuttal/submit_eval_bridges2.sh
#   IMPORTANT: sbatch --export splits its argument on COMMAS, so a value that
#   itself contains commas (TASKS=gsm8k,math500) is truncated. Export in the
#   shell and pass plain --export=ALL:
#     export TASKS=gsm8k,math500 RUNS="t2_full:T2_triviaqa_full_qwen25_15b" BASE_MODEL=
#     sbatch --export=ALL open_r1_imp/scripts/rebuttal/submit_eval_bridges2.sh
#   sbatch --gres=gpu:h100-80:1 ... (faster queue slot if L40S pool is busy)
#
# Env knobs: RUNS ("short:dirname" pairs under $RUNS_DIR), TASKS, BASE_MODEL,
#   BASE_NAME, STEPS, OUT, MAX_SAMPLES, SAMPLE_SEED, OVERWRITE.
# MMLU-Pro protocol note: the A2 suite evaluated mmlu_pro only at step 200
#   with MAX_SAMPLES=2000 + a fixed SAMPLE_SEED (its test set is ordered by
#   category — never subsample without a seed).
# ============================================================

#SBATCH --job-name=eval-sweep
#SBATCH --account=cis260218p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -uo pipefail

PROJECT_DIR="${HOME}/projects/unsupervised-rl"
VENV_DIR="${MR_SCRATCH}"
SCRATCH="${MR_SCRATCH}"
RUNS_DIR="${RUNS_DIR:-${SCRATCH}/outputs/rebuttal}"

# NB: ${VAR-default} (no colon) so an explicitly EMPTY value means "none"
# rather than falling back to the default.
RUNS="${RUNS-b6_full:B6_llama32_3b_full b6_intuitor:B6_llama32_3b_intuitor b6_grpo_gt:B6_llama32_3b_grpo_gt}"
BASE_MODEL="${BASE_MODEL-meta-llama/Llama-3.2-3B}"   # explicitly empty skips the base row
BASE_NAME="${BASE_NAME:-llama32_3b_base}"
TASKS="${TASKS:-gsm8k,math500}"
STEPS="${STEPS:-20 40 60 80 100 120 140 160 180 200}"
OUT="${OUT:-eval-rebuttal-results}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
OVERWRITE="${OVERWRITE:-0}"

source /usr/share/lmod/lmod/init/bash
module load anaconda3/2024.10-1 cuda/12.6.1
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/eval" "${OUT}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-}"     # gated base models (Llama); set in your shell
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

EXTRA=""
[ -n "${MAX_SAMPLES}" ] && EXTRA="${EXTRA} --max_samples ${MAX_SAMPLES}"
[ -n "${SAMPLE_SEED}" ] && EXTRA="${EXTRA} --sample_seed ${SAMPLE_SEED}"
[ "${OVERWRITE}" = "1" ] && EXTRA="${EXTRA} --overwrite"

JOBS=()
[ -n "${BASE_MODEL}" ] && JOBS+=("${BASE_NAME}|${BASE_MODEL}")
for spec in ${RUNS}; do
    short="${spec%%:*}"; dirname="${spec#*:}"
    for step in ${STEPS}; do
        d="${RUNS_DIR}/${dirname}/checkpoint-${step}"
        [ -d "$d" ] && JOBS+=("${short}_ckpt${step}|${d}")
    done
done

echo "============================================================"
echo "  Eval sweep — ${#JOBS[@]} jobs | tasks=${TASKS} | out=${OUT}"
for j in "${JOBS[@]}"; do echo "   - ${j%%|*}  <-  ${j##*|}"; done
echo "============================================================"

FAILED=0
for job in "${JOBS[@]}"; do
    name="${job%%|*}"; ckpt="${job##*|}"
    echo "[$(date +%H:%M:%S)] START ${name}"
    python evaluation/run_task.py \
        --checkpoint "${ckpt}" \
        --run_name "${name}" \
        --tasks "${TASKS}" \
        --output_dir "${OUT}" \
        --enforce_eager \
        ${EXTRA} > "logs/eval/${name}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE  ${name}"
    else
        echo "[$(date +%H:%M:%S)] FAIL  ${name} (rc=${rc}) — see logs/eval/${name}.log"
        FAILED=$((FAILED+1))
    fi
done
echo "Sweep complete: ${#JOBS[@]} jobs, ${FAILED} failed."
exit $([ $FAILED -eq 0 ] && echo 0 || echo 1)
