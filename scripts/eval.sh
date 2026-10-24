#!/usr/bin/env bash
# Evaluate every checkpoint of a training run with evaluation/run_task.py (vLLM, one GPU).
#
#   bash scripts/eval.sh outputs/qwen2.5-1.5b_multi_reward_klcov
#   TASKS=aime2024,aime2025,lcb,cruxeval_o STEPS=160 bash scripts/eval.sh outputs/qwen2.5-1.5b_multi_reward_klcov
#   bash scripts/eval.sh Qwen/Qwen3-1.7B-Base results/qwen3-1.7b      # base model -> run name "base"
#
# Results go to {RESULTS_DIR}/{method}_ckpt{step}/{method}_ckpt{step}__{task}.json, where
# a run directory "<model>_<method>" gives RESULTS_DIR=results/<model> by default. This is
# the layout read by evaluation/paper_figures/* and evaluation/significance.py.
#
# Environment variables:
#   TASKS        comma-separated tasks (default gsm8k,math500,mmlu_pro)
#   STEPS        comma-separated checkpoint steps (default: all checkpoint-* dirs)
#   EVAL_ARGS    extra run_task.py arguments, e.g. "--n_samples 8 --temperature 1.0 --seed 0"
#   METHOD       override the method name used in the run names
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN="${1:?usage: bash scripts/eval.sh RUN_DIR|HF_MODEL [RESULTS_DIR]}"
TASKS="${TASKS:-gsm8k,math500,mmlu_pro}"
EVAL_ARGS="${EVAL_ARGS:-}"
PY="${PYTHON:-python}"

if [ ! -d "$RUN" ]; then
    # Not a local run directory: evaluate a (base) model by name.
    RESULTS_DIR="${2:?give RESULTS_DIR when evaluating a model by name}"
    # shellcheck disable=SC2086
    "$PY" evaluation/run_task.py --checkpoint "$RUN" --run_name "${METHOD:-base}" \
        --tasks "$TASKS" --output_dir "$RESULTS_DIR" $EVAL_ARGS
    exit 0
fi

NAME="$(basename "$RUN")"
MODEL="${NAME%%_*}"
METHOD="${METHOD:-${NAME#*_}}"
RESULTS_DIR="${2:-results/$MODEL}"

if [ -n "${STEPS:-}" ]; then
    CKPTS=(); IFS=',' read -ra S <<< "$STEPS"
    for s in "${S[@]}"; do CKPTS+=("$RUN/checkpoint-$s"); done
else
    CKPTS=(); while IFS= read -r d; do CKPTS+=("$d"); done < <(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
fi
[ "${#CKPTS[@]}" -gt 0 ] || { echo "No checkpoints found in $RUN"; exit 1; }

for ckpt in "${CKPTS[@]}"; do
    [ -d "$ckpt" ] || { echo "Missing $ckpt"; exit 1; }
    step="${ckpt##*-}"
    echo "== $METHOD step $step -> $RESULTS_DIR"
    # shellcheck disable=SC2086
    "$PY" evaluation/run_task.py --checkpoint "$ckpt" --run_name "${METHOD}_ckpt${step}" \
        --tasks "$TASKS" --output_dir "$RESULTS_DIR" $EVAL_ARGS
done
