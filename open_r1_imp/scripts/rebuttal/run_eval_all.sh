#!/bin/bash
# ============================================================
# Remaining A-2 evaluation, at HALF LOAD (2 GPUs, not 4).
#
#   systemd-run --user --scope --collect --unit=urleval \
#       bash open_r1_imp/scripts/rebuttal/run_eval_all.sh
#
# Why 2 GPUs: the box has taken unclean reboots (Jul 27 13:35 and 19:50) and two
# eval attempts died mid-mmlu_pro. Cause unconfirmed, so we deliberately avoid
# driving all four cards at ~90% memory and full power simultaneously. Cost of
# halving: mmlu_pro subset goes ~20min -> ~40min. Cheap insurance.
#
# Why systemd scope: both earlier attempts were killed a few minutes in with no
# error and no OOM. A transient scope puts the job in its own cgroup so it is
# not reaped with the parent shell.
#
# PHASE 1  mmlu_pro on a SEEDED RANDOM 2000-example subset (seed 7).
#          NOT the first 2000 — those are 55% law / 39.5% business and cover
#          3 of MMLU-Pro's 14 categories. Seeded subset matches the full
#          category distribution to ~1pt; SE at n=2000 is ~+-1.1%.
# PHASE 2  trajectory sweep: every checkpoint (20..200) x 3 runs on
#          gsm8k + math500. This is what RUN_PLAN.md actually asks for on A-2,
#          and it yields best-checkpoint selection + accuracy-vs-step.
#
# Both phases are idempotent; re-running only redoes what is missing.
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
mkdir -p logs

NUM_GPUS="${NUM_GPUS:-2}"

# ─── Telemetry, so a crash leaves evidence instead of guesswork ──
(
  echo "timestamp,gpu,power_w,temp_c,mem_used_mib,util_pct"
  while true; do
    nvidia-smi --query-gpu=index,power.draw,temperature.gpu,memory.used,utilization.gpu \
               --format=csv,noheader,nounits \
      | sed "s/^/$(date +%H:%M:%S),/"
    sleep 10
  done
) > logs/eval-telemetry.csv 2>&1 &
TELEMETRY_PID=$!
trap 'kill ${TELEMETRY_PID} 2>/dev/null || true' EXIT INT TERM

echo "============================================================"
echo "[$(date)] PHASE 1/2: mmlu_pro, seeded random 2000-subset, ${NUM_GPUS} GPUs"
echo "============================================================"
NUM_GPUS="${NUM_GPUS}" PHASE=headline TASKS=mmlu_pro \
MAX_SAMPLES=2000 SAMPLE_SEED=7 \
    bash open_r1_imp/scripts/rebuttal/run_eval_local48g.sh
echo "[$(date)] phase 1 exit=$?"

echo "============================================================"
echo "[$(date)] PHASE 2/2: trajectory sweep, gsm8k+math500, ${NUM_GPUS} GPUs"
echo "============================================================"
NUM_GPUS="${NUM_GPUS}" PHASE=trajectory TASKS=gsm8k,math500 \
    bash open_r1_imp/scripts/rebuttal/run_eval_local48g.sh
echo "[$(date)] phase 2 exit=$?"

echo "============================================================"
echo "[$(date)] ALL EVAL COMPLETE"
find eval-rebuttal-results -name "*.json" | wc -l | xargs echo "  total result JSONs:"
echo "  peak power/temp seen:"
awk -F, 'NR>1 && $3+0>mp{mp=$3+0} NR>1 && $4+0>mt{mt=$4+0} END{printf "    max %.0f W/GPU, max %.0f C\n", mp, mt}' logs/eval-telemetry.csv
echo "============================================================"
