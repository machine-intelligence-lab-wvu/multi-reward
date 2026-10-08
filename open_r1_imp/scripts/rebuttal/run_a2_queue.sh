#!/bin/bash
# ============================================================
# Run the A-2 rebuttal suite (Qwen3-1.7B-Base) sequentially on the local box.
# One run at a time — each needs all 4 GPUs (1 vLLM + 3 ZeRO-3).
#
#   nohup bash open_r1_imp/scripts/rebuttal/run_a2_queue.sh > logs/a2-queue.out 2>&1 &
#
# A failing run does NOT stop the queue; it is recorded and the next one starts,
# so an overnight failure costs one run rather than all three.
# Per-run logs: logs/<CONFIG_ID>.out   Summary: logs/a2-queue-summary.txt
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
mkdir -p logs outputs/rebuttal

RUNS=(
  A2_qwen3_17b_full
  A2_qwen3_17b_intuitor
  A2_qwen3_17b_grpo_gt
)

SUMMARY="logs/a2-queue-summary.txt"
echo "A-2 queue started $(date)" | tee -a "${SUMMARY}"

for ID in "${RUNS[@]}"; do
    echo "============================================================" | tee -a "${SUMMARY}"
    echo "[$(date)] START ${ID}" | tee -a "${SUMMARY}"

    START=$(date +%s)
    bash open_r1_imp/scripts/rebuttal/run_local48g.sh "${ID}" > "logs/${ID}.out" 2>&1
    RC=$?
    END=$(date +%s)
    ELAPSED=$(( (END - START) / 60 ))

    if [ $RC -eq 0 ]; then
        echo "[$(date)] DONE  ${ID} (exit 0, ${ELAPSED} min)" | tee -a "${SUMMARY}"
    else
        echo "[$(date)] FAIL  ${ID} (exit ${RC}, ${ELAPSED} min) — see logs/${ID}.out" | tee -a "${SUMMARY}"
        tail -25 "logs/${ID}.out" | sed 's/^/    /' | tee -a "${SUMMARY}"
    fi

    # Make sure nothing is still holding a GPU before the next run starts.
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
            echo "    clearing leftover GPU pid $p" | tee -a "${SUMMARY}"
            kill -9 "$p" 2>/dev/null || true
        fi
    done
    sleep 20
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' | tee -a "${SUMMARY}"
    echo "" | tee -a "${SUMMARY}"
done

echo "[$(date)] A-2 queue complete." | tee -a "${SUMMARY}"
