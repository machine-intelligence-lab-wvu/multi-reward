#!/bin/bash
# ============================================================
# Run a list of rebuttal config ids sequentially on the local box.
# Each run needs all 4 GPUs (1 vLLM + 3 ZeRO-3), so no parallelism.
#
#   systemd-run --user --scope --collect --unit=urlq \
#     bash open_r1_imp/scripts/rebuttal/run_queue_local48g.sh \
#       A4_grpo_gt_klcov_qwen25_15b A4_grpo_gt_noklcov_qwen25_15b
#
# Generic replacement for run_a2_queue.sh. A failing run does NOT stop the
# queue: it is recorded and the next starts, so an overnight failure costs one
# run rather than all of them. GPUs are force-cleared between runs because
# vllm-serve's EngineCore child does not always die with its parent.
#
# Logs: logs/<CONFIG_ID>.out   Summary: logs/queue-summary.txt
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
mkdir -p logs outputs/rebuttal

[ $# -ge 1 ] || {
    echo "usage: run_queue_local48g.sh <CONFIG_ID[@resume_checkpoint]> [CONFIG_ID[@ckpt] ...]"
    echo "  e.g. run_queue_local48g.sh A4_x@outputs/rebuttal/A4_x/checkpoint-60 A4_y"
    exit 1
}
RUNS=("$@")

SUMMARY="logs/queue-summary.txt"
echo "queue started $(date): ${RUNS[*]}" | tee -a "${SUMMARY}"

for SPEC in "${RUNS[@]}"; do
    # "ID" or "ID@/path/to/checkpoint-N" to resume that run mid-way.
    ID="${SPEC%%@*}"
    CKPT=""
    [ "${SPEC}" != "${ID}" ] && CKPT="${SPEC#*@}"

    echo "============================================================" | tee -a "${SUMMARY}"
    if [ -n "${CKPT}" ]; then
        echo "[$(date)] START ${ID} (resume from ${CKPT})" | tee -a "${SUMMARY}"
    else
        echo "[$(date)] START ${ID}" | tee -a "${SUMMARY}"
    fi
    START=$(date +%s)
    RESUME="${CKPT}" bash open_r1_imp/scripts/rebuttal/run_local48g.sh "${ID}" \
        >> "logs/${ID}.out" 2>&1
    RC=$?
    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    if [ $RC -eq 0 ]; then
        echo "[$(date)] DONE  ${ID} (exit 0, ${ELAPSED} min)" | tee -a "${SUMMARY}"
    else
        echo "[$(date)] FAIL  ${ID} (exit ${RC}, ${ELAPSED} min) — logs/${ID}.out" | tee -a "${SUMMARY}"
        tail -25 "logs/${ID}.out" | sed 's/^/    /' | tee -a "${SUMMARY}"
    fi
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

echo "[$(date)] queue complete." | tee -a "${SUMMARY}"
