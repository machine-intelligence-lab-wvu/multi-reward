#!/bin/bash
# ============================================================
# Truncate A5_nonverif_intuitor at step 100 and hand the box to A5_nonverif_ours.
#
# Why truncate: INTUITOR length-hacks on Alpaca. Completions hit the 3072-token
# cap by step 40 (clipped_ratio 1.000) and self_certainty plateaus at ~41.2 from
# step 40 onward, so steps 100-200 add a flat line while costing 450 s/step
# (up from 303 s/step, because every completion now runs to the cap). Stopping
# at 100 buys ~7.5 h and still shows the collapse with a wide margin.
#
# The running supervisor reads MAX_STEPS once before its loop, so it has 200
# cached and would simply resume intuitor if we only killed training. We must
# stop the supervisor unit itself, then start a fresh supervisor for `ours`.
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
CKPT="${PROJECT_DIR}/outputs/rebuttal/A5_nonverif_intuitor_qwen25_15b/checkpoint-100"
LOG="logs/a5-handoff.out"

echo "[$(date)] waiting for intuitor checkpoint-100 ..."
until [ -d "${CKPT}" ]; do sleep 120; done
# The dir appears before the write finishes; wait for the DeepSpeed shard dir.
until [ -d "${CKPT}/global_step100" ]; do sleep 30; done
sleep 60
echo "[$(date)] checkpoint-100 present. Stopping the intuitor supervisor."

systemctl --user stop urla5sup.service 2>/dev/null
sleep 20
pkill -f "intuitor_train.py" 2>/dev/null
pkill -f "vllm-serve" 2>/dev/null
sleep 15
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && kill -9 "$p" 2>/dev/null
done
sleep 20
echo "[$(date)] GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

echo "[$(date)] starting supervisor for A5_nonverif_ours_qwen25_15b"
exec bash open_r1_imp/scripts/rebuttal/supervise_run.sh A5_nonverif_ours_qwen25_15b
