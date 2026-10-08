#!/bin/bash
set -uo pipefail
cd /home/WVU-AD/aa00307/projects/unsupervised-rl
echo "[$(date)] waiting for A5_nonverif_intuitor to finish before starting ours..."
while pgrep -f "run_local48g.sh A5_nonverif_intuitor" > /dev/null 2>&1; do sleep 120; done
echo "[$(date)] GPUs free; starting A5_nonverif_ours"
exec bash open_r1_imp/scripts/rebuttal/run_queue_local48g.sh A5_nonverif_ours_qwen25_15b
