#!/bin/bash
# Stop A5_nonverif_ours at step 100 so both A-5 arms have MATCHED training.
#
# Deliberately NOT done via max_steps: 100 in the config. max_steps drives the
# cosine LR schedule: intuitor ran on a 200-step cosine and was cut at 100
# (LR there = 1.76e-6, mid-schedule). Setting max_steps: 100 for ours would
# complete the anneal (LR -> 0.000e+00 at step 100), giving a fully-annealed
# model versus a mid-schedule one -- a confound, and annealing usually flatters
# the endpoint. So ours keeps max_steps: 200 and is truncated the same way.
#
# The `ours` supervisor runs inside urla5handoff.service (handoff_a5.sh execs
# it), so that is the unit to stop.
set -uo pipefail
cd /home/WVU-AD/aa00307/projects/unsupervised-rl
CK="outputs/rebuttal/A5_nonverif_ours_qwen25_15b/checkpoint-100"
echo "[$(date)] waiting for ours checkpoint-100 ..."
until [ -d "${CK}/global_step100" ]; do sleep 120; done
sleep 60
echo "[$(date)] ours reached step 100; stopping supervisor unit urla5handoff.service"
systemctl --user stop urla5handoff.service 2>/dev/null
sleep 20
pkill -f "grpo_gt_train.py" 2>/dev/null; pkill -f "vllm-serve" 2>/dev/null
sleep 15
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && kill -9 "$p" 2>/dev/null
done
sleep 10
echo "[$(date)] A-5 COMPLETE: both arms truncated at step 100."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
