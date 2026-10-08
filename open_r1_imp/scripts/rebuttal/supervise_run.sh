#!/bin/bash
# ============================================================
# Run a rebuttal config to completion, surviving interruptions.
#
#   bash open_r1_imp/scripts/rebuttal/supervise_run.sh <CONFIG_ID> [CONFIG_ID ...]
#
# Training on this box is repeatedly killed by an external SIGTERM (signal 15):
# no OOM, no traceback, no reboot (box has 4d uptime), and systemd reports the
# unit exiting cleanly -- so something outside the unit signals the training
# process. A-2 (40h) and A-4 (19h) survived; the eval runs and A-5 did not. The
# source is unidentified, so rather than trying to prevent the signal this
# supervisor makes the run tolerant of it: after any non-zero exit it finds the
# newest checkpoint and resumes from there, repeating until global_step reaches
# max_steps.
#
# Costs at most save_steps (20) steps of redone work per interruption.
#
# Guard: if MAX_STALLS consecutive attempts fail to advance global_step, abort.
# Without that, a run being killed faster than it can reach its first checkpoint
# would relaunch from scratch forever.
# ============================================================
set -uo pipefail

PROJECT_DIR="/home/WVU-AD/aa00307/projects/unsupervised-rl"
cd "${PROJECT_DIR}"
mkdir -p logs

MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
MAX_STALLS="${MAX_STALLS:-3}"
SUMMARY="logs/supervisor-summary.txt"

[ $# -ge 1 ] || { echo "usage: supervise_run.sh <CONFIG_ID> [CONFIG_ID ...]"; exit 1; }

# Echo the current global_step for a run (0 if it has never checkpointed).
current_step() {
    local outdir="$1"
    local latest
    latest=$(ls "${outdir}" 2>/dev/null | grep -o 'checkpoint-[0-9]*' \
             | sed 's/checkpoint-//' | sort -n | tail -1)
    [ -z "${latest}" ] && { echo 0; return; }
    echo "${latest}"
}

for ID in "$@"; do
    CFG="open_r1_imp/recipes/rebuttal_local48g/${ID}.yaml"
    if [ ! -f "${CFG}" ]; then
        echo "[$(date)] ERROR no config ${CFG}" | tee -a "${SUMMARY}"; continue
    fi
    MAX_STEPS=$(grep -E "^max_steps:" "${CFG}" | head -1 | awk '{print $2}')
    OUTDIR=$(grep -E "^output_dir:" "${CFG}" | head -1 | awk '{print $2}')

    echo "============================================================" | tee -a "${SUMMARY}"
    echo "[$(date)] SUPERVISE ${ID} -> target ${MAX_STEPS} steps" | tee -a "${SUMMARY}"

    attempt=0; stalls=0
    while : ; do
        step=$(current_step "${OUTDIR}")
        if [ "${step}" -ge "${MAX_STEPS}" ]; then
            echo "[$(date)] COMPLETE ${ID} at step ${step}" | tee -a "${SUMMARY}"; break
        fi
        attempt=$((attempt+1))
        if [ "${attempt}" -gt "${MAX_ATTEMPTS}" ]; then
            echo "[$(date)] GIVE UP ${ID}: ${MAX_ATTEMPTS} attempts, stuck at ${step}" | tee -a "${SUMMARY}"; break
        fi

        RESUME_ARG=""
        if [ "${step}" -gt 0 ]; then
            RESUME_ARG="${OUTDIR}/checkpoint-${step}"
            echo "[$(date)] attempt ${attempt}: resuming ${ID} from step ${step}" | tee -a "${SUMMARY}"
        else
            echo "[$(date)] attempt ${attempt}: starting ${ID} from scratch" | tee -a "${SUMMARY}"
        fi

        RESUME="${RESUME_ARG}" bash open_r1_imp/scripts/rebuttal/run_local48g.sh "${ID}" \
            >> "logs/${ID}.out" 2>&1
        rc=$?

        new_step=$(current_step "${OUTDIR}")
        echo "[$(date)]   exit=${rc}  step ${step} -> ${new_step}" | tee -a "${SUMMARY}"

        # Clear any GPU stragglers before the next attempt (vllm-serve's
        # EngineCore child does not reliably die with its parent).
        for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
            [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && kill -9 "$p" 2>/dev/null
        done
        sleep 20

        if [ "${new_step}" -le "${step}" ]; then
            stalls=$((stalls+1))
            echo "[$(date)]   no progress (stall ${stalls}/${MAX_STALLS})" | tee -a "${SUMMARY}"
            if [ "${stalls}" -ge "${MAX_STALLS}" ]; then
                echo "[$(date)] GIVE UP ${ID}: ${MAX_STALLS} consecutive attempts made no progress." \
                    | tee -a "${SUMMARY}"
                tail -20 "logs/${ID}.out" | sed 's/^/    /' | tee -a "${SUMMARY}"
                break
            fi
        else
            stalls=0
        fi
    done
done

echo "[$(date)] supervisor finished." | tee -a "${SUMMARY}"
