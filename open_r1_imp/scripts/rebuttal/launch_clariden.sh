#!/bin/bash
# ============================================================
# Launch a rebuttal run on Clariden by config id.
#   bash open_r1_imp/scripts/rebuttal/launch_clariden.sh B6_llama32_3b_intuitor
#   bash open_r1_imp/scripts/rebuttal/launch_clariden.sh T2_triviaqa_full_qwen25_15b --time=6:00:00
#
# Clariden equivalent of launch.sh (which targets Bridges-2's submit_bridges2.sh
# and its scarce GPU-shared queue). Here we always get a full node — 4 GPUs, so
# every recipe runs its "GPUS=4" shape as written (GPU0 vLLM + 3 training GPUs),
# no batch-geometry override needed.
# ============================================================
set -euo pipefail
ID="${1:?usage: launch_clariden.sh <config_id> [extra sbatch args]}"; shift || true
REPO="${HOME}/projects/unsupervised-rl"
CFG="open_r1_imp/recipes/rebuttal/${ID}.yaml"
SUBMIT="open_r1_imp/scripts/rebuttal/submit_clariden.sh"
[ -f "${REPO}/${CFG}" ] || { echo "ERROR: no config ${CFG}"; exit 1; }

# Entry script by method (match ANYWHERE in id — not always the suffix, e.g.
# A5_nonverif_intuitor_qwen25_15b). Order matters: check intuitor first.
case "$ID" in
  *intuitor*)                    TRAIN=open_r1_imp/intuitor_train.py ;;
  *full*|*multireward*|B7_*)     TRAIN=open_r1_imp/multi_reward_train.py ;;
  *)                             TRAIN=open_r1_imp/grpo_gt_train.py ;;   # grpo_gt, empo, corewarding, A4*, A5-ours
esac

# Accelerate config + walltime by model size (3B needs more grad-accum steps
# per micro-batch and more wall-clock; everything else here is 1.5B/1.7B).
case "$ID" in
  B6_*) ACCEL=open_r1_imp/accelerate_configs/zero3_grad21.yaml; TIME=12:00:00 ;;  # normal partition caps at 12h; auto-resumes if it times out
  *)    ACCEL=open_r1_imp/accelerate_configs/zero3.yaml;        TIME=10:00:00 ;;
esac

# ── Guard: fail here, not 30s into a scheduled job. Verifies the recipe's own
# batch geometry hits the protocol's fixed effective batch (882 = per_device x
# grad_accum x 3 training GPUs), and that the accelerate config's grad_accum
# matches what the recipe expects (DeepSpeed requires them to agree). ──
EFF_BATCH_TARGET=882
NUM_TRAIN_GPUS=3
CFG_PDBS=$(awk '/^per_device_train_batch_size:/{print $2; exit}' "${REPO}/${CFG}")
CFG_GA=$(awk '/^gradient_accumulation_steps:/{print $2; exit}' "${REPO}/${CFG}")
CFG_G=$(awk '/^num_generations:/{print $2; exit}' "${REPO}/${CFG}")
EFF=$(( CFG_PDBS * CFG_GA * NUM_TRAIN_GPUS ))
[ "$EFF" -eq "$EFF_BATCH_TARGET" ] || {
  echo "ERROR: effective batch ${CFG_PDBS}x${CFG_GA}x${NUM_TRAIN_GPUS}=${EFF}, expected ${EFF_BATCH_TARGET}." >&2; exit 1; }
[ -n "$CFG_G" ] && [ $(( CFG_PDBS % CFG_G )) -eq 0 ] || {
  echo "ERROR: per_device ${CFG_PDBS} is not a multiple of num_generations ${CFG_G}." >&2; exit 1; }
ACCEL_GA=$(awk '/^[[:space:]]*gradient_accumulation_steps:/{print $2; exit}' "${REPO}/${ACCEL}")
[ "$ACCEL_GA" = "$CFG_GA" ] || {
  echo "ERROR: ${ACCEL} sets gradient_accumulation_steps=${ACCEL_GA} but recipe needs ${CFG_GA}." >&2; exit 1; }

OUT=$(awk '/^output_dir:/{print $2; exit}' "${REPO}/${CFG}")
cd "$REPO"
echo "config=${CFG}  entry=${TRAIN}  accel=${ACCEL}  time=${TIME}  out=${OUT}"
echo "batch: per_device=${CFG_PDBS} x grad_accum=${CFG_GA} x train_gpus=${NUM_TRAIN_GPUS} = ${EFF} (target ${EFF_BATCH_TARGET})"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[DRY_RUN] would submit: sbatch --job-name=${ID} --time=${TIME} --export=ALL,CONFIG_OVERRIDE=${CFG},TRAIN_SCRIPT=${TRAIN},ACCEL_CONFIG_OVERRIDE=${ACCEL} ${SUBMIT}"
  exit 0
fi
set -x
sbatch --job-name="${ID}" --time="${TIME}" "$@" \
  --export=ALL,CONFIG_OVERRIDE=${CFG},TRAIN_SCRIPT=${TRAIN},ACCEL_CONFIG_OVERRIDE=${ACCEL} \
  "${SUBMIT}"
