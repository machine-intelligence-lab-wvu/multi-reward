#!/bin/bash
# ============================================================
# Launch a rebuttal run on Bridges-2 by config id.
#   bash open_r1_imp/scripts/rebuttal/launch.sh A2_qwen3_17b_full
#   bash open_r1_imp/scripts/rebuttal/launch.sh A2_qwen3_17b_full --time=2:00:00   # extra sbatch args passthrough
#
# Picks the right entry script (multi/intuitor/grpo) + accelerate config + walltime
# from the config id, then submits scripts/rebuttal/submit_bridges2.sh.
# ============================================================
set -euo pipefail
ID="${1:?usage: launch.sh <config_id> [extra sbatch args]}"; shift || true
REPO="${HOME}/projects/unsupervised-rl"
CFG="open_r1_imp/recipes/rebuttal/${ID}.yaml"
SUBMIT="open_r1_imp/scripts/rebuttal/submit_bridges2.sh"
[ -f "${REPO}/${CFG}" ] || { echo "ERROR: no config ${CFG}"; exit 1; }

# entry script by method (match method name ANYWHERE in id — it is not always the suffix,
# e.g. A5_nonverif_intuitor_qwen25_15b). Order matters: check intuitor first.
case "$ID" in
  *intuitor*)                    TRAIN=open_r1_imp/intuitor_train.py ;;
  *full*|*multireward*|B7_*)     TRAIN=open_r1_imp/multi_reward_train.py ;;
  *)                             TRAIN=open_r1_imp/grpo_gt_train.py ;;   # grpo_gt, empo, corewarding, A4*, A5-ours
esac

# ── GPU shape ─────────────────────────────────────────────
# GPUS=2 (default): 2×H100 — GPU0 vLLM + 1 training GPU. Schedules FAR faster on
#   the jammed GPU-shared queue. grad_accum=42 keeps effective batch 882 (21×42×1).
# GPUS=4: 4×H100 — GPU0 vLLM + 3 training GPUs (original shape, grad_accum from config).
# GPU_TYPE=h100-80 (default) or l40s-48.
GPUS="${GPUS:-2}"
GPU_TYPE="${GPU_TYPE:-h100-80}"

# Co-rewarding REQUIRES a single training process (ori/aug pairing in the reward
# batch) — force the 2-GPU shape.
if [[ "$ID" == *corewarding* && "$GPUS" != 2 ]]; then
  echo "NOTE: ${ID} requires a single training process (view pairing) — forcing GPUS=2."
  GPUS=2
fi

# ── Batch geometry ────────────────────────────────────────
# The protocol fixes the EFFECTIVE batch at 882 sequences per optimizer step
# (per_device x grad_accum x training GPUs). Every shape must hit it exactly or
# the run is not comparable to the Qwen3/Qwen2.5 suites, so the guards below
# assert it rather than trusting the comments.
EFF_BATCH_TARGET=882
CFG_PDBS=$(awk '/^per_device_train_batch_size:/{print $2; exit}' "${REPO}/${CFG}")
CFG_GA=$(awk '/^gradient_accumulation_steps:/{print $2; exit}' "${REPO}/${CFG}")
CFG_G=$(awk '/^num_generations:/{print $2; exit}' "${REPO}/${CFG}")

if [ "$GPUS" = 2 ]; then
  NUM_TRAIN_GPUS=1
  GRES="gpu:${GPU_TYPE}:2"
  if [[ "$ID" == B6_* ]]; then
    # 3B on ONE training GPU: ZeRO-3 has nothing to shard (world_size=1), so the
    # AdamW state goes to CPU and the micro-batch shrinks — peak memory is driven
    # by the logits tensor (per_device x 3072 tokens x 128k vocab).
    PDBS=7;  GA=126
    ACCEL=open_r1_imp/accelerate_configs/zero3_grad126_cpu_offload.yaml
    TIME=48:00:00   # ~2-3x the 3-GPU wall-clock; resume-safe (ckpt every 20 steps)
  else
    PDBS=21; GA=42
    ACCEL=open_r1_imp/accelerate_configs/zero3_grad42.yaml
    TIME=36:00:00
  fi
  EXTRA_ARGS="${EXTRA_ARGS:-} --per_device_train_batch_size ${PDBS} --gradient_accumulation_steps ${GA}"
else
  NUM_TRAIN_GPUS=3
  GRES="gpu:${GPU_TYPE}:4"
  PDBS="${CFG_PDBS}"; GA="${CFG_GA}"        # 4-GPU shape runs the recipe as written
  case "$ID" in
    B6_*) ACCEL=open_r1_imp/accelerate_configs/zero3_grad21.yaml; TIME=14:00:00 ;;  # 3B
    *)    ACCEL=open_r1_imp/accelerate_configs/zero3.yaml;        TIME=10:00:00 ;;  # 1.5B / 1.7B
  esac
fi

# ── Guards: fail here, not 30 s into a scheduled job ──────
EFF=$(( PDBS * GA * NUM_TRAIN_GPUS ))
[ "$EFF" -eq "$EFF_BATCH_TARGET" ] || {
  echo "ERROR: effective batch ${PDBS}x${GA}x${NUM_TRAIN_GPUS}=${EFF}, expected ${EFF_BATCH_TARGET}." >&2; exit 1; }
[ -n "$CFG_G" ] && [ $(( PDBS % CFG_G )) -eq 0 ] || {
  echo "ERROR: per_device ${PDBS} is not a multiple of num_generations ${CFG_G}; TRL GRPO needs whole generation groups per micro-batch." >&2; exit 1; }
# anchor to the key (a leading '#' comment mentioning it must not match)
ACCEL_GA=$(awk '/^[[:space:]]*gradient_accumulation_steps:/{print $2; exit}' "${REPO}/${ACCEL}")
[ "$ACCEL_GA" = "$GA" ] || {
  echo "ERROR: ${ACCEL} sets gradient_accumulation_steps=${ACCEL_GA} but this shape needs ${GA} (DeepSpeed requires a match)." >&2; exit 1; }
export NUM_TRAIN_GPUS EXTRA_ARGS

OUT="${MR_SCRATCH}${ID}"
cd "$REPO"
echo "config=${CFG}  entry=${TRAIN}  accel=${ACCEL}  gpus=${GPUS}(${GRES})  time=${TIME}  out=${OUT}"
echo "batch: per_device=${PDBS} x grad_accum=${GA} x train_gpus=${NUM_TRAIN_GPUS} = ${EFF} (target ${EFF_BATCH_TARGET})"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[DRY_RUN] would submit: sbatch --time=${TIME} --gres=${GRES} --export=ALL,CONFIG_OVERRIDE=${CFG},TRAIN_SCRIPT=${TRAIN},ACCEL_CONFIG_OVERRIDE=${ACCEL},OUTPUT_DIR=${OUT} ${SUBMIT}"
  exit 0
fi
set -x
sbatch --time="${TIME}" --gres="${GRES}" "$@" \
  --export=ALL,CONFIG_OVERRIDE=${CFG},TRAIN_SCRIPT=${TRAIN},ACCEL_CONFIG_OVERRIDE=${ACCEL},OUTPUT_DIR=${OUT} \
  "${SUBMIT}"
