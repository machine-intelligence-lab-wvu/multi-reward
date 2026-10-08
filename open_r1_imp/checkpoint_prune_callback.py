"""
Checkpoint disk-usage callback — keeps only the newest checkpoint fully
resumable, strips the rest down to plain HuggingFace format.

Every `checkpoint-N/` directory written by `Trainer._save_checkpoint` already
contains TWO things layered on top of each other:
  1. A standalone HF-loadable model: config.json, tokenizer files,
     generation_config.json, and model.safetensors — written by
     `Trainer.save_model()`, consolidated to bf16 because the accelerate
     config sets `zero3_save_16bit_model: true`. This is small (~2x the
     model's raw bf16 size) and is everything `from_pretrained()` /
     eval / regenerating a paper figure ever needs.
  2. DeepSpeed's own engine checkpoint (fp32 master weights + Adam
     momentum/variance), written by `_save_optimizer_and_scheduler()` into a
     `global_step<N>/` subfolder plus a top-level `latest` marker. This is
     what's needed to resume training bit-for-bit, and is ~3-4x the model
     size on top of (1) — for a 1.5B model that's why a single checkpoint
     directory is ~20 GB instead of ~3 GB.

With `save_total_limit` capping the *count* of full checkpoints (not their
size), a 200+-step run with `save_steps: 20` keeps 10 of these repeat's-worth
of optimizer state around — 1-2 TB per run isn't unusual on shared scratch.

This callback keeps `keep_full` most-recent checkpoints fully resumable and,
for every older one, deletes just the DeepSpeed `global_step*/` engine dump
(and any FSDP/plain-Trainer optimizer.pt/scheduler.pt/rng_state*.pt/scaler.pt
sidecar files, if present) — leaving the HF-format weights in every
checkpoint directory forever, at a fraction of the size. If a run needs to
resume from anything other than the newest checkpoint, that older
checkpoint's optimizer state is gone by design; that trade-off is the whole
point of this callback (see ICLR-TODO.md's save-everything policy: the goal
is to never lose *evaluable* checkpoints, not to keep every one resumable).
"""

import glob
import logging
import os
import re
import shutil

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# Sidecar files a non-DeepSpeed (FSDP / plain) Trainer save can also write.
_OPTIMIZER_SIDECARS = ("optimizer.pt", "scheduler.pt", "scaler.pt")
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


class PruneOptimizerStateCallback(TrainerCallback):
    """After each save, strip resumability from all but the newest N checkpoints."""

    def __init__(self, keep_full: int = 1):
        self.keep_full = max(1, keep_full)

    def _prune_one(self, ckpt_dir: str) -> int:
        """Remove optimizer/scheduler state from one checkpoint dir. Returns bytes freed."""
        freed = 0
        for gs in glob.glob(os.path.join(ckpt_dir, "global_step*")):
            if os.path.isdir(gs):
                freed += _dir_size(gs)
                shutil.rmtree(gs, ignore_errors=True)
        latest = os.path.join(ckpt_dir, "latest")
        if os.path.isfile(latest):
            os.remove(latest)
        for name in _OPTIMIZER_SIDECARS:
            p = os.path.join(ckpt_dir, name)
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
        for p in glob.glob(os.path.join(ckpt_dir, "rng_state*.pt")):
            freed += os.path.getsize(p)
            os.remove(p)
        return freed

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        run_dir = args.output_dir
        try:
            steps = []
            for name in os.listdir(run_dir):
                m = _CKPT_RE.match(name)
                if m and os.path.isdir(os.path.join(run_dir, name)):
                    steps.append(int(m.group(1)))
        except FileNotFoundError:
            return
        steps.sort()
        to_prune = steps[: -self.keep_full] if len(steps) > self.keep_full else []
        total_freed = 0
        for step in to_prune:
            ckpt_dir = os.path.join(run_dir, f"checkpoint-{step}")
            try:
                total_freed += self._prune_one(ckpt_dir)
            except Exception as e:
                logger.warning(f"Could not prune optimizer state in {ckpt_dir}: {e}")
        if to_prune:
            logger.info(
                f"Pruned optimizer state from {len(to_prune)} older checkpoint(s) "
                f"(kept full: {steps[-self.keep_full:]}) — freed ~{total_freed / 1e9:.1f} GB"
            )


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total
