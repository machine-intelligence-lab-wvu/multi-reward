"""
Subprocess wrapper for train_from_scratch.py / train_intuitor.py.

Runs training N steps at a time in a child process. When the child exits,
ALL CUDA memory is returned to the OS — zero leakage between rounds.

Usage:
    python scripts/train_subprocess_wrapper.py \
        --script train_from_scratch.py \
        --steps_per_round 20 \
        --total_steps 500 \
        -- --model_name Qwen/Qwen2.5-1.5B --dataset math ...

    # Or wrap an existing shell script:
    python scripts/train_subprocess_wrapper.py \
        --shell_script scripts/train_multi-reward_scratch.sh \
        --steps_per_round 20 \
        --total_steps 500
"""

import argparse
import subprocess
import sys
import os
import glob
import time
import json


def find_latest_checkpoint(output_dir: str) -> tuple:
    """Find the latest checkpoint in output_dir/run_*/checkpoint-N."""
    best_step = -1
    best_path = None
    best_run = None

    for run_dir in glob.glob(os.path.join(output_dir, "run_*")):
        for ckpt_dir in glob.glob(os.path.join(run_dir, "checkpoint-*")):
            name = os.path.basename(ckpt_dir)
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    if step > best_step:
                        best_step = step
                        best_path = ckpt_dir
                        best_run = run_dir
                except ValueError:
                    continue

    # Also check latest_lora_adapter
    latest = os.path.join(output_dir, "latest_lora_adapter")
    if os.path.isdir(latest) and best_path is None:
        best_path = latest
        best_step = 0

    return best_step, best_path


def main():
    parser = argparse.ArgumentParser(
        description="Run training in subprocess rounds to prevent CUDA memory leaks"
    )
    parser.add_argument("--script", type=str, default=None,
                        help="Python training script (e.g. train_from_scratch.py)")
    parser.add_argument("--shell_script", type=str, default=None,
                        help="Shell training script (e.g. scripts/train_multi-reward_scratch.sh)")
    parser.add_argument("--steps_per_round", type=int, default=20,
                        help="Steps per subprocess round (default: 20)")
    parser.add_argument("--total_steps", type=int, default=500,
                        help="Total training steps")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir to find checkpoints (auto-detected from args if not set)")

    # Everything after -- is passed to the training script
    args, training_args = parser.parse_known_args()

    if args.script is None and args.shell_script is None:
        parser.error("Must specify --script or --shell_script")

    # Try to auto-detect output_dir from training args
    output_dir = args.output_dir
    if output_dir is None:
        for i, a in enumerate(training_args):
            if a == "--output_dir" and i + 1 < len(training_args):
                output_dir = training_args[i + 1]
                break
        # For shell scripts, we can't auto-detect — user must provide it
        if output_dir is None and args.shell_script:
            parser.error("Must specify --output_dir when using --shell_script")

    current_step = 0
    round_num = 0

    # Check if there's an existing checkpoint to resume from
    if output_dir and os.path.isdir(output_dir):
        existing_step, existing_path = find_latest_checkpoint(output_dir)
        if existing_step > 0:
            current_step = existing_step
            print(f"Found existing checkpoint at step {current_step}: {existing_path}")

    print(f"=== Subprocess Training Wrapper ===")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Steps per round: {args.steps_per_round}")
    print(f"  Rounds needed: {(args.total_steps - current_step + args.steps_per_round - 1) // args.steps_per_round}")
    print(f"  Starting from step: {current_step}")
    print(f"  Output dir: {output_dir}")
    print()

    while current_step < args.total_steps:
        round_num += 1
        round_steps = min(args.steps_per_round, args.total_steps - current_step)
        round_start = time.time()

        print(f"{'='*60}")
        print(f"  Round {round_num}: steps {current_step} → {current_step + round_steps}")
        print(f"{'='*60}")

        # Find latest checkpoint for resume
        resume_path = None
        if output_dir and os.path.isdir(output_dir):
            _, resume_path = find_latest_checkpoint(output_dir)

        # Build command
        if args.shell_script:
            cmd = ["bash", args.shell_script]
            # Append override args
            cmd.extend(["--max_steps", str(current_step + round_steps)])
            if resume_path:
                cmd.extend(["--resume_from", resume_path])
            # Pass any additional training args
            cmd.extend(training_args)
        else:
            cmd = [sys.executable, args.script]
            cmd.extend(training_args)
            cmd.extend(["--max_steps", str(current_step + round_steps)])
            if resume_path:
                cmd.extend(["--resume_from", resume_path])

        print(f"  Command: {' '.join(cmd[:6])} ...")
        print()

        # Run in subprocess — when it exits, ALL CUDA memory is freed
        result = subprocess.run(cmd, env=os.environ.copy())

        round_time = time.time() - round_start

        if result.returncode != 0:
            print(f"\n  Round {round_num} FAILED (exit code {result.returncode})")
            print(f"  Attempting to resume from last checkpoint...")
            # Check if any progress was made
            if output_dir:
                new_step, new_path = find_latest_checkpoint(output_dir)
                if new_step > current_step:
                    print(f"  Progress saved up to step {new_step}, continuing...")
                    current_step = new_step
                    continue
                else:
                    print(f"  No progress made. Retrying same round...")
                    # Retry once
                    result2 = subprocess.run(cmd, env=os.environ.copy())
                    if result2.returncode != 0:
                        print(f"  Retry also failed. Stopping.")
                        sys.exit(1)

        # Update current step
        if output_dir:
            new_step, _ = find_latest_checkpoint(output_dir)
            if new_step > current_step:
                current_step = new_step
            else:
                current_step += round_steps

        print(f"\n  Round {round_num} complete: step {current_step}/{args.total_steps} "
              f"({round_time:.0f}s, {round_time/round_steps:.0f}s/step)")
        print()

    print(f"\n{'='*60}")
    print(f"  Training complete! {args.total_steps} steps in {round_num} rounds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
