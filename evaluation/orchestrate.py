"""Orchestrator: distributes (run_name, checkpoint) jobs across GPUs.

Each GPU gets assigned a subset of runs and iterates over them sequentially,
loading vLLM once per run and evaluating all tasks.

Usage:
  python evaluation/orchestrate.py \
      --num_gpus 4 \
      --output_dir evaluation_results/best_ckpts_v1 \
      --tasks gsm8k,math500,math,aime2024,aime2025,mmlu_pro \
      --enforce_eager

Idempotent: skips (run, task) pairs whose output JSON already exists.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Best-checkpoint list — each method's peak based on the reference table.
# Multi-reward base: ckpt-160 doesn't exist in run1 (only 150,200,250,300,350)
#   → using ckpt-150 as closest proxy to the reference-table peak.
JOBS = [
    # (run_name, checkpoint_path_or_hf_id)
    ("qwen25_1.5b_base",                 "Qwen/Qwen2.5-1.5B"),
    ("multi_reward_base_ckpt150",        "${MR_OUTPUT_DIR}/run1_full_qwen2.5-1.5b/checkpoint-150"),
    ("multi_reward_klcov05_ckpt160",     "${MR_OUTPUT_DIR}/run2_full_qwen2.5-1.5b_klcov-0.05/checkpoint-160"),
    ("multi_reward_klcov02_ckpt160",     "${MR_OUTPUT_DIR}/run3_full_qwen2.5-1.5b_klcov-0.02/checkpoint-160"),
    ("intuitor_imp_ckpt80",              "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_15b/checkpoint-80"),
    ("intuitor_klcov_ckpt80",            "${MR_OUTPUT_DIR}/run3_intuitor_klcov_qwen25_15b/checkpoint-80"),
    ("intuitor_klcov_ckpt160",           "${MR_OUTPUT_DIR}/run3_intuitor_klcov_qwen25_15b/checkpoint-160"),
    ("grpo_gt_ckpt300",                  "${MR_OUTPUT_DIR}/grpo_gt_qwen25_15b/checkpoint-300"),
    # Qwen2.5-3B runs
    ("multi_reward_3b_noklcov_ckpt160",  "${MR_OUTPUT_DIR}/multi_reward_noklcov_qwen25_3b/checkpoint-160"),
]


# ─────────────────────────────────────────────────────────────
# Trajectory mode: for each METHOD, auto-expand its run_dir into
# one eval job per checkpoint-N (N ≤ --max_step). Used for the
# "accuracy over training steps" plot.
# ─────────────────────────────────────────────────────────────
TRAJECTORY_METHODS = [
    # (method_short_name, run_dir)
    ("grpo_gt",             "${MR_OUTPUT_DIR}/grpo_gt_qwen25_15b"),
    ("intuitor",            "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_15b"),
    ("intuitor_klcov",      "${MR_OUTPUT_DIR}/run3_intuitor_klcov_qwen25_15b"),
    # Pre-fix multi-reward runs (effectively SC-only — kept for reproducibility of
    # earlier trajectory plots; do not use for post-fix comparisons)
    ("multi_reward",        "${MR_OUTPUT_DIR}/multi_reward_noklcov_qwen25_1.5b"),
    ("multi_reward_klcov05","${MR_OUTPUT_DIR}/run2_full_qwen2.5-1.5b_klcov-0.05"),
    ("multi_reward_klcov02","${MR_OUTPUT_DIR}/run3_full_qwen2.5-1.5b_klcov-0.02"),
    # Post-fix multi-reward (canonical — matches commit bfb30a8 fixes)
    ("multi_reward_final",         "${MR_OUTPUT_DIR}/multi_reward_final_noklcov_qwen25_1.5b"),
    ("multi_reward_final_klcov05", "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_1.5b"),
    ("multi_reward_final_klcov0.1", "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.1_qwen25_1.5b"),
    ("multi_reward_final_klcov0005", "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.0005_qwen25_1.5b"),
    ("cluster_only",                 "${MR_OUTPUT_DIR}/cluster_only_noklcov_qwen25_1.5b"),

    # ── Qwen2.5-3B trajectory runs (post-fix for multi-reward variants) ──
    ("grpo_gt_3b",                    "${MR_OUTPUT_DIR}/grpo_gt_qwen25_3b"),
    ("intuitor_3b",                   "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_3b"),
    ("multi_reward_final_3b",         "${MR_OUTPUT_DIR}/multi_reward_final_noklcov_qwen25_3b"),
    ("multi_reward_final_klcov05_3b", "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_3b"),

    # ── Qwen2.5-7B trajectory runs (rebuttal) ──
    ("multi_reward_final_klcov05_7b", "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_7b"),
    ("intuitor_7b",                   "${MR_OUTPUT_DIR}/intuitor_qwen25_7b"),
    ("grpo_gt_7b",                    "${MR_OUTPUT_DIR}/grpo_gt_qwen25_7b"),
]


def _expand_trajectory(methods, max_step: int, step_stride: int = 20):
    """Turn (method, run_dir) pairs into one job per checkpoint-N, N ≤ max_step.

    Only keeps checkpoints whose step is a multiple of `step_stride` (or equal
    to `max_step`, so the end-of-horizon ckpt is always included even if it
    doesn't divide the stride). Default stride 20 matches the training save
    cadence (every checkpoint).

    Checkpoints are sorted NUMERICALLY by step (not lexicographically — otherwise
    "checkpoint-100" sorts before "checkpoint-20"). This makes sure GPUs process
    early steps first, so wandb gets ascending step= log entries.

    run_name becomes "{method}_ckpt{step:03d}" so grouping/wandb can parse it.
    """
    import re
    jobs = []
    for method, run_dir in methods:
        p = Path(run_dir)
        if not p.is_dir():
            print(f"[trajectory] skip {method}: {run_dir} not a directory")
            continue
        ckpts = []
        for ckpt_dir in p.glob("checkpoint-*"):
            m = re.match(r"checkpoint-(\d+)$", ckpt_dir.name)
            if not m:
                continue
            step = int(m.group(1))
            if step > max_step:
                continue
            if step_stride > 1 and step % step_stride != 0 and step != max_step:
                continue
            ckpts.append((step, ckpt_dir))
        ckpts.sort(key=lambda x: x[0])
        for step, ckpt_dir in ckpts:
            run_name = f"{method}_ckpt{step:03d}"
            jobs.append((run_name, str(ckpt_dir)))
    return jobs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--output_dir", default="evaluation_results/best_ckpts_v1")
    p.add_argument("--tasks", default="gsm8k,math500,aime2024,aime2025,mmlu_pro")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", default=None,
                   help="Comma-separated run_names to include (default: all)")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb logging (one run per checkpoint, grouped)")
    p.add_argument("--wandb_project", default="unsupervised-rl-eval")
    p.add_argument("--wandb_group", default="best_ckpts_v1")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per prompt (n>1 enables pass@k / maj@k)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--trajectory", action="store_true",
                   help="Expand TRAJECTORY_METHODS into per-checkpoint jobs "
                        "(for the training-dynamics sweep).")
    p.add_argument("--max_step", type=int, default=340,
                   help="In trajectory mode, skip checkpoints with step > this.")
    p.add_argument("--step_stride", type=int, default=20,
                   help="In trajectory mode, only keep ckpts whose step is a "
                        "multiple of this (max_step is always kept). "
                        "Default 20 = every saved ckpt.")
    return p.parse_args()


def _assign_jobs_to_gpus(jobs, num_gpus):
    """Round-robin assignment (longer jobs first is not easy to know a priori)."""
    assignments = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        assignments[i % num_gpus].append(job)
    return assignments


def _launch_worker(gpu_id: int, jobs_for_gpu, args):
    """Launch a subprocess that iterates over this GPU's jobs."""
    # Build a small bash command that runs run_task.py for each job
    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path(args.output_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmds = []
    for run_name, ckpt in jobs_for_gpu:
        log_file = log_dir / f"gpu{gpu_id}_{run_name}.log"
        cmd = [
            sys.executable, "-u", str(project_root / "evaluation" / "run_task.py"),
            "--checkpoint", ckpt,
            "--run_name", run_name,
            "--tasks", args.tasks,
            "--output_dir", args.output_dir,
        ]
        if args.max_samples is not None:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.enforce_eager:
            cmd += ["--enforce_eager"]
        if args.overwrite:
            cmd += ["--overwrite"]
        if args.n_samples > 1:
            cmd += ["--n_samples", str(args.n_samples)]
        if args.temperature != 0.0:
            cmd += ["--temperature", str(args.temperature)]
        if args.top_p != 1.0:
            cmd += ["--top_p", str(args.top_p)]
        if args.wandb:
            cmd += ["--wandb",
                    "--wandb_project", args.wandb_project,
                    "--wandb_group", args.wandb_group]
        if args.trajectory:
            cmd += ["--trajectory"]
        cmds.append((cmd, log_file, run_name))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Chain commands so the worker runs them in order
    procs = []
    for cmd, log_file, run_name in cmds:
        print(f"[GPU{gpu_id}] launching {run_name} (log: {log_file})")
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        p.wait()  # sequential within one GPU
        rc = p.returncode
        print(f"[GPU{gpu_id}] {run_name} exit={rc}")
        procs.append((run_name, rc))
    return procs


def main():
    args = parse_args()

    if args.trajectory:
        jobs = _expand_trajectory(TRAJECTORY_METHODS, args.max_step, args.step_stride)
        if args.only:
            wanted = set(args.only.split(","))
            # In trajectory mode, --only filters by method prefix (e.g. "intuitor")
            jobs = [j for j in jobs if any(j[0].startswith(w + "_ckpt") for w in wanted)]
        print(f"[trajectory] {len(jobs)} jobs across "
              f"{len(TRAJECTORY_METHODS)} methods, max_step={args.max_step}, "
              f"step_stride={args.step_stride}")
        # Count per method
        from collections import Counter
        by_method = Counter(j[0].rsplit("_ckpt", 1)[0] for j in jobs)
        for method, n in by_method.items():
            print(f"  {method}: {n} checkpoints")
    else:
        jobs = JOBS
        if args.only:
            wanted = set(args.only.split(","))
            jobs = [j for j in jobs if j[0] in wanted]
        print(f"Jobs: {[j[0] for j in jobs]}")
    if not jobs:
        print("No jobs to run"); return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    assignments = _assign_jobs_to_gpus(jobs, args.num_gpus)
    for i, ass in enumerate(assignments):
        names = [j[0] for j in ass]
        print(f"  GPU{i}: {names}")

    # Launch one worker process per GPU, each runs its jobs sequentially,
    # and all GPUs run in parallel. We use multiprocessing for this.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, jobs_for_gpu in enumerate(assignments):
        if not jobs_for_gpu:
            continue
        p = ctx.Process(target=_launch_worker, args=(gpu_id, jobs_for_gpu, args))
        p.start()
        procs.append(p)

    t0 = time.time()
    for p in procs:
        p.join()
    print(f"All GPUs done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
