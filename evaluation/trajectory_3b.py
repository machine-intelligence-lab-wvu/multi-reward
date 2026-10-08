"""Trajectory evaluation for Qwen2.5-3B runs.

Iterates every-20-step checkpoints of 4 trained methods and evaluates each
on 5 benchmarks. Distributes (method, checkpoint) pairs across 4 GPUs.
Per-GPU worker loads vLLM once per checkpoint and evaluates all 5 tasks.

Output layout (one top-level dir per task, as requested):
  evaluation_results/accuracy_trajectory_gsm8k_qwen3b/{method}_ckpt{step:03d}/{run}__gsm8k.json
  evaluation_results/accuracy_trajectory_math500_qwen3b/{method}_ckpt{step:03d}/{run}__math500.json
  evaluation_results/accuracy_trajectory_mmlu_pro_qwen3b/...
  evaluation_results/accuracy_trajectory_aime24_qwen3b/...
  evaluation_results/accuracy_trajectory_aime25_qwen3b/...

Idempotent: skips (run, task) pairs whose JSON already exists.

Usage:
  python evaluation/trajectory_3b.py --num_gpus 4 --enforce_eager
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# Method short-name → absolute output_dir from training
METHODS = [
    ("grpo_gt",              "${MR_OUTPUT_DIR}/grpo_gt_qwen25_3b"),
    ("multi_reward",         "${MR_OUTPUT_DIR}/multi_reward_noklcov_qwen25_3b"),
    ("multi_reward_klcov05", "${MR_OUTPUT_DIR}/multi_reward_klcov_0.05_qwen25_3b"),
    ("intuitor",             "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_3b"),
]

# Tasks to evaluate per checkpoint
TASKS = ["gsm8k", "math500", "mmlu_pro", "aime2024", "aime2025"]

# Map task-name → subdirectory under evaluation_results/
# (user requested names: aime24/aime25, not aime2024/aime2025 in the dir)
TASK_DIR_NAME = {
    "gsm8k":     "gsm8k",
    "math500":   "math500",
    "mmlu_pro":  "mmlu_pro",
    "aime2024":  "aime24",
    "aime2025":  "aime25",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--min_step", type=int, default=20)
    p.add_argument("--max_step", type=int, default=340,
                   help="Cap checkpoints at this training step "
                        "(multi_reward_klcov05 has ckpt-360; 340 aligns with others)")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--output_root_template",
                   default="evaluation_results/accuracy_trajectory_{task_dir}_qwen3b",
                   help="Template for per-task output dir. {task_dir} filled from TASK_DIR_NAME.")
    p.add_argument("--methods", default=",".join(m for m, _ in METHODS),
                   help="Comma-separated subset of method short-names to include")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--wandb", action="store_true",
                   help="Log to wandb (one run per (task, method), step = training step)")
    p.add_argument("--wandb_project", default="unsupervised-rl-eval")
    p.add_argument("--wandb_group", default="trajectory_3b_v1")
    return p.parse_args()


def _discover_jobs(methods_subset, min_step, max_step):
    """Return list of (run_name, ckpt_path, step, method) for every checkpoint in
    [min_step, max_step] that exists on disk."""
    jobs = []
    ckpt_re = re.compile(r"^checkpoint-(\d+)$")
    for method, out_dir in METHODS:
        if method not in methods_subset:
            continue
        out = Path(out_dir)
        if not out.exists():
            print(f"[warn] missing {out_dir} — skipping method={method}")
            continue
        ckpts = []
        for d in out.iterdir():
            m = ckpt_re.match(d.name)
            if not m: continue
            step = int(m.group(1))
            if step < min_step or step > max_step: continue
            ckpts.append((step, d))
        ckpts.sort()
        for step, path in ckpts:
            run_name = f"{method}_ckpt{step:03d}"
            jobs.append((run_name, str(path), step, method))
    return jobs


def _assign_jobs_to_gpus(jobs, num_gpus):
    assignments = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        assignments[i % num_gpus].append(job)
    return assignments


def _launch_worker(gpu_id, jobs_for_gpu, args):
    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path("evaluation_results") / "_logs_trajectory_3b"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build per-task template: the run_task.py arg uses {task} — our
    # user-facing template uses {task_dir} (e.g. aime24 vs aime2024).
    # Translate by pre-substituting per-task then handing literal path to
    # run_task.py via --task_output_dir_template using {task} inside.
    # Simplest: we pass a template with {task} and handle dir-name mapping via
    # a small wrapper env var. But run_task.py only knows task names. Instead,
    # set an env-var map AND rewrite template below for each task separately.
    #
    # Workaround: since user's dir names differ from task names (aime24 vs
    # aime2024), we build ONE template where {task} is the task name, and if
    # the user wants shorter dir names we use TASK_DIR_NAME mapping here by
    # passing a per-task --task_output_dir_template-ish approach. Simpler:
    # pass a template where {task} is literal and accept task_name == dir_name
    # except we override via symlinks. Cleanest instead: pass task_output_dir
    # per-task by calling run_task.py once per task. But that reloads the model
    # per task. To keep one-load-per-ckpt, we hand the run_task.py a template
    # with {task} and rely on it matching the actual task name; then we
    # post-rename the aime2024 dir to aime24 via symlinks below.

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_USE_V1"] = "0"

    for run_name, ckpt, step, method in jobs_for_gpu:
        log_file = log_dir / f"gpu{gpu_id}_{run_name}.log"
        cmd = [
            sys.executable, "-u", str(project_root / "evaluation" / "run_task.py"),
            "--checkpoint", ckpt,
            "--run_name", run_name,
            "--tasks", args.tasks,
            "--task_output_dir_template", args.output_root_template.replace(
                "{task_dir}", "{task}"),
            "--trajectory",
        ]
        if args.max_samples is not None:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.enforce_eager:
            cmd += ["--enforce_eager"]
        if args.overwrite:
            cmd += ["--overwrite"]
        if args.wandb:
            cmd += ["--wandb",
                    "--wandb_project", args.wandb_project,
                    "--wandb_group", args.wandb_group]

        print(f"[GPU{gpu_id}] {run_name} → {log_file}")
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        rc = p.wait()
        print(f"[GPU{gpu_id}] {run_name} exit={rc}")


def _symlink_aime_dirs():
    """User asked for aime24/aime25 dir names; task names are aime2024/aime2025.
    After eval, create convenience symlinks so
    evaluation_results/accuracy_trajectory_aime24_qwen3b points to the
    aime2024-named dir (and aime25 → aime2025)."""
    base = Path("evaluation_results")
    for task, short in [("aime2024", "aime24"), ("aime2025", "aime25")]:
        src = base / f"accuracy_trajectory_{task}_qwen3b"
        dst = base / f"accuracy_trajectory_{short}_qwen3b"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.name)  # relative symlink inside evaluation_results
                print(f"symlink: {dst} → {src.name}")
            except Exception as e:
                print(f"[warn] could not symlink {dst} → {src}: {e}")


def main():
    args = parse_args()
    methods_subset = set(m.strip() for m in args.methods.split(",") if m.strip())

    jobs = _discover_jobs(methods_subset, args.min_step, args.max_step)
    print(f"\nDiscovered {len(jobs)} (method, ckpt) jobs "
          f"in step range [{args.min_step}, {args.max_step}]:")
    from collections import defaultdict
    per_method = defaultdict(list)
    for run, _, step, method in jobs:
        per_method[method].append(step)
    for method in per_method:
        steps = sorted(per_method[method])
        print(f"  {method:25s} {len(steps):3d} ckpts  [{steps[0]}..{steps[-1]}]")

    if not jobs:
        print("No jobs; aborting."); return

    # Round-robin across GPUs
    assignments = _assign_jobs_to_gpus(jobs, args.num_gpus)
    print(f"\nGPU assignments:")
    for i, ass in enumerate(assignments):
        print(f"  GPU{i}: {len(ass)} jobs")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, jobs_for_gpu in enumerate(assignments):
        if not jobs_for_gpu: continue
        p = ctx.Process(target=_launch_worker, args=(gpu_id, jobs_for_gpu, args))
        p.start()
        procs.append(p)

    t0 = time.time()
    for p in procs:
        p.join()
    print(f"\nAll GPUs done in {time.time() - t0:.1f}s")

    _symlink_aime_dirs()


if __name__ == "__main__":
    main()
