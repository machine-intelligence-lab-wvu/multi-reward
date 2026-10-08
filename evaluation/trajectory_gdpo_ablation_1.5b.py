"""Trajectory evaluation for the GDPO-ablation Qwen2.5-1.5B runs.

Iterates every-20-step checkpoints of the multi-reward klcov(0.05) variants
(combnorm / nognorm / gdpo baseline) and evaluates each on GSM8K, MATH500,
and MMLU-Pro. Distributes (method, ckpt) pairs across 4 GPUs; per-GPU
worker loads vLLM once per checkpoint and runs all three tasks.

Output layout (flat, ready for the gdpo_ablation plot script):
  eval_results_final/gdpo_ablation/{method}_ckpt{step:03d}/{run}__gsm8k.json
  eval_results_final/gdpo_ablation/{method}_ckpt{step:03d}/{run}__math500.json
  eval_results_final/gdpo_ablation/{method}_ckpt{step:03d}/{run}__mmlu_pro.json

Idempotent: skips (run, task) pairs whose JSON already exists.

Default scope: only the combnorm run (ckpt 20..280).
Use --methods to add nognorm / gdpo once those runs are available.

Usage:
  python evaluation/trajectory_gdpo_ablation_1.5b.py --num_gpus 4 --enforce_eager
  python evaluation/trajectory_gdpo_ablation_1.5b.py --methods combnorm,gdpo
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# Method short-name → absolute output_dir from training. Each method is one of
# the three GDPO-normalization variants of multi-reward klcov(0.05) Qwen2.5-1.5B.
METHODS = [
    ("multi_reward_combnorm",
     "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_1.5b-logps-combnorm"),
    ("multi_reward_nognorm",
     "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_1.5b-logps-nognorm"),
    ("multi_reward_gdpo",
     "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_1.5b-logps-gdpo"),
]

DEFAULT_METHODS = ["multi_reward_combnorm"]
TASKS = ["gsm8k", "math500", "mmlu_pro"]
DEFAULT_OUTPUT_ROOT = "eval_results_final/gdpo_ablation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--min_step", type=int, default=20)
    p.add_argument("--max_step", type=int, default=280)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help="Comma-separated subset of method short-names to include")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                   help="Flat dir where per-ckpt subdirs will be written")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="unsupervised-rl-eval")
    p.add_argument("--wandb_group", default="trajectory_gdpo_ablation_1.5b_v1")
    return p.parse_args()


def _discover_jobs(methods_subset, min_step, max_step):
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
            if not m:
                continue
            step = int(m.group(1))
            if step < min_step or step > max_step:
                continue
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
    log_dir = Path(args.output_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

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
            "--output_dir", args.output_root,
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
        print("No jobs; aborting.")
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    assignments = _assign_jobs_to_gpus(jobs, args.num_gpus)
    print(f"\nGPU assignments:")
    for i, ass in enumerate(assignments):
        print(f"  GPU{i}: {len(ass)} jobs")

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
    print(f"\nAll GPUs done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
