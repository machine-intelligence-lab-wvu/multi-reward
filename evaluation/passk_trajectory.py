"""Avg-pass@1 / pass@k trajectory runner.

Same protocol as `final_table.py` (T=1.0, per-task N) but expands across
multiple checkpoints of each method to give a trajectory view of
avg-pass@1 and the pass@k curve.

For the 7B sweep this evaluates every 40-step checkpoint of three methods
on GSM8K + MATH500. One (method, step, task) tuple per subprocess; 4 GPUs
round-robin.

Output layout (parallel to trajectory_7B/ structure):
    eval_results_final/passk_trajectory_7B/
        {method}_ckpt{step:03d}/{method}_ckpt{step:03d}__{task}.json
        _logs/gpu{i}_{method}_ckpt{step:03d}__{task}.log

Usage:
    python -u evaluation/passk_trajectory.py --scale 7b
    python -u evaluation/passk_trajectory.py --scale 7b --steps 80,160,240
    python -u evaluation/passk_trajectory.py --scale 7b --methods intuitor
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base directories per (scale, method); checkpoint dir = "{base}/checkpoint-{step}".
TRAJECTORY_RUNS = {
    "7b": {
        "intuitor":             "${MR_OUTPUT_DIR}/intuitor_qwen25_7b",
        "grpo_gt":              "${MR_OUTPUT_DIR}/grpo_gt_qwen25_7b",
        "multi_reward_klcov05": "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_7b",
    },
    "3b": {
        "intuitor":             "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_3b",
        "grpo_gt":              "${MR_OUTPUT_DIR}/grpo_gt_qwen25_3b",
        "multi_reward_klcov05": "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_3b",
    },
}

TASK_CONFIG = {
    "gsm8k":    {"n_samples": 8,  "max_new_tokens": 1024},
    "math500":  {"n_samples": 16, "max_new_tokens": 3072},
    "aime2024": {"n_samples": 32, "max_new_tokens": 8192},
    "aime2025": {"n_samples": 32, "max_new_tokens": 8192},
}

TEMPERATURE = 1.0
TOP_P = 1.0

DEFAULT_STEPS = [40, 80, 120, 160, 200, 240]
DEFAULT_TASKS_PER_SCALE = {
    "7b": ["gsm8k", "math500"],
    "3b": ["gsm8k", "math500"],
}


def build_jobs(scale, steps, methods=None, tasks=None):
    if methods is None:
        methods = list(TRAJECTORY_RUNS[scale].keys())
    if tasks is None:
        tasks = list(DEFAULT_TASKS_PER_SCALE[scale])
    jobs = []
    for method in methods:
        if method not in TRAJECTORY_RUNS[scale]:
            print(f"[passk_trajectory] skip unknown method {method}")
            continue
        base = TRAJECTORY_RUNS[scale][method]
        for step in steps:
            ckpt = f"{base}/checkpoint-{step}"
            if not Path(ckpt).exists():
                print(f"[passk_trajectory] skip missing ckpt {ckpt}")
                continue
            run_name = f"{method}_ckpt{step:03d}"
            for task in tasks:
                if task not in TASK_CONFIG:
                    continue
                jobs.append((run_name, ckpt, task, TASK_CONFIG[task]))
    return jobs


def assign_jobs(jobs, num_gpus):
    """Round-robin so each GPU gets a mix of methods/steps/tasks."""
    assignments = [[] for _ in range(num_gpus)]
    for i, job in enumerate(jobs):
        assignments[i % num_gpus].append(job)
    return assignments


def launch_worker(gpu_id, jobs_for_gpu, out_dir, overwrite, wandb_args):
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"

    for run_name, ckpt, task, cfg in jobs_for_gpu:
        log_file = log_dir / f"gpu{gpu_id}_{run_name}__{task}.log"
        cmd = [
            sys.executable, "-u", str(PROJECT_ROOT / "evaluation" / "run_task.py"),
            "--checkpoint", ckpt,
            "--run_name", run_name,
            "--tasks", task,
            "--output_dir", str(out_dir),
            "--n_samples", str(cfg["n_samples"]),
            "--temperature", str(TEMPERATURE),
            "--top_p", str(TOP_P),
            "--max_new_tokens_override", str(cfg["max_new_tokens"]),
            "--enforce_eager",
        ]
        if overwrite:
            cmd += ["--overwrite"]
        if wandb_args:
            cmd += wandb_args
        print(f"[GPU{gpu_id}] launching {run_name}__{task} "
              f"(N={cfg['n_samples']}, max_tok={cfg['max_new_tokens']})  → {log_file}")
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        p.wait()
        print(f"[GPU{gpu_id}] {run_name}__{task} exit={p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["3b", "7b"], required=True)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--steps", default=",".join(str(s) for s in DEFAULT_STEPS),
                    help="Comma-separated checkpoint steps (default: 40,80,120,160,200,240)")
    ap.add_argument("--output_dir", default=None,
                    help="default: eval_results_final/passk_trajectory_{scale}")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated subset of methods")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of tasks (default: gsm8k,math500)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="unsupervised-rl-eval")
    ap.add_argument("--wandb_group", default=None,
                    help="default: passk_trajectory_{scale}_v1")
    args = ap.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    out_dir = Path(args.output_dir or
                   f"eval_results_final/passk_trajectory_{args.scale.upper()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_args = []
    if args.wandb:
        group = args.wandb_group or f"passk_trajectory_{args.scale}_v1"
        wandb_args = ["--wandb", "--wandb_project", args.wandb_project,
                      "--wandb_group", group]

    jobs = build_jobs(args.scale, steps, methods, tasks)
    if not jobs:
        print("[passk_trajectory] no jobs"); return

    print(f"[passk_trajectory] scale={args.scale} num_gpus={args.num_gpus} jobs={len(jobs)}")
    for j in jobs:
        print(f"  {j[0]:35s} {j[2]:10s} N={j[3]['n_samples']:>2}  max_tok={j[3]['max_new_tokens']}")

    assignments = assign_jobs(jobs, args.num_gpus)
    for i, ass in enumerate(assignments):
        print(f"  GPU{i}: {len(ass)} jobs")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, ass in enumerate(assignments):
        if not ass:
            continue
        p = ctx.Process(target=launch_worker,
                        args=(gpu_id, ass, out_dir, args.overwrite, wandb_args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[passk_trajectory] all workers done.")


if __name__ == "__main__":
    main()
