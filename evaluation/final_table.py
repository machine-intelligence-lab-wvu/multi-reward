"""Final-checkpoint comparison table (3B / 7B, T=1.0, avg-pass@1 + pass@k curve).

Pins the FINAL checkpoint per method (step 340 for 3B, step 240 for 7B —
equal-compute) and evaluates each on {GSM8K, MATH500, AIME24, AIME25} with
per-task n_samples at T=1.0. One sampling pass produces avg-pass@1, the
unbiased pass@k curve (Chen et al. 2021), and maj@k.

Output layout:
    eval_results_final/final_table/{scale}/
        {method}/{method}__{task}.json
        _logs/gpu{i}_{method}__{task}.log

Usage (typically wrapped by submit_final_table_{3b,7b}.sh):
    python -u evaluation/final_table.py --scale 3b --num_gpus 4
    python -u evaluation/final_table.py --scale 7b --num_gpus 4
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Pinned final checkpoints (equal-compute per scale) ────────────────────
FINAL_CHECKPOINTS = {
    "3b": {
        "base":                 "Qwen/Qwen2.5-3B",
        "intuitor":             "${MR_OUTPUT_DIR}/run2_intuitor_qwen25_3b/checkpoint-340",
        "grpo_gt":              "${MR_OUTPUT_DIR}/grpo_gt_qwen25_3b/checkpoint-340",
        "multi_reward_klcov05": "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_3b/checkpoint-340",
    },
    "7b": {
        "base":                 "Qwen/Qwen2.5-7B",
        "intuitor":             "${MR_OUTPUT_DIR}/intuitor_qwen25_7b/checkpoint-240",
        "grpo_gt":              "${MR_OUTPUT_DIR}/grpo_gt_qwen25_7b/checkpoint-240",
        "multi_reward_klcov05": "${MR_OUTPUT_DIR}/multi_reward_final_klcov_0.05_qwen25_7b/checkpoint-240",
    },
}

# ── Per-task sampling config (DeepSeek-R1 / Zuo et al. convention: T=1) ───
TASK_CONFIG = {
    "gsm8k":    {"n_samples": 8,  "max_new_tokens": 1024},
    "math500":  {"n_samples": 16, "max_new_tokens": 3072},
    "aime2024": {"n_samples": 32, "max_new_tokens": 8192},
    "aime2025": {"n_samples": 32, "max_new_tokens": 8192},
}

TEMPERATURE = 1.0
TOP_P = 1.0


def build_jobs(scale, methods=None, tasks=None):
    if methods is None:
        methods = list(FINAL_CHECKPOINTS[scale].keys())
    if tasks is None:
        tasks = list(TASK_CONFIG.keys())
    jobs = []
    for method in methods:
        if method not in FINAL_CHECKPOINTS[scale]:
            print(f"[final_table] skip unknown method {method}")
            continue
        ckpt = FINAL_CHECKPOINTS[scale][method]
        for task in tasks:
            if task not in TASK_CONFIG:
                print(f"[final_table] skip unknown task {task}")
                continue
            jobs.append((method, ckpt, task, TASK_CONFIG[task]))
    return jobs


def assign_jobs(jobs, num_gpus):
    """Round-robin. AIME jobs are longer-tail; interleave methods so each GPU
    gets a mix rather than all-AIME on one GPU."""
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

    for method, ckpt, task, cfg in jobs_for_gpu:
        log_file = log_dir / f"gpu{gpu_id}_{method}__{task}.log"
        cmd = [
            sys.executable, "-u", str(PROJECT_ROOT / "evaluation" / "run_task.py"),
            "--checkpoint", ckpt,
            "--run_name", method,
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
        print(f"[GPU{gpu_id}] launching {method}__{task} "
              f"(N={cfg['n_samples']}, max_tok={cfg['max_new_tokens']})  → {log_file}")
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        p.wait()
        print(f"[GPU{gpu_id}] {method}__{task} exit={p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["3b", "7b"], required=True)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--output_dir", default=None,
                    help="default: eval_results_final/final_table/{scale}")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated subset of methods (default: all 4).")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated subset of tasks (default: all 4).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="unsupervised-rl-eval")
    ap.add_argument("--wandb_group", default=None,
                    help="default: final_table_{scale}_v1")
    args = ap.parse_args()

    methods = args.methods.split(",") if args.methods else None
    tasks = args.tasks.split(",") if args.tasks else None

    out_dir = Path(args.output_dir or f"eval_results_final/final_table/{args.scale}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_args = []
    if args.wandb:
        group = args.wandb_group or f"final_table_{args.scale}_v1"
        wandb_args = ["--wandb", "--wandb_project", args.wandb_project,
                      "--wandb_group", group]

    jobs = build_jobs(args.scale, methods, tasks)
    if not jobs:
        print("[final_table] no jobs"); return

    print(f"[final_table] scale={args.scale} num_gpus={args.num_gpus} jobs={len(jobs)}")
    for j in jobs:
        print(f"  {j[0]:22s} {j[2]:10s} N={j[3]['n_samples']:>2}  max_tok={j[3]['max_new_tokens']}")

    assignments = assign_jobs(jobs, args.num_gpus)
    for i, ass in enumerate(assignments):
        names = [f"{j[0]}__{j[2]}" for j in ass]
        print(f"  GPU{i}: {names}")

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
    print("[final_table] all workers done.")


if __name__ == "__main__":
    main()
