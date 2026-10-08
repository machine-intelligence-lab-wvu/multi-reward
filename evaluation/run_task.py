"""Single-GPU runner. Loads vLLM once, evaluates a checkpoint on N tasks.

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluation/run_task.py \
      --checkpoint /path/to/checkpoint-300 \
      --run_name multi_reward_base_ckpt300 \
      --tasks gsm8k,math500,math,aime2024,aime2025,mmlu_pro \
      --output_dir evaluation_results/best_ckpts_v1

Writes one JSON per (run, task) at
  {output_dir}/{run_name}/{task}.json

Skips tasks whose JSON already exists (idempotent) unless --overwrite.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

# vLLM must be imported after CUDA_VISIBLE_DEVICES is set in env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Training system prompt (from every recipe YAML)
TRAINING_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. "
    "You FIRST think about the reasoning process step by step and then provide the user with the answer. "
    "Please enclose your final answer in the box: \\boxed{Your Answer}. "
    "Please stop generation immediately after outputing the box."
)

# Fallback for BASE models that ship no chat template (meta-llama/Llama-3.2-3B).
# Must stay identical to `chat_template:` in recipes/rebuttal/B6_llama32_3b_*.yaml.
# B6 CHECKPOINTS need no fallback: training saves the template into their
# tokenizer_config (open_r1_imp/utils.get_tokenizer).
BASE_MODEL_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}\n\n"
    "{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}"
    "Problem: {{ message['content'] }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}\n\nSolution:{% endif %}"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="HF model id or local checkpoint path")
    p.add_argument("--run_name", required=True,
                   help="Short identifier used in output directory and logs")
    p.add_argument("--tasks", required=True,
                   help="Comma-separated task names (gsm8k,math500,math,aime2024,aime2025,mmlu_pro)")
    p.add_argument("--output_dir", default="evaluation_results/best_ckpts_v1")
    p.add_argument("--task_output_dir_template", default=None,
                   help="Optional .format template for per-task output dir, "
                        "e.g. 'evaluation_results/accuracy_trajectory_{task}_qwen3b'. "
                        "If set, each task's JSON is written to "
                        "{template_with_task}/{run_name}/{run_name}__{task}.json, "
                        "overriding --output_dir.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=None,
                   help="If set, --max_samples takes a SEEDED RANDOM subset instead of the "
                        "first N. Needed for any subsampled benchmark whose rows are ordered "
                        "by category — MMLU-Pro is: its first 2000 test rows are only "
                        "law/business/psychology (3 of 14 categories), so `--max_samples 2000` "
                        "alone does not estimate MMLU-Pro accuracy. The same seed yields the "
                        "same subset across models, which is required for a fair comparison. "
                        "Leave unset to preserve the original first-N behaviour exactly.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_new_tokens_override", type=int, default=None,
                   help="Override task.max_new_tokens (e.g. for AIME at T=1 where "
                        "8k traces are common). Default: use the task's default.")
    p.add_argument("--max_model_len", type=int, default=16384,
                   help="vLLM max seq len (needs headroom for 5-shot MMLU-Pro prompts)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graphs (slower but avoids illegal-instruction crashes on GH200)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run tasks even if output JSON exists")
    p.add_argument("--wandb", action="store_true",
                   help="Log results to wandb")
    p.add_argument("--wandb_project", default="unsupervised-rl-eval",
                   help="wandb project name")
    p.add_argument("--wandb_group", default="best_ckpts_v1",
                   help="wandb group name (groups all runs from one eval sweep)")
    p.add_argument("--trajectory", action="store_true",
                   help="Trajectory mode: treat run_name as '{method}_ckpt{step}' and "
                        "log all checkpoints of a method under one wandb run "
                        "(wandb step = training step).")
    return p.parse_args()


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from evaluation.tasks import get_task

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    def _out_dir_for_task(tn: str) -> Path:
        if args.task_output_dir_template:
            base = Path(args.task_output_dir_template.format(task=tn))
        else:
            base = Path(args.output_dir)
        return base / args.run_name

    task_out_dirs = {tn: _out_dir_for_task(tn) for tn in task_names}
    for d in set(task_out_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # Filter tasks that already have output (idempotent)
    todo_tasks = []
    for tn in task_names:
        dst = task_out_dirs[tn] / f"{args.run_name}__{tn}.json"
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst}")
            continue
        todo_tasks.append(tn)
    if not todo_tasks:
        print(f"[done] all tasks already complete for {args.run_name}")
        return

    # Trajectory mode: parse "{method}_ckpt{step:03d}" → (method, step)
    # so the wandb run is per-method and x-axis is training step.
    trajectory_step = None
    trajectory_method = None
    if args.trajectory:
        import re as _re
        m = _re.match(r"^(.*)_ckpt(\d+)$", args.run_name)
        if m:
            trajectory_method = m.group(1)
            trajectory_step = int(m.group(2))
        else:
            print(f"[{args.run_name}] warning: --trajectory but run_name doesn't match "
                  f"'{{method}}_ckpt{{step}}'; logging as a one-off run")

    # Init wandb
    # - default: one run per checkpoint, wandb step = 0 (scalar summary)
    # - trajectory mode: one run per METHOD, wandb step = training step; multiple
    #   processes re-join by using a deterministic run id.
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            if trajectory_method is not None:
                # Deterministic id so sibling processes resume the same run
                run_id = f"{args.wandb_group}__{trajectory_method}"
                wandb_name = trajectory_method
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    group=args.wandb_group,
                    name=wandb_name,
                    id=run_id,
                    resume="allow",
                    job_type="trajectory",
                    config={
                        "method": trajectory_method,
                        "tasks": task_names,
                        "temperature": args.temperature,
                        "n_samples": args.n_samples,
                        "max_model_len": args.max_model_len,
                    },
                    reinit=True,
                )
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    group=args.wandb_group,
                    name=args.run_name,
                    job_type="eval",
                    config={
                        "checkpoint": args.checkpoint,
                        "tasks": task_names,
                        "temperature": args.temperature,
                        "n_samples": args.n_samples,
                        "max_samples": args.max_samples,
                        "max_model_len": args.max_model_len,
                        "system_prompt": TRAINING_SYSTEM_PROMPT,
                    },
                    reinit=True,
                )
            print(f"[{args.run_name}] wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"[{args.run_name}] wandb init failed (continuing without): {e}")
            wandb_run = None

    print(f"[{args.run_name}] Loading vLLM for {args.checkpoint}")
    t0 = time.time()
    llm = LLM(
        model=args.checkpoint,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print(f"[{args.run_name}] no chat template on {args.checkpoint} -> "
              "using BASE_MODEL_CHAT_TEMPLATE (plain Problem/Solution format)")
        tokenizer.chat_template = BASE_MODEL_CHAT_TEMPLATE
    print(f"[{args.run_name}] vLLM loaded in {time.time() - t0:.1f}s")

    task_results = {}
    for tn in todo_tasks:
        acc = _run_one_task(
            task_name=tn,
            llm=llm,
            tokenizer=tokenizer,
            args=args,
            out_dir=task_out_dirs[tn],
            wandb_run=wandb_run,
            trajectory_step=trajectory_step,
        )
        task_results[tn] = acc

    # Also include tasks that were already done (read their JSONs for summary)
    for tn in task_names:
        if tn not in task_results:
            dst = task_out_dirs[tn] / f"{args.run_name}__{tn}.json"
            if dst.exists():
                try:
                    data = json.loads(dst.read_text())
                    task_results[tn] = data["accuracy"]
                except Exception:
                    pass

    if wandb_run is not None:
        try:
            # Flat summary keys for easy sorting in wandb UI
            for tn, acc in task_results.items():
                wandb_run.summary[f"{tn}/accuracy"] = acc
            wandb_run.summary["mean_accuracy"] = (
                sum(task_results.values()) / len(task_results) if task_results else 0.0
            )
            wandb_run.finish()
        except Exception as e:
            print(f"[{args.run_name}] wandb finish failed: {e}")

    print(f"[{args.run_name}] all tasks done: {task_results}")


def _build_prompt(tokenizer, system_prompt: str, user_content: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def _run_one_task(task_name: str, llm, tokenizer, args, out_dir: Path,
                  wandb_run=None, trajectory_step=None):
    from vllm import SamplingParams
    from evaluation.tasks import get_task

    print(f"[{args.run_name}] === task: {task_name} ===")
    task = get_task(task_name)
    if args.sample_seed is None:
        examples = task.load(max_samples=args.max_samples)
    else:
        # Load the FULL set, then take a seeded uniform-random subset. Doing the
        # cut here (rather than via task.load's `select(range(N))`) keeps every
        # task loader untouched and avoids the category-ordering bias.
        import random as _random
        examples = task.load(max_samples=None)
        if args.max_samples and len(examples) > args.max_samples:
            idx = sorted(_random.Random(args.sample_seed).sample(
                range(len(examples)), args.max_samples))
            examples = [examples[i] for i in idx]
            print(f"[{args.run_name}/{task_name}] seeded random subset: "
                  f"{len(examples)} of full set (seed={args.sample_seed})")
    print(f"[{args.run_name}/{task_name}] {len(examples)} examples "
          f"(chat_template={task.use_chat_template})")

    if task.use_chat_template:
        prompts = [_build_prompt(tokenizer, TRAINING_SYSTEM_PROMPT, ex.question) for ex in examples]
    else:
        # Raw completion prompts (e.g. MMLU-Pro 5-shot CoT): send text verbatim
        prompts = [ex.question for ex in examples]

    max_tok = args.max_new_tokens_override or task.max_new_tokens
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tok,
        min_tokens=task.min_new_tokens,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_time = time.time() - t0
    print(f"[{args.run_name}/{task_name}] generation: {gen_time:.1f}s")

    # Score all N samples per example, then compute:
    #   - pass@1 (per-problem):
    #         N=1 → first-sample 0/1   (greedy convention)
    #         N>1 → mean of per-sample 0/1 ∈ [0,1]   (DeepSeek-R1 / Chen 2021 avg-pass@1)
    #   - pass@k curve: unbiased Chen et al. 2021 estimator for k ∈ {1,2,4,8,16,32} ∩ [1,N]
    #   - maj@k curve: majority vote over first k samples, k ∈ {1,2,4,8,16,32} ∩ [1,N]
    #   - legacy single-N pass@k / maj@k fields preserved for back-compat with summarize.py
    from math import comb as _comb
    _curve_ks = (1, 2, 4, 8, 16, 32)
    sum_pass1   = 0.0
    sum_passk_curve = {k: 0.0 for k in _curve_ks if k <= args.n_samples}
    sum_majk_curve  = {k: 0.0 for k in _curve_ks if k <= args.n_samples}
    records = []
    for ex, out in zip(examples, outputs):
        completions = [s.text for s in out.outputs]
        per_sample_scores = [task.score(c, ex) for c in completions]
        n_samp = len(per_sample_scores)
        n_correct = sum(1 for s in per_sample_scores if s >= 1.0)

        # pass@1 (per problem)
        if n_samp > 1:
            pass1 = n_correct / n_samp
        else:
            pass1 = 1.0 if per_sample_scores[0] >= 1.0 else 0.0
        sum_pass1 += pass1

        # pass@k curve (unbiased) and maj@k curve (first-k majority vote)
        passk_curve = {}
        majk_curve  = {}
        for k_ in _curve_ks:
            if k_ > n_samp:
                continue
            if (n_samp - n_correct) < k_:
                passk_curve[k_] = 1.0
            else:
                passk_curve[k_] = 1.0 - _comb(n_samp - n_correct, k_) / _comb(n_samp, k_)
            sum_passk_curve[k_] += passk_curve[k_]
            first_k = per_sample_scores[:k_]
            majk_curve[k_] = 1.0 if sum(s >= 1.0 for s in first_k) > k_ / 2 else 0.0
            sum_majk_curve[k_] += majk_curve[k_]

        # Strip heavy task-specific metadata (LCB test cases can be MBs each →
        # 8GB+ JSON for 1055 problems). Tests are recoverable from the dataset.
        meta = {k: v for k, v in (ex.metadata or {}).items()
                if k not in ("public_tests", "private_tests")}
        records.append({
            "id": ex.id,
            "gold": ex.gold,
            "per_sample_scores": per_sample_scores,
            "pass@1": pass1,
            "pass@k_curve": passk_curve,
            "maj@k_curve": majk_curve,
            # Legacy single-N fields (k = args.n_samples)
            "pass@k": passk_curve.get(n_samp, pass1),
            "maj@k": majk_curve.get(n_samp, pass1),
            "completions": completions,
            "metadata": meta,
        })

    n_total = len(examples)
    accuracy_pass1 = sum_pass1 / n_total if n_total else 0.0
    passk_curve_avg = {k_: (s / n_total if n_total else 0.0)
                       for k_, s in sum_passk_curve.items()}
    majk_curve_avg  = {k_: (s / n_total if n_total else 0.0)
                       for k_, s in sum_majk_curve.items()}
    k = args.n_samples
    accuracy_passk = passk_curve_avg.get(k, accuracy_pass1)
    accuracy_majk  = majk_curve_avg.get(k, accuracy_pass1)
    # back-compat counters (treated as floats when N>1)
    correct_pass1 = sum_pass1
    correct_passk = sum(1 for r in records if r["pass@k"] >= 1.0)
    correct_majk  = sum(1 for r in records if r["maj@k"] >= 1.0)

    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "task": task_name,
        "metric": task.metric,
        "n_examples": n_total,
        "n_samples": k,
        # Legacy fields (backward-compat with summarize.py / existing JSONs)
        "correct": correct_pass1,
        "accuracy": accuracy_pass1,
        # New pass@k fields
        "pass@1": accuracy_pass1,
        f"pass@{k}": accuracy_passk,
        f"maj@{k}": accuracy_majk,
        # Full unbiased pass@k / maj@k curves over k ∈ {1,2,4,8,16,32} ∩ [1,N]
        "pass@k_curve": {str(kk): v for kk, v in passk_curve_avg.items()},
        "maj@k_curve":  {str(kk): v for kk, v in majk_curve_avg.items()},
        "correct_pass1": correct_pass1,
        f"correct_pass{k}": correct_passk,
        f"correct_maj{k}": correct_majk,
        "generation_time_s": gen_time,
        # Records how the example set was cut, so a subsampled result can never be
        # mistaken for a full-set one when reading the JSON later.
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "subset": ("full" if not args.max_samples
                   else ("seeded_random" if args.sample_seed is not None else "first_n")),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": max_tok,
        "system_prompt": TRAINING_SYSTEM_PROMPT if task.use_chat_template else None,
        "use_chat_template": task.use_chat_template,
        "records": records,
    }
    dst = out_dir / f"{args.run_name}__{task_name}.json"
    dst.write_text(json.dumps(result, indent=2, default=str))
    if k > 1:
        print(f"[{args.run_name}/{task_name}] pass@1={accuracy_pass1:.4f}  "
              f"pass@{k}={accuracy_passk:.4f}  maj@{k}={accuracy_majk:.4f}  → {dst}")
    else:
        print(f"[{args.run_name}/{task_name}] accuracy={accuracy_pass1:.4f} "
              f"({correct_pass1}/{n_total}) → {dst}")

    if wandb_run is not None:
        try:
            log = {
                f"{task_name}/pass@1": accuracy_pass1,
                f"{task_name}/total": n_total,
                f"{task_name}/gen_time_s": gen_time,
            }
            if k > 1:
                log[f"{task_name}/pass@{k}"] = accuracy_passk
                log[f"{task_name}/maj@{k}"] = accuracy_majk
            # In trajectory mode, also log the training step as its own metric
            # (so the user can pick "train_step" as the wandb x-axis in the UI,
            # avoiding the monotonic-step assumption of wandb's internal axis).
            if trajectory_step is not None:
                log["train_step"] = trajectory_step
                # Use wandb's step only if it would be monotonic; otherwise
                # let wandb auto-increment its internal step and rely on
                # train_step as the displayed x-axis.
                try:
                    wandb_run.log(log, step=trajectory_step)
                except Exception:
                    wandb_run.log(log)
            else:
                wandb_run.log(log)
        except Exception as e:
            print(f"[{args.run_name}/{task_name}] wandb log failed: {e}")

    return accuracy_pass1


if __name__ == "__main__":
    main()
