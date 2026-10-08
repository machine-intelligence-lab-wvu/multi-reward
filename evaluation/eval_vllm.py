"""
Fast vLLM-based evaluation for LoRA and full fine-tuned checkpoints.

Uses vLLM for batched inference — typically 100-200x faster than the old
sequential HuggingFace evaluation (eval_checkpoint.py). For LoRA checkpoints,
uses vLLM's native LoRA support (enable_lora=True) by default, which applies
adapter weights dynamically at inference time without merging — matching
exactly how training inference works.

LoRA modes:
  - native (default): Loads base model + LoRA adapter via vLLM's native
    LoRA support. No merge step, exact weights, no precision loss.
    Requires: VLLM_WORKER_MULTIPROC_METHOD=spawn and
              VLLM_ENABLE_V1_MULTIPROCESSING=0 environment variables.
  - merge: Merges LoRA into base model on CPU, then loads merged model
    into vLLM. Faster inference but ~1-2% accuracy drop from merge
    precision loss.

System prompts:
  - training (default): The prompt used during GRPO training.
  - eval_v2: Alternative prompt with stricter formatting instructions.
    Tends to score ~5-7% higher due to cleaner boxed answer extraction.
  - all: Run both prompts for comparison.

  NOTE: Each system prompt must be run as a separate process to avoid
  CUDA re-initialization errors. Use the shell script or run separately.

Benchmarks: GSM8K (1319 problems), MATH500 (500 problems), full MATH.
Answer extraction: Uses regex-based \\boxed{} extraction with numeric
normalization from train_4090_imp/rewards/answer_extraction.py.

Usage:
  # Basic: single LoRA checkpoint on GSM8K (greedy, training prompt)
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  CUDA_VISIBLE_DEVICES=2 python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --dataset gsm8k

  # Use the eval_v2 system prompt instead
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --system_prompt eval_v2

  # Multiple checkpoints on multiple benchmarks
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-400 \\
                    outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --dataset gsm8k math500

  # All checkpoints in an output directory
  ... python evaluation/eval_vllm.py \\
      --output_dir outputs/intuitor_klcov_lora_local \\
      --dataset gsm8k

  # Majority voting (16 samples, temperature 0.6)
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --dataset gsm8k --n_samples 16 --temperature 0.6

  # Quick test (100 samples)
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --dataset gsm8k --max_samples 100

  # Thinking mode (Qwen3)
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/thinking/checkpoint-100 \\
      --dataset gsm8k --thinking --max_new_tokens 4096

  # Also evaluate the base model for comparison
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --dataset gsm8k --eval_base_model

  # Use merge mode instead of native LoRA (faster but ~1-2% accuracy drop)
  ... python evaluation/eval_vllm.py \\
      --checkpoint outputs/intuitor_klcov_lora_local/checkpoint-700 \\
      --lora_mode merge

  # Or use the convenience shell script:
  bash open_r1_imp/scripts/run_eval_vllm.sh
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_4090_imp"))
from rewards.answer_extraction import (
    answers_equivalent,
    extract_boxed_answer,
    normalize_answer,
)


SYSTEM_PROMPTS = {
    "training": (
        "You are a helpful AI Assistant, designed to provided well-reasoned "
        "and detailed responses. You FIRST think about the reasoning process "
        "step by step and then provide the user with the answer. Please enclose "
        "your final answer in the box: \\boxed{Your Answer}. Please stop "
        "generation immediately after outputing the box."
    ),
    "eval_v2": (
        "You are a helpful AI Assistant, designed to provided well-reasoned "
        "You FIRST think about the reasoning process step by step and then "
        "provide the user with the answer. Solve the problem step by step "
        "and show your reasoning clearly."
        "On the final line only, write exactly: Final Answer: \\boxed{...}. "
        "Do not output any text after that final line."
    ),
}


# ════════════════════════════════════════════════════════════
# DATASET LOADERS
# ════════════════════════════════════════════════════════════

def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"question": x["question"],
             "answer": x["answer"].split("####")[-1].strip()}
            for x in ds]


def load_math500():
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    data = []
    for x in ds:
        question = x.get("problem", "")
        answer = x.get("answer", "")
        if not answer:
            sol = x.get("solution", "")
            extracted = extract_boxed_answer(sol)
            answer = extracted if extracted else ""
        data.append({"question": question, "answer": answer})
    return data


def load_math_test():
    dataset_names = [
        ("nlile/hendrycks-MATH-benchmark", None, "test"),
        ("hendrycks/competition_math", None, "test"),
        ("DigitalLearningGmbH/MATH-lighteval", "default", "test"),
    ]
    for name, config, split in dataset_names:
        try:
            ds = load_dataset(name, config, split=split)
            data = []
            for x in ds:
                question = x.get("problem", x.get("question", ""))
                answer = x.get("answer", "")
                if not answer:
                    sol = x.get("solution", "")
                    extracted = extract_boxed_answer(sol)
                    answer = extracted if extracted else ""
                data.append({"question": question, "answer": answer})
            return data
        except Exception:
            continue
    raise RuntimeError("Could not load MATH test set.")


def load_eval_dataset(name):
    loaders = {"gsm8k": load_gsm8k, "math500": load_math500, "math": load_math_test}
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(loaders.keys())}")
    data = loaders[name]()
    print(f"Loaded {name.upper()}: {len(data)} problems")
    return data


# ════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ════════════════════════════════════════════════════════════

def is_lora_checkpoint(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "adapter_config.json"))


def resolve_base_model(checkpoint_path, base_model_override=None):
    if base_model_override:
        return base_model_override
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if base:
            return base
    raise ValueError(
        f"Cannot detect base model from {checkpoint_path}. Use --model_name.")


def merge_lora_checkpoint(checkpoint_path, base_model_name, merge_dir):
    """Merge LoRA adapter into base model and save to merge_dir."""
    print(f"  Merging LoRA adapter into base model...")
    print(f"    Base model:  {base_model_name}")
    print(f"    Adapter:     {checkpoint_path}")
    print(f"    Merge dir:   {merge_dir}")

    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(merge_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(merge_dir)

    del base_model, model, merged
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"    Merged in {elapsed:.1f}s")
    return merge_dir


def find_checkpoints(output_dir):
    """Find all checkpoint-* directories sorted by step number."""
    checkpoints = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if os.path.isdir(path):
            match = re.search(r"checkpoint-(\d+)", path)
            if match:
                checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ════════════════════════════════════════════════════════════

def build_prompts(tokenizer, questions, system_prompt, thinking=False):
    prompts = []
    for q in questions:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        try:
            p = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking)
        except TypeError:
            p = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    return prompts


# ════════════════════════════════════════════════════════════
# vLLM EVALUATION
# ════════════════════════════════════════════════════════════

def create_vllm_engine(model_path, max_model_len=4096, tensor_parallel_size=1,
                       gpu_memory_utilization=0.9, lora_path=None,
                       max_lora_rank=64):
    """Create a vLLM engine. Only call once per process to avoid CUDA re-init."""
    print(f"  Loading model into vLLM: {model_path}")
    if lora_path:
        print(f"  Native LoRA adapter: {lora_path}")

    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f"  vLLM loaded in {load_time:.1f}s")

    lora_request = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("eval_lora", 1, lora_path)

    return llm, lora_request


def evaluate_with_vllm_engine(llm, lora_request, tokenizer, data, system_prompt,
                              max_new_tokens=1024, min_new_tokens=64,
                              temperature=0.0, n_samples=1, thinking=False):
    """Run evaluation using an already-loaded vLLM engine."""
    questions = [x["question"] for x in data]
    answers = [x["answer"] for x in data]
    prompts = build_prompts(tokenizer, questions, system_prompt, thinking=thinking)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
    )

    print(f"  Generating {len(prompts)} prompts × {n_samples} samples "
          f"(temp={temperature}, max_tokens={max_new_tokens})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request,
                           use_tqdm=True)
    gen_time = time.time() - t0
    print(f"  Generation done in {gen_time:.1f}s "
          f"({gen_time / len(prompts):.2f}s/prompt)")

    # Score results
    correct_pass1 = 0
    correct_majority = 0
    total = len(data)
    records = []

    for idx, (out, gt, q) in enumerate(zip(outputs, answers, questions)):
        completions = [o.text for o in out.outputs]

        # Handle thinking mode: extract answer after </think>
        if thinking:
            answer_texts = []
            for c in completions:
                pos = c.rfind("</think>")
                answer_texts.append(c[pos + len("</think>"):] if pos >= 0 else c)
        else:
            answer_texts = completions

        extracted = [extract_boxed_answer(t) for t in answer_texts]

        # pass@1: first completion
        pred = extracted[0]
        is_correct = answers_equivalent(pred, gt) if pred else False
        if is_correct:
            correct_pass1 += 1

        # majority vote
        normalized = [normalize_answer(a) for a in extracted if a is not None]
        if normalized:
            majority = Counter(normalized).most_common(1)[0][0]
            if answers_equivalent(majority, gt):
                correct_majority += 1

        records.append({
            "id": idx,
            "question": q[:200],
            "ground_truth": gt,
            "predicted": pred,
            "correct": bool(is_correct),
            "completion_preview": completions[0][:400],
        })

    # Cleanup vLLM
    del llm
    torch.cuda.empty_cache()

    return {
        "pass@1": correct_pass1 / total if total > 0 else 0.0,
        f"majority@{n_samples}": correct_majority / total if total > 0 else 0.0,
        "correct_pass1": correct_pass1,
        "correct_majority": correct_majority,
        "total": total,
        "generation_time_s": gen_time,
        "records": records,
    }


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def eval_single_checkpoint(checkpoint_path, args, datasets_cache, prompt_configs):
    """Evaluate one checkpoint on all datasets x all system prompts.

    Creates the vLLM engine ONCE, then loops over prompt variants and datasets.
    Returns a list of result dicts (one per prompt variant).

    Args:
        prompt_configs: list of (prompt_name, system_prompt_text) tuples.
    """
    step_match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    step = int(step_match.group(1)) if step_match else 0
    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    is_lora = is_lora_checkpoint(checkpoint_path)
    lora_mode = args.lora_mode

    print(f"\n{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Type: {'LoRA' if is_lora else 'Full FT'}  |  Step: {step}")
    print(f"LoRA mode: {lora_mode if is_lora else 'n/a'}")
    print(f"Prompts: {[p[0] for p in prompt_configs]}")
    print(f"{'=' * 60}")

    # Determine model path and LoRA loading strategy
    merge_dir = None
    lora_path = None

    if is_lora:
        base_model = resolve_base_model(checkpoint_path, args.model_name)
        if lora_mode == "native":
            model_path = base_model
            lora_path = checkpoint_path
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        else:
            merge_dir = os.path.join(
                tempfile.gettempdir(),
                f"merged_lora_{ckpt_name}_{os.getpid()}")
            model_path = merge_lora_checkpoint(checkpoint_path, base_model, merge_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model_path = checkpoint_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create engine ONCE
    llm, lora_request = create_vllm_engine(
        model_path=model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_path=lora_path,
        max_lora_rank=args.max_lora_rank,
    )

    all_prompt_results = []
    try:
        for prompt_name, system_prompt in prompt_configs:
            print(f"\n  ── System prompt: {prompt_name} ──")

            run_results = {"step": step, "checkpoint": checkpoint_path,
                           "type": "lora" if is_lora else "full",
                           "lora_mode": lora_mode if is_lora else "n/a",
                           "system_prompt": prompt_name}

            for dataset_name in args.dataset:
                if dataset_name not in datasets_cache:
                    datasets_cache[dataset_name] = load_eval_dataset(dataset_name)
                data = datasets_cache[dataset_name]
                if args.max_samples:
                    data = data[:args.max_samples]

                print(f"\n  --- {dataset_name.upper()} ({len(data)} samples) ---")
                result = evaluate_with_vllm_engine(
                    llm=llm,
                    lora_request=lora_request,
                    tokenizer=tokenizer,
                    data=data,
                    system_prompt=system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    n_samples=args.n_samples,
                    thinking=args.thinking,
                )

                run_results[dataset_name] = result
                print(f"  {dataset_name.upper()}: pass@1={result['pass@1']:.4f} "
                      f"({result['correct_pass1']}/{result['total']}) "
                      f"majority@{args.n_samples}={result[f'majority@{args.n_samples}']:.4f} "
                      f"[{result['generation_time_s']:.0f}s]")

            all_prompt_results.append(run_results)
    finally:
        del llm
        torch.cuda.empty_cache()
        if merge_dir and os.path.isdir(merge_dir):
            print(f"  Cleaning up merge dir: {merge_dir}")
            shutil.rmtree(merge_dir, ignore_errors=True)

    return all_prompt_results


def main():
    p = argparse.ArgumentParser(
        description="Fast vLLM-based evaluation for LoRA and full checkpoints")
    # Checkpoint selection (pick one)
    p.add_argument("--checkpoint", nargs="+", default=None,
                   help="One or more checkpoint paths to evaluate")
    p.add_argument("--output_dir", default=None,
                   help="Training output dir — evaluates all checkpoint-* inside it")
    # Model
    p.add_argument("--model_name", default=None,
                   help="Base model name (auto-detected for LoRA checkpoints)")
    p.add_argument("--eval_base_model", action="store_true",
                   help="Also evaluate the base model for comparison")
    # Dataset
    p.add_argument("--dataset", nargs="+", default=["gsm8k"],
                   choices=["gsm8k", "math500", "math"],
                   help="Evaluation datasets")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples per dataset")
    # Generation
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--min_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0=greedy)")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per question (for majority voting)")
    p.add_argument("--thinking", action="store_true",
                   help="Enable Qwen3 thinking mode")
    p.add_argument("--system_prompt", nargs="+",
                   default=["training"],
                   choices=list(SYSTEM_PROMPTS.keys()) + ["all"],
                   help="System prompt(s) to evaluate. Use 'all' to compare all prompts.")
    # vLLM
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--lora_mode", default="native",
                   choices=["native", "merge"],
                   help="LoRA loading strategy: 'native' uses vLLM's built-in LoRA "
                        "support (no merge, exact weights); 'merge' merges LoRA into "
                        "base model first (default: native)")
    p.add_argument("--max_lora_rank", type=int, default=64,
                   help="Max LoRA rank for native mode (must be >= adapter rank)")
    # Output
    p.add_argument("--results_dir", default="eval_results")
    p.add_argument("--save_records", action="store_true",
                   help="Save all per-example records (large files)")
    p.add_argument("--save_records_limit", type=int, default=20,
                   help="Save up to N sample records per benchmark (default: 20)")
    # Wandb
    p.add_argument("--wandb", action="store_true",
                   help="Log results to Weights & Biases")
    p.add_argument("--wandb_project", default="multi-reward-eval")
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    # Resolve checkpoint list
    checkpoints = []
    if args.checkpoint:
        checkpoints = args.checkpoint
    elif args.output_dir:
        found = find_checkpoints(args.output_dir)
        if not found:
            print(f"No checkpoints found in {args.output_dir}")
            return
        checkpoints = [path for _, path in found]
        print(f"Found {len(checkpoints)} checkpoints in {args.output_dir}")
    else:
        if not args.eval_base_model:
            p.error("Provide --checkpoint, --output_dir, or --eval_base_model")

    os.makedirs(args.results_dir, exist_ok=True)

    # Init wandb
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f"eval_vllm_{datetime.now():%Y%m%d_%H%M%S}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Resolve system prompt configs
    prompt_names = args.system_prompt
    if "all" in prompt_names:
        prompt_names = list(SYSTEM_PROMPTS.keys())
    prompt_configs = [(name, SYSTEM_PROMPTS[name]) for name in prompt_names]
    print(f"System prompts to evaluate: {[p[0] for p in prompt_configs]}")

    datasets_cache = {}
    all_run_results = []

    # Evaluate base model if requested (single engine, all prompts)
    if args.eval_base_model:
        base_model_name = args.model_name
        if not base_model_name and checkpoints:
            first_ckpt = checkpoints[0]
            if is_lora_checkpoint(first_ckpt):
                base_model_name = resolve_base_model(first_ckpt)
            else:
                base_model_name = first_ckpt
        if not base_model_name:
            print("ERROR: Cannot determine base model. Use --model_name.")
        else:
            print(f"\n{'=' * 60}")
            print(f"Evaluating BASE model: {base_model_name}")
            print(f"{'=' * 60}")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            llm, _ = create_vllm_engine(
                model_path=base_model_name,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            try:
                for prompt_name, system_prompt in prompt_configs:
                    base_results = {"step": 0, "checkpoint": base_model_name,
                                    "type": "base", "system_prompt": prompt_name}
                    for dataset_name in args.dataset:
                        if dataset_name not in datasets_cache:
                            datasets_cache[dataset_name] = load_eval_dataset(dataset_name)
                        data = datasets_cache[dataset_name]
                        if args.max_samples:
                            data = data[:args.max_samples]

                        print(f"\n  --- {dataset_name.upper()} ({len(data)} samples) "
                              f"[{prompt_name}] ---")
                        result = evaluate_with_vllm_engine(
                            llm=llm, lora_request=None,
                            tokenizer=tokenizer, data=data,
                            system_prompt=system_prompt,
                            max_new_tokens=args.max_new_tokens,
                            min_new_tokens=args.min_new_tokens,
                            temperature=args.temperature,
                            n_samples=args.n_samples,
                            thinking=args.thinking,
                        )
                        base_results[dataset_name] = result
                        print(f"  {dataset_name.upper()}: pass@1={result['pass@1']:.4f} "
                              f"({result['correct_pass1']}/{result['total']})")
                    all_run_results.append(base_results)
            finally:
                del llm
                torch.cuda.empty_cache()

    # Evaluate each checkpoint (single engine per checkpoint, all prompts)
    for ckpt_path in checkpoints:
        try:
            results_list = eval_single_checkpoint(
                ckpt_path, args, datasets_cache, prompt_configs)
            all_run_results.extend(results_list)

            # Log to wandb
            if args.wandb:
                import wandb
                for results in results_list:
                    pn = results.get("system_prompt", "")
                    for ds in args.dataset:
                        if ds in results:
                            wandb.log({
                                f"eval/{pn}/{ds}/pass@1": results[ds]["pass@1"],
                                f"eval/{pn}/{ds}/majority@{args.n_samples}":
                                    results[ds][f"majority@{args.n_samples}"],
                                f"eval/{pn}/{ds}/generation_time_s":
                                    results[ds]["generation_time_s"],
                            }, step=results["step"])
        except Exception as e:
            print(f"\nERROR evaluating {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    for ds in args.dataset:
        print(f"\n{ds.upper()}:")
        print(f"  {'Model':>30s}  {'prompt':>10s}  {'pass@1':>8s}  {'majority':>8s}  {'time':>6s}")
        print(f"  {'-' * 30}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 6}")
        for r in all_run_results:
            if ds in r:
                label = f"{r['type']}:step-{r['step']}" if r['type'] != 'base' else 'base'
                pn = r.get("system_prompt", "?")
                p1 = r[ds]["pass@1"]
                mv = r[ds][f"majority@{args.n_samples}"]
                t = r[ds]["generation_time_s"]
                print(f"  {label:>30s}  {pn:>10s}  {p1:8.4f}  {mv:8.4f}  {t:5.0f}s")

    # ── Save results ──
    for r in all_run_results:
        # Trim records for saving
        save_r = dict(r)
        for ds in args.dataset:
            if ds in save_r and "records" in save_r[ds]:
                if args.save_records:
                    pass  # keep all
                elif args.save_records_limit > 0:
                    save_r[ds]["records"] = save_r[ds]["records"][:args.save_records_limit]
                else:
                    del save_r[ds]["records"]

        step = r["step"]
        ckpt_type = r["type"]
        pn = r.get("system_prompt", "unknown")
        datasets_str = "_".join(args.dataset)

        # Extract experiment name from checkpoint path
        ckpt_path = r.get("checkpoint", "")
        # e.g. "outputs/multi_reward_lora_local_klcov005/checkpoint-500" -> "multi_reward_lora_local_klcov005"
        parts = ckpt_path.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1].startswith("checkpoint-"):
            experiment_name = parts[-2]
        else:
            experiment_name = parts[-1] if parts else "unknown"

        # Add metadata to saved JSON
        lora_mode = r.get("lora_mode", "n/a")
        save_r["metadata"] = {
            "method": "vllm",
            "lora_mode": lora_mode,
            "experiment": experiment_name,
            "checkpoint_type": ckpt_type,
            "step": step,
            "datasets": args.dataset,
            "system_prompt_name": pn,
            "system_prompt_text": SYSTEM_PROMPTS.get(pn, ""),
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "max_model_len": args.max_model_len,
            "timestamp": datetime.now().isoformat(),
        }

        fname = (f"eval_vllm_{experiment_name}_{datasets_str}"
                 f"_step{step}_{pn}_{datetime.now():%Y%m%d_%H%M%S}.json")
        out_path = os.path.join(args.results_dir, fname)
        with open(out_path, "w") as f:
            json.dump(save_r, f, indent=2, default=str)
        print(f"Saved: {out_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
