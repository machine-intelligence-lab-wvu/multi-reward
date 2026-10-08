"""
Evaluate a trained model (LoRA, full fine-tuned, or base) on GSM8K, MATH500, or full MATH test set.

Supports:
  - LoRA checkpoints (auto-detected by adapter_config.json)
  - Full fine-tuned models (merged weights, no adapter)
  - Base models (no training)
  - Thinking mode (Qwen3 with --thinking)

Usage:
  # LoRA checkpoint on GSM8K (default)
  python evaluation/eval_checkpoint.py --checkpoint outputs/run_0/checkpoint-400

  # Full fine-tuned model on MATH500
  python evaluation/eval_checkpoint.py --checkpoint outputs/full_ft/final --dataset math500

  # Evaluate on full MATH test set
  python evaluation/eval_checkpoint.py --checkpoint outputs/run_0/checkpoint-400 --dataset math

  # Evaluate base model only
  python evaluation/eval_checkpoint.py --base_only --dataset gsm8k

  # Quick test (first 50 samples)
  python evaluation/eval_checkpoint.py --checkpoint outputs/run_0/checkpoint-400 --max_samples 50

  # Majority vote with multiple samples
  python evaluation/eval_checkpoint.py --checkpoint outputs/run_0/checkpoint-400 --n_samples 16 --temperature 0.6

  # Thinking mode (Qwen3) — enables <think> in generation
  python evaluation/eval_checkpoint.py --checkpoint outputs/thinking/checkpoint-100 --thinking

  # Force LoRA or full mode (skips auto-detection)
  python evaluation/eval_checkpoint.py --checkpoint outputs/ckpt --mode lora
  python evaluation/eval_checkpoint.py --checkpoint outputs/ckpt --mode full
"""
import argparse, json, os, sys, re, random
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_4090_imp"))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent


# ════════════════════════════════════════════════════════════
# DATASET LOADERS
# ════════════════════════════════════════════════════════════


system_prompt = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box."




def load_gsm8k_test():
    """Load GSM8K test set. Returns list of {question, answer}."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"question": x["question"],
             "answer": x["answer"].split("####")[-1].strip()}
            for x in ds]


def load_math500():
    """Load MATH-500 test set from nlile/hendrycks-MATH-benchmark (500 problems)."""
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
    print(f"Loaded MATH-500: {len(data)} problems")
    return data


def load_math_test():
    """Load full MATH test set from hendrycks benchmark."""
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
            print(f"Loaded MATH test from {name}: {len(data)} problems")
            return data
        except Exception as e:
            continue

    raise RuntimeError("Could not load MATH test set.")


def load_eval_dataset(dataset_name):
    """Load evaluation dataset by name."""
    if dataset_name == "gsm8k":
        return load_gsm8k_test()
    elif dataset_name == "math500":
        return load_math500()
    elif dataset_name == "math":
        return load_math_test()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: gsm8k, math500, math")


# ════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════

def build_prompt(tokenizer, question, thinking=False):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate_model(model, tokenizer, data, max_new_tokens=1024, temperature=0.0,
                   n_samples=1, label="model", thinking=False):
    """Generate and evaluate on test data.

    Args:
        thinking: if True, enables Qwen3 thinking mode (<think>...</think>).
                  Answer is extracted from after </think> tag.
    """
    correct_pass1 = 0
    correct_majority = 0
    total = len(data)
    results = []

    for i, item in enumerate(data):
        prompt = build_prompt(tokenizer, item["question"], thinking=thinking)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        outputs = model.generate(**gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]

        # In thinking mode, extract answer from after </think>
        if thinking:
            answer_texts = []
            for c in completions:
                idx = c.rfind("</think>")
                if idx >= 0:
                    answer_texts.append(c[idx + len("</think>"):])
                else:
                    answer_texts.append(c)
        else:
            answer_texts = completions

        # Extract answers from \boxed{}
        extracted = [extract_boxed_answer(t) for t in answer_texts]
        gt = item["answer"]

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

        results.append({
            "question": item["question"][:100],
            "ground_truth": gt,
            "predicted": pred,
            "correct": is_correct,
            "completion_preview": completions[0][:300],
        })

        if (i + 1) % 50 == 0 or i == total - 1:
            p1 = correct_pass1 / (i + 1)
            mv = correct_majority / (i + 1)
            print(f"  [{label}] {i+1}/{total}  pass@1={p1:.4f}  majority@{n_samples}={mv:.4f}")

    return {
        "pass@1": correct_pass1 / total,
        f"majority@{n_samples}": correct_majority / total,
        "total": total,
        "correct_pass1": correct_pass1,
        "correct_majority": correct_majority,
        "samples": results[:10],
    }


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def is_hf_repo(path):
    """Check if a path looks like a HuggingFace repo ID (e.g., 'user/repo')."""
    if path is None:
        return False
    # HF repo IDs have format "user/repo" or "org/repo", no path separators beyond one /
    # Local paths have . or / at the start, or multiple / separators
    if os.path.exists(path):
        return False  # local path takes priority
    parts = path.split("/")
    return len(parts) == 2 and not path.startswith(".") and not path.startswith("/")


def is_lora_checkpoint(path):
    """Auto-detect if a checkpoint is LoRA or full fine-tuned.

    LoRA checkpoints have adapter_config.json.
    Full fine-tuned checkpoints have model weights directly.
    Works for both local paths and HuggingFace repo IDs.
    """
    if path is None:
        return False
    # Local path
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, "adapter_config.json"))
    # HF repo — try to check via huggingface_hub
    if is_hf_repo(path):
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(path)
            return "adapter_config.json" in files
        except Exception:
            return False
    return False


def main():
    p = argparse.ArgumentParser(description="Evaluate trained models on math benchmarks")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B",
                   help="Base model name (for LoRA) or ignored (for full FT)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint dir OR HuggingFace repo ID (e.g., user/repo)")
    p.add_argument("--revision", type=str, default=None,
                   help="HF repo branch/revision (e.g., step-50, exp1-step-100)")
    p.add_argument("--mode", type=str, default="auto",
                   choices=["auto", "lora", "full"],
                   help="Checkpoint type: auto-detect, force LoRA, or force full FT")
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math500", "math"],
                   help="Evaluation dataset (default: gsm8k)")
    p.add_argument("--base_only", action="store_true",
                   help="Evaluate the base model without any adapter/fine-tuning")
    p.add_argument("--thinking", action="store_true",
                   help="Enable Qwen3 thinking mode (generate with <think> tags)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of test samples (for quick testing)")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Number of samples per question (for majority voting)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0=greedy)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--wandb", action="store_true",
                   help="Log results to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="multi-reward-eval",
                   help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="W&B run name (auto-generated if not set)")
    args = p.parse_args()

    # Initialize wandb
    if args.wandb:
        ckpt_label = os.path.basename(args.checkpoint.rstrip("/")) if args.checkpoint else "base"
        model_short = args.model_name.split("/")[-1]
        run_name = args.wandb_run_name or f"eval_{args.dataset}_{model_short}_{ckpt_label}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "base_only": args.base_only,
                "thinking": args.thinking,
                "n_samples": args.n_samples,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "max_samples": args.max_samples,
            },
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print(f"Loading {args.dataset.upper()} test set...")
    data = load_eval_dataset(args.dataset)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Evaluating on {len(data)} samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Determine checkpoint type
    if args.checkpoint:
        if args.mode == "auto":
            is_lora = is_lora_checkpoint(args.checkpoint)
            ckpt_type = "lora" if is_lora else "full"
            print(f"Auto-detected checkpoint type: {ckpt_type.upper()}")
        else:
            ckpt_type = args.mode
            print(f"Forced checkpoint type: {ckpt_type.upper()}")
    else:
        ckpt_type = None

    thinking_str = " (thinking mode)" if args.thinking else ""

    # Evaluate base model
    if args.base_only or args.checkpoint is None:
        print(f"\n{'='*60}")
        print(f"Evaluating BASE model: {args.model_name} on {args.dataset.upper()}{thinking_str}")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model.eval()
        all_results["base"] = evaluate_model(
            model, tokenizer, data,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_samples=args.n_samples,
            label="base",
            thinking=args.thinking,
        )
        print(f"\nBase model: pass@1={all_results['base']['pass@1']:.4f}  "
              f"majority@{args.n_samples}={all_results['base'][f'majority@{args.n_samples}']:.4f}")

        if args.checkpoint:
            del model
            torch.cuda.empty_cache()

    # Evaluate checkpoint
    if args.checkpoint:
        print(f"\n{'='*60}")
        print(f"Evaluating {ckpt_type.upper()} CHECKPOINT: {args.checkpoint} on {args.dataset.upper()}{thinking_str}")
        print(f"{'='*60}")

        revision = getattr(args, "revision", None)
        rev_str = f" (revision: {revision})" if revision else ""

        if ckpt_type == "lora":
            # LoRA checkpoint: load base model + adapter
            print(f"Loading base model: {args.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            print(f"Loading LoRA adapter: {args.checkpoint}{rev_str}")
            model = PeftModel.from_pretrained(
                model, args.checkpoint, revision=revision)
        else:
            # Full fine-tuned checkpoint: load directly (local path or HF repo)
            print(f"Loading full fine-tuned model: {args.checkpoint}{rev_str}")
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, revision=revision)
            # Try loading tokenizer from checkpoint, fall back to model_name
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.checkpoint, trust_remote_code=True, revision=revision)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        all_results["checkpoint"] = evaluate_model(
            model, tokenizer, data,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            n_samples=args.n_samples,
            label="ckpt",
            thinking=args.thinking,
        )
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
        print(f"\nCheckpoint ({ckpt_name}): pass@1={all_results['checkpoint']['pass@1']:.4f}  "
              f"majority@{args.n_samples}={all_results['checkpoint'][f'majority@{args.n_samples}']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.dataset.upper()})")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name:15s}  pass@1={res['pass@1']:.4f}  "
              f"majority@{args.n_samples}={res[f'majority@{args.n_samples}']:.4f}  "
              f"({res['correct_pass1']}/{res['total']})")

    # Log to wandb
    if args.wandb:
        for name, res in all_results.items():
            wandb.log({
                f"{name}/{args.dataset}/pass@1": res["pass@1"],
                f"{name}/{args.dataset}/majority@{args.n_samples}": res[f"majority@{args.n_samples}"],
                f"{name}/{args.dataset}/correct": res["correct_pass1"],
                f"{name}/{args.dataset}/total": res["total"],
            })
        # Log a summary table
        table = wandb.Table(columns=["model", "dataset", "pass@1", f"majority@{args.n_samples}", "correct", "total"])
        for name, res in all_results.items():
            table.add_data(name, args.dataset, res["pass@1"], res[f"majority@{args.n_samples}"],
                           res["correct_pass1"], res["total"])
        wandb.log({f"eval_summary_{args.dataset}": table})

        # Log sample predictions
        if any("samples" in res for res in all_results.values()):
            for name, res in all_results.items():
                if "samples" in res:
                    sample_table = wandb.Table(
                        columns=["question", "ground_truth", "predicted", "correct", "completion_preview"])
                    for s in res["samples"]:
                        sample_table.add_data(
                            s["question"], s["ground_truth"], s["predicted"],
                            s["correct"], s["completion_preview"])
                    wandb.log({f"{name}_{args.dataset}_samples": sample_table})

    # Save — filename includes dataset name
    if args.checkpoint:
        rel_path = args.checkpoint.rstrip("/")
        safe_name = rel_path.replace("/", "__").replace(".", "").strip("_")
        output_file = os.path.join(args.output_dir, f"eval_{args.dataset}_{safe_name}.json")
    else:
        output_file = os.path.join(args.output_dir, f"eval_{args.dataset}_base_model.json")

    save_results = {"dataset": args.dataset, "model": args.model_name}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()}
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
