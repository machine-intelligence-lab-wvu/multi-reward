"""
Async evaluation during training — watches for new checkpoints and evaluates them.

Runs in a separate process/GPU while training continues. No training pause needed.
Results saved as JSON and optionally logged to wandb.

Usage:
  # Watch a training output dir, eval each new checkpoint on MATH500:
  python evaluation/eval_during_training.py \
      --output_dir outputs/open_r1_multi_reward_lora \
      --dataset math500 \
      --max_samples 200 \
      --poll_interval 60x

  # With thinking mode:
  python evaluation/eval_during_training.py \
      --output_dir outputs/open_r1_multi_reward_thinking \
      --dataset math500 --thinking --max_new_tokens 4096

  # Eval on multiple benchmarks:
  python evaluation/eval_during_training.py \
      --output_dir outputs/open_r1_multi_reward_lora \
      --dataset math500 gsm8k

  # One-shot: eval all existing checkpoints then exit:
  python evaluation/eval_during_training.py \
      --output_dir outputs/open_r1_multi_reward_lora \
      --dataset math500 --no_watch
"""

import argparse
import glob
import json
import os
import sys
import time
import re
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_4090_imp"))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent
from evaluation.eval_checkpoint import (
    load_eval_dataset, evaluate_model, is_lora_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_checkpoints(output_dir):
    """Find all checkpoint directories, sorted by step number."""
    patterns = [
        os.path.join(output_dir, "checkpoint-*"),
        os.path.join(output_dir, "*/checkpoint-*"),
    ]
    checkpoints = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                match = re.search(r"checkpoint-(\d+)", path)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def detect_model_name(checkpoint_path):
    """Try to detect the base model name from the checkpoint config."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", None)

    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config.get("_name_or_path", None)

    return None


def load_model_for_eval(checkpoint_path, model_name):
    """Load model from checkpoint (auto-detects LoRA vs full FT)."""
    is_lora = is_lora_checkpoint(checkpoint_path)

    if is_lora:
        logger.info(f"  Loading LoRA checkpoint: {checkpoint_path}")
        base_name = model_name or detect_model_name(checkpoint_path)
        if not base_name:
            raise ValueError(
                f"Cannot detect base model for LoRA checkpoint. Use --model_name.")
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(
            base_name, trust_remote_code=True)
    else:
        logger.info(f"  Loading full FT checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def eval_one_checkpoint(checkpoint_path, model_name, datasets, max_samples,
                        max_new_tokens, temperature, n_samples, thinking,
                        results_dir):
    """Evaluate one checkpoint on all specified datasets."""
    step_match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    step = int(step_match.group(1)) if step_match else 0

    model, tokenizer = load_model_for_eval(checkpoint_path, model_name)

    all_results = {"step": step, "checkpoint": checkpoint_path}

    for dataset_name in datasets:
        logger.info(f"  Evaluating step {step} on {dataset_name.upper()}...")
        data = load_eval_dataset(dataset_name)
        if max_samples:
            data = data[:max_samples]

        result = evaluate_model(
            model, tokenizer, data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_samples=n_samples,
            label=f"step-{step}",
            thinking=thinking,
        )
        all_results[dataset_name] = {
            "pass@1": result["pass@1"],
            f"majority@{n_samples}": result[f"majority@{n_samples}"],
            "correct": result["correct_pass1"],
            "total": result["total"],
        }
        logger.info(f"  Step {step} | {dataset_name.upper()}: "
                    f"pass@1={result['pass@1']:.4f} "
                    f"({result['correct_pass1']}/{result['total']})")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"eval_step_{step}.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Log to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            for dataset_name in datasets:
                wandb.log({
                    f"eval/{dataset_name}/pass@1": all_results[dataset_name]["pass@1"],
                    f"eval/{dataset_name}/majority@{n_samples}":
                        all_results[dataset_name][f"majority@{n_samples}"],
                }, step=step)
    except (ImportError, Exception):
        pass

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

    return all_results


def main():
    p = argparse.ArgumentParser(description="Async evaluation during training")
    p.add_argument("--output_dir", required=True,
                   help="Training output directory to watch for checkpoints")
    p.add_argument("--model_name", default=None,
                   help="Base model name (auto-detected for LoRA checkpoints)")
    p.add_argument("--dataset", nargs="+", default=["math500"],
                   help="Evaluation datasets: gsm8k, math500, math")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit eval samples per dataset")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0=greedy)")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per question (for majority voting)")
    p.add_argument("--thinking", action="store_true",
                   help="Enable Qwen3 thinking mode")
    p.add_argument("--poll_interval", type=int, default=60,
                   help="Seconds between checking for new checkpoints")
    p.add_argument("--no_watch", action="store_true",
                   help="Eval all existing checkpoints and exit (no polling)")
    p.add_argument("--results_dir", default=None,
                   help="Where to save results (default: output_dir/eval_results)")
    p.add_argument("--wandb_project", default=None,
                   help="Wandb project for eval logging (separate from training)")
    p.add_argument("--wandb_run_name", default=None,
                   help="Wandb run name for eval")
    args = p.parse_args()

    results_dir = args.results_dir or os.path.join(args.output_dir, "eval_results")

    # Init wandb for eval (separate run from training)
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"eval_{os.path.basename(args.output_dir)}",
                config=vars(args),
            )
        except Exception as e:
            logger.warning(f"Failed to init wandb: {e}")

    # Track which steps we've already evaluated
    evaluated_steps = set()
    existing_results = glob.glob(os.path.join(results_dir, "eval_step_*.json"))
    for r in existing_results:
        match = re.search(r"eval_step_(\d+)\.json", r)
        if match:
            evaluated_steps.add(int(match.group(1)))
    if evaluated_steps:
        logger.info(f"Already evaluated steps: {sorted(evaluated_steps)}")

    logger.info(f"Watching: {args.output_dir}")
    logger.info(f"Datasets: {', '.join(args.dataset)}")
    if args.max_samples:
        logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Thinking mode: {args.thinking}")

    while True:
        checkpoints = find_checkpoints(args.output_dir)
        new_checkpoints = [(step, path) for step, path in checkpoints
                          if step not in evaluated_steps]

        if new_checkpoints:
            for step, path in new_checkpoints:
                logger.info(f"\n{'='*60}")
                logger.info(f"New checkpoint: step {step}")
                logger.info(f"{'='*60}")

                try:
                    results = eval_one_checkpoint(
                        path, args.model_name, args.dataset,
                        args.max_samples, args.max_new_tokens,
                        args.temperature, args.n_samples,
                        args.thinking, results_dir,
                    )
                    evaluated_steps.add(step)
                except Exception as e:
                    logger.error(f"Failed to evaluate step {step}: {e}")
                    import traceback
                    traceback.print_exc()

        if args.no_watch:
            break

        logger.info(f"Waiting {args.poll_interval}s for new checkpoints...")
        time.sleep(args.poll_interval)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")

    all_results = []
    for result_file in sorted(glob.glob(os.path.join(results_dir, "eval_step_*.json"))):
        with open(result_file) as f:
            all_results.append(json.load(f))

    if all_results:
        for ds in args.dataset:
            logger.info(f"\n{ds.upper()}:")
            logger.info(f"  {'Step':>6s}  {'pass@1':>8s}  {'correct':>8s}")
            logger.info(f"  {'-'*6}  {'-'*8}  {'-'*8}")
            for r in all_results:
                if ds in r:
                    logger.info(f"  {r['step']:6d}  {r[ds]['pass@1']:8.4f}  "
                               f"{r[ds]['correct']:>4d}/{r[ds]['total']}")


if __name__ == "__main__":
    main()
