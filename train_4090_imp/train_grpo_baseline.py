"""
Traditional GRPO baseline with ground-truth rewards for comparison.

Uses TRL's GRPOTrainer directly with supervised correctness rewards:
  - Correctness reward: 1.0 if \boxed{answer} matches ground truth, 0.0 otherwise
  - Format reward: 1.0 if completion contains \boxed{}, 0.0 otherwise

This is the standard supervised GRPO approach (like DeepSeek-R1-Zero).
Compare against the unsupervised Multi-Reward GRPO in train.py.

Usage:
  python train_grpo_baseline.py
  bash scripts/train_grpo_baseline_4090.sh
"""
import os, sys, logging, argparse, inspect

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent
from rewards.format_reward import compute_format_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# REWARD FUNCTIONS (ground-truth supervised)
# ============================================================

def correctness_reward_fn(prompts, completions, answer, **kwargs):
    """Binary correctness reward: 1.0 if extracted answer matches ground truth.

    TRL automatically passes extra dataset columns (like 'answer') as kwargs.
    """
    rewards = []
    for comp, gt in zip(completions, answer):
        if isinstance(comp, list):
            text = comp[0]["content"] if comp else ""
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        pred = extract_boxed_answer(text)
        if pred is not None and answers_equivalent(pred, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_fn(prompts, completions, **kwargs):
    """Format reward: 1.0 if completion contains \\boxed{}."""
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            text = comp[0]["content"] if comp else ""
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        rewards.append(compute_format_reward(text))
    return rewards


# ============================================================
# DATASET (includes ground truth answer for supervised reward)
# ============================================================

def prepare_dataset(tokenizer, dataset_name, split="train", max_samples=None):
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        question_key = "question"
    elif dataset_name == "math":
        try:
            ds = load_dataset("lighteval/MATH", split=split, trust_remote_code=True)
        except Exception:
            ds = load_dataset("lighteval/MATH", "all", split=split, trust_remote_code=True)
        question_key = "problem"
    elif dataset_name == "numina":
        ds = load_dataset("AI-MO/NuminaMath-CoT", split=split, trust_remote_code=True)
        question_key = "problem"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    records = []
    for ex in ds:
        q = ex.get(question_key, ex.get("question", ""))
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        # Extract ground truth answer
        if dataset_name == "gsm8k":
            gt = ex["answer"].split("####")[-1].strip()
        elif dataset_name == "math":
            gt = extract_boxed_answer(ex.get("solution", "")) or ""
        elif dataset_name == "numina":
            gt = extract_boxed_answer(ex.get("solution", "")) or ""
        else:
            gt = ""
        records.append({"prompt": prompt, "answer": gt})
    n_with_answer = sum(1 for r in records if r["answer"])
    logger.info(f"Dataset: {len(records)} samples, {n_with_answer} with ground truth answers")
    return Dataset.from_list(records)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Traditional GRPO Baseline with Ground Truth (TRL)")
    p.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math", "numina"])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--w_correctness", type=float, default=1.0,
                    help="Weight for correctness reward")
    p.add_argument("--w_format", type=float, default=0.1,
                    help="Weight for format reward")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--loss_type", default="grpo", choices=["grpo", "dapo", "dr_grpo", "bnpo"])
    p.add_argument("--beta_kl", type=float, default=0.04)
    p.add_argument("--output_dir", default="./outputs/grpo_baseline_4090")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--run_name", default="grpo_baseline_4090")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info(f"=== Traditional GRPO Baseline (Ground Truth Rewards) ===")
    logger.info(f"Args: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(tokenizer, args.dataset, max_samples=args.max_train_samples)
    logger.info(f"Training samples: {len(train_dataset)}")

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules="all-linear", task_type="CAUSAL_LM", lora_dropout=0.05,
    )

    grpo_kwargs = dict(
        output_dir=args.output_dir, run_name=args.run_name,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature, top_p=1.0,
        beta=args.beta_kl,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        report_to="wandb" if args.use_wandb else "none",
        reward_weights=[args.w_correctness, args.w_format],
    )
    if "loss_type" in inspect.signature(GRPOConfig.__init__).parameters:
        grpo_kwargs["loss_type"] = args.loss_type

    training_args = GRPOConfig(**grpo_kwargs)

    # Standard TRL GRPOTrainer — no custom subclass
    # Reward functions: correctness (ground truth) + format
    trainer = GRPOTrainer(
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[correctness_reward_fn, format_reward_fn],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting traditional GRPO training (supervised)...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
