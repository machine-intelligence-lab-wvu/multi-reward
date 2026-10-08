"""
In-training accuracy evaluation callback.

Evaluates pass@1 accuracy on a small sample of a benchmark dataset
every N training steps. Logs results to wandb alongside training metrics.

WARNING: This pauses training for ~2-5 minutes per evaluation.
For async eval without pausing, use evaluation/eval_during_training.py instead.

Usage:
  Add to recipe YAML:
    eval_accuracy_steps: 20        # eval every 20 steps
    eval_accuracy_dataset: math500 # dataset to eval on
    eval_accuracy_samples: 50      # samples per eval (keep small for speed)
"""

import logging
import os
import sys

import torch
from transformers import TrainerCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train_4090_imp"))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent
from evaluation.eval_checkpoint import load_eval_dataset

logger = logging.getLogger(__name__)


class AccuracyEvalCallback(TrainerCallback):
    """Evaluate pass@1 accuracy during training every N steps.

    Uses greedy decoding on a small sample of the eval dataset.
    Runs on the main process only. Pauses training during eval.
    """

    def __init__(self, tokenizer, eval_every_n_steps=20,
                 dataset_name="math500", max_samples=50,
                 max_new_tokens=2048, thinking=False):
        self.tokenizer = tokenizer
        self.eval_every_n_steps = eval_every_n_steps
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self._eval_data = None
        self._last_eval_step = -1

    def _load_data(self):
        """Load eval data (lazily, once)."""
        if self._eval_data is None:
            data = load_eval_dataset(self.dataset_name)
            if self.max_samples:
                data = data[:self.max_samples]
            self._eval_data = data
            logger.info(f"AccuracyEval: loaded {len(data)} samples from {self.dataset_name}")

    def _build_prompt(self, question):
        msgs = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _evaluate(self, model):
        """Run greedy eval on the sample set."""
        self._load_data()
        model.eval()
        correct = 0
        total = len(self._eval_data)

        for item in self._eval_data:
            prompt = self._build_prompt(item["question"])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            completion = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

            # Extract answer (handle thinking mode)
            if self.thinking:
                idx = completion.rfind("</think>")
                answer_text = completion[idx + len("</think>"):] if idx >= 0 else completion
            else:
                answer_text = completion

            pred = extract_boxed_answer(answer_text)
            gt = item["answer"]
            if pred and answers_equivalent(pred, gt):
                correct += 1

        model.train()
        return correct / total if total > 0 else 0.0, correct, total

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run eval every N steps on main process."""
        step = state.global_step
        if step <= 0 or step % self.eval_every_n_steps != 0:
            return
        if step == self._last_eval_step:
            return

        # Only eval on main process
        from accelerate import PartialState
        if not PartialState().is_main_process:
            return

        self._last_eval_step = step

        logger.info(f"\n{'='*50}")
        logger.info(f"AccuracyEval: step {step} on {self.dataset_name} ({self.max_samples} samples)...")
        logger.info(f"{'='*50}")

        try:
            accuracy, correct, total = self._evaluate(model)
            logger.info(f"AccuracyEval: step {step} | {self.dataset_name} "
                        f"pass@1={accuracy:.4f} ({correct}/{total})")

            # Log to wandb
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        f"eval/{self.dataset_name}/pass@1": accuracy,
                        f"eval/{self.dataset_name}/correct": correct,
                        f"eval/{self.dataset_name}/total": total,
                    }, step=step)
            except (ImportError, Exception):
                pass

            # Log via trainer metrics
            mode = "train"
            if hasattr(state, "_metrics") and mode in state._metrics:
                state._metrics[mode][f"eval/{self.dataset_name}/pass@1"] = accuracy

        except Exception as e:
            logger.warning(f"AccuracyEval failed at step {step}: {e}")
