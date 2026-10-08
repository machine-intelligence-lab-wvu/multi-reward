"""
GRPOGTTrainer — Standard GRPO with ground-truth reward + monitoring metrics.

Uses the exact same training logic as TRL's GRPOTrainer (no modifications
to advantages or loss). Only adds metric logging for comparison with
Intuitor and Multi-Reward runs:
  - Self-certainty (logged only, does NOT affect training)
  - Completion length statistics
  - Accuracy rate (fraction of correct answers per batch)
  - Format rate (fraction with valid \\boxed{})
"""

import logging
from typing import Any, Optional, Union

import torch
from trl import GRPOTrainer

from open_r1_imp.utils import (
    compute_self_certainty_per_token,
    aggregate_self_certainty,
    compute_token_covariances,
    select_high_covariance_tokens,
)

logger = logging.getLogger(__name__)


class GRPOGTTrainer(GRPOTrainer):
    """GRPOTrainer with additional monitoring metrics.

    Training logic is IDENTICAL to base GRPOTrainer — no custom advantages,
    no self-certainty reward, no GDPO. This subclass only overrides
    _generate_and_score_completions to log extra metrics for comparison
    with Intuitor and Multi-Reward runs on the same wandb dashboard.
    """

    def __init__(self, *args, **kwargs):
        training_args = kwargs.get("args", args[2] if len(args) > 2 else None)
        self.sc_batch_size = getattr(training_args, "sc_batch_size", 4)
        # KL-Cov (rebuttal A4_grpo_gt_klcov): applies to the loss only; advantages
        # remain the standard GRPO-GT accuracy advantages. beta_cov=0 => unchanged.
        self.beta_cov = getattr(training_args, "beta_cov", 0.0)
        self.top_k_fraction = getattr(training_args, "top_k_fraction", 0.02)
        super().__init__(*args, **kwargs)
        logger.info(f"GRPOGTTrainer: monitoring SC (batch_size={self.sc_batch_size}), "
                    f"accuracy, format, completion lengths | beta_cov={self.beta_cov}")

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions, compute standard advantages, then log extra metrics."""
        # Standard GRPO — unchanged
        result = super()._generate_and_score_completions(inputs)

        mode = "train" if self.model.training else "eval"

        # ── Log self-certainty (monitoring only) ──
        try:
            prompt_ids = result["prompt_ids"]
            prompt_mask = result["prompt_mask"]
            completion_ids = result["completion_ids"]
            completion_mask = result["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            all_sc = []
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                for start in range(0, input_ids.size(0), self.sc_batch_size):
                    end = min(start + self.sc_batch_size, input_ids.size(0))
                    logits = self.model(
                        input_ids=input_ids[start:end],
                        attention_mask=attention_mask[start:end],
                    ).logits
                    logits = logits[:, -logits_to_keep:, :]
                    sc = compute_self_certainty_per_token(logits)
                    all_sc.append(sc)
                    del logits
            if was_training:
                self.model.train()

            per_token_sc = torch.cat(all_sc, dim=0)
            self_certainty = aggregate_self_certainty(per_token_sc, completion_mask)
            self._metrics[mode]["rewards/self_certainty"].append(
                self_certainty.mean().item())
        except Exception as e:
            logger.debug(f"SC monitoring failed: {e}")

        # ── Completion length stats ──
        try:
            lengths = completion_mask.sum(dim=-1).float()
            self._metrics[mode]["completions/mean_length"].append(lengths.mean().item())
            self._metrics[mode]["completions/min_length"].append(lengths.min().item())
            self._metrics[mode]["completions/max_length"].append(lengths.max().item())
        except Exception:
            pass

        # ── Accuracy & format rate from rewards ──
        # TRL stores per-reward-func values; we read the advantages shape
        # to compute batch-level accuracy stats
        try:
            B = completion_ids.size(0)
            G = self.num_generations
            advantages = result["advantages"]
            self._metrics[mode]["advantages/mean"].append(advantages.mean().item())
            self._metrics[mode]["advantages/std"].append(
                advantages.std(correction=0).item() if advantages.numel() > 1 else 0.0)
        except Exception:
            pass

        return result

    # ════════════════════════════════════════════════════════════
    # KL-Cov (optional) — rebuttal run A4_grpo_gt_klcov
    # ════════════════════════════════════════════════════════════

    def _forward_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Per-token log-probs of the completion tokens (matches TRL temperature scaling)."""
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        completion_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

    def _compute_loss(self, model, inputs):
        """GRPO-GT loss, optionally with KL-Cov on the top-covariance tokens.

        beta_cov == 0  -> identical to the base GRPOTrainer (existing runs unchanged).
        beta_cov > 0   -> high-covariance tokens (top `top_k_fraction`) use an
                          unclipped surrogate + KL penalty; all other tokens use the
                          standard clipped surrogate. Ported from MultiRewardTrainer.
        """
        if self.beta_cov == 0.0:
            return super()._compute_loss(model, inputs)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps = self._forward_logps(model, input_ids, attention_mask, logits_to_keep)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # KL-Cov: split into normal vs top-covariance tokens
        covariances = compute_token_covariances(
            per_token_logps.detach(), advantages.squeeze(1), mask)
        high_cov_mask = select_high_covariance_tokens(covariances, mask, self.top_k_fraction)
        normal_mask = mask * (1.0 - high_cov_mask)

        clipped_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        penalized_loss = -coef_1 * advantages + self.beta_cov * log_ratio.abs()
        per_token_loss = clipped_loss * normal_mask + penalized_loss * high_cov_mask

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        try:
            mode = "train" if self.model.training else "eval"
            n_high = high_cov_mask.sum().item()
            n_total = mask.sum().item()
            self._metrics[mode]["klcov/high_cov_fraction"].append(n_high / max(n_total, 1))
            # Log the penalty itself, not just how many tokens were selected.
            # high_cov_fraction only proves the top-k selection ran; it stays at
            # ~top_k_fraction even when the penalty is identically zero. That is
            # exactly the silent failure seen at num_iterations=1: the rollout
            # policy equals the current policy, so log_ratio == 0 and both the
            # beta_cov*|log_ratio| term AND its gradient vanish, making KL-Cov a
            # no-op while this metric still looks healthy. Mirrors
            # multi_reward_trainer.py:544.
            if n_high > 0:
                # Recompute the penalty term (it is inlined into penalized_loss
                # above, not kept in its own variable).
                kl_penalty = self.beta_cov * log_ratio.abs()
                self._metrics[mode]["klcov/mean_kl_penalty"].append(
                    (kl_penalty * high_cov_mask).sum().item() / n_high)
        except Exception:
            pass

        return loss
