"""
IntuitorTrainer — Implements the INTUITOR method from
"Learning to Reason without External Rewards" (Zhao et al., 2025).

Uses self-certainty (KL(U||p)) as the *sole* reward signal in GRPO,
replacing all external rewards. This is a clean baseline for comparison
with the multi-reward approach.

Key design choices matching the paper:
  - Online self-certainty: computed from the current policy (not frozen)
  - Standard GRPO loss (no KL-Cov, no entropy bonus, no GDPO)
  - Z-score normalization of self-certainty within prompt groups (Eq. 6)
  - Short-completion and no-reasoning penalties to prevent collapse
"""

import logging
from typing import Any, Union

import torch
import torch.distributed as dist
from trl import GRPOTrainer

from open_r1_imp.utils import (
    compute_self_certainty_per_token,
    aggregate_self_certainty,
    penalize_short_completions,
    penalize_no_reasoning,
    build_reasoning_mask,
    THINK_END_ID,
)
from callbacks.klcov import compute_token_covariances, select_high_covariance_tokens

logger = logging.getLogger(__name__)


def _zscore_normalize(rewards: torch.Tensor, group_idx: torch.Tensor,
                      eps: float = 1e-4) -> torch.Tensor:
    """Z-score normalize rewards within each prompt group (Eq. 6 in paper).

    A_i = (u_i - mean(u)) / std(u)  for each group of G completions.
    Uses Bessel's correction (correction=1) and eps=1e-4 to match
    the paper's implementation.
    """
    normalized = torch.zeros_like(rewards)
    for gid in group_idx.unique():
        mask = group_idx == gid
        group = rewards[mask]
        if group.numel() <= 1:
            continue
        mu = group.mean()
        sigma = group.std(correction=1)
        normalized[mask] = (group - mu) / (sigma + eps)
    return normalized


class IntuitorTrainer(GRPOTrainer):
    """GRPOTrainer with self-certainty as sole reward (INTUITOR method).

    Overrides _generate_and_score_completions to replace TRL's external
    reward-based advantages with self-certainty-derived advantages.
    No _compute_loss override needed — standard GRPO loss is used.
    """

    def __init__(self, *args, **kwargs):
        training_args = kwargs.get("args", args[2] if len(args) > 2 else None)

        self.thinking = getattr(training_args, "thinking", False)
        self.min_completion_tokens = getattr(training_args, "min_completion_tokens", 50)
        self.sc_batch_size = getattr(training_args, "sc_batch_size", 4)

        # KL-Cov parameters
        self.beta_cov = getattr(training_args, "beta_cov", 0.0)
        self.top_k_fraction = getattr(training_args, "top_k_fraction", 0.02)

        super().__init__(*args, **kwargs)

        logger.info(
            f"IntuitorTrainer: thinking={self.thinking}, "
            f"min_tokens={self.min_completion_tokens}, "
            f"sc_batch_size={self.sc_batch_size}, "
            f"beta_cov={self.beta_cov}, top_k_fraction={self.top_k_fraction}"
        )

    # ════════════════════════════════════════════════════════════
    # SELF-CERTAINTY COMPUTATION (online, from current policy)
    # ════════════════════════════════════════════════════════════

    def _compute_self_certainty(
        self, model, input_ids, attention_mask, completion_ids, completion_mask,
        logits_to_keep,
    ) -> torch.Tensor:
        """Compute per-sample self-certainty scores from current policy.

        Online computation (not frozen reference) — critical for preventing
        reward hacking (Section 5.4, Figure 7 of the paper).

        Returns:
            (B,) self-certainty scores
        """
        batch_size = self.sc_batch_size
        all_sc_per_token = []

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for start in range(0, input_ids.size(0), batch_size):
                end = min(start + batch_size, input_ids.size(0))
                chunk_ids = input_ids[start:end]
                chunk_attn = attention_mask[start:end]

                logits = model(input_ids=chunk_ids, attention_mask=chunk_attn).logits
                # Raw logits (no temperature) — matches paper's definition (Eq. 4)
                logits = logits[:, -logits_to_keep:, :]
                sc = compute_self_certainty_per_token(logits)
                all_sc_per_token.append(sc)
                del logits

        if was_training:
            model.train()

        per_token_sc = torch.cat(all_sc_per_token, dim=0)  # (B, S)

        # Mask: reasoning tokens only (thinking mode) or full completion
        if self.thinking:
            reasoning_mask = build_reasoning_mask(completion_ids, completion_mask)
            sc_mask = reasoning_mask
        else:
            sc_mask = completion_mask

        return aggregate_self_certainty(per_token_sc, sc_mask)

    # ════════════════════════════════════════════════════════════
    # OVERRIDE: _generate_and_score_completions
    # ════════════════════════════════════════════════════════════

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions and replace rewards with self-certainty advantages.

        1. Call parent to generate completions and compute external rewards (logged only)
        2. Compute online self-certainty from current policy
        3. Apply collapse guards (short completion, no reasoning)
        4. Z-score normalize within prompt groups (Eq. 6)
        5. Replace TRL's advantages with self-certainty advantages
        """
        # Parent generates completions and computes external rewards
        result = super()._generate_and_score_completions(inputs)

        device = result["advantages"].device
        mode = "train" if self.model.training else "eval"

        # ── Compute online self-certainty ──
        prompt_ids = result["prompt_ids"]
        prompt_mask = result["prompt_mask"]
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        self_certainty = self._compute_self_certainty(
            self.model, input_ids, attention_mask,
            completion_ids, completion_mask, logits_to_keep,
        )

        # ── Build group indices ──
        B = self_certainty.shape[0]
        G = self.num_generations
        group_idx = torch.arange(B, device=device) // G

        # ── Collapse guards ──
        self_certainty = penalize_short_completions(
            self_certainty, completion_mask, group_idx, self.min_completion_tokens)
        if self.thinking:
            reasoning_mask = build_reasoning_mask(completion_ids, completion_mask)
            self_certainty = penalize_no_reasoning(
                self_certainty, reasoning_mask, group_idx)

        # ── Gather self-certainty across all GPUs before normalization ──
        # Paper gathers globally so all ranks see the same mean/std per group
        if dist.is_initialized():
            all_sc = self.accelerator.gather(self_certainty.detach())
            B_global = all_sc.shape[0]
            G = self.num_generations
            # Reshape to (num_prompts, G), compute per-group mean/std
            all_sc_grouped = all_sc.view(-1, G)
            mean_per_group = all_sc_grouped.mean(dim=1)   # (num_prompts,)
            std_per_group = all_sc_grouped.std(dim=1, correction=1)  # Bessel's correction
            # Expand back and slice to local process
            mean_expanded = mean_per_group.repeat_interleave(G)
            std_expanded = std_per_group.repeat_interleave(G)
            process_slice = self.accelerator.process_index * B + torch.arange(B, device=device)
            local_mean = mean_expanded[process_slice]
            local_std = std_expanded[process_slice]
            sc_advantages = (self_certainty.detach() - local_mean) / (local_std + 1e-4)
        else:
            # ── Fallback: single-GPU z-score normalize (Eq. 6) ──
            sc_advantages = _zscore_normalize(self_certainty.detach(), group_idx)

        # ── Log external rewards for comparison (don't affect training) ──
        self._metrics[mode]["rewards/external_advantages_mean"].append(
            result["advantages"].mean().item())

        # ── Replace advantages with self-certainty ──
        result["advantages"] = sc_advantages

        # ── Log metrics ──
        self._metrics[mode]["rewards/self_certainty"].append(
            self_certainty.mean().item())
        self._metrics[mode]["advantages/ch_sc_std"].append(
            sc_advantages.std(correction=0).item() if sc_advantages.numel() > 1 else 0.0)

        if self.thinking:
            reasoning_mask = build_reasoning_mask(completion_ids, completion_mask)
            self._metrics[mode]["reasoning/frac_with_think"].append(
                (reasoning_mask.sum(dim=-1) > 0).float().mean().item())
            self._metrics[mode]["reasoning/avg_reasoning_tokens"].append(
                reasoning_mask.sum(dim=-1).float().mean().item())

        return result

    # ════════════════════════════════════════════════════════════
    # OVERRIDE: _compute_loss (KL-Cov extension)
    # ════════════════════════════════════════════════════════════

    def _compute_loss(self, model, inputs):
        """Compute GRPO loss with optional KL-Cov regularization.

        When beta_cov > 0, high-covariance tokens get unclipped loss + KL penalty
        to encourage exploration at decision-critical tokens.
        When beta_cov == 0, falls back to standard GRPO loss.
        """
        if self.beta_cov == 0.0:
            return super()._compute_loss(model, inputs)

        # ── Replicate base computation with KL-Cov ──
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask

        # Forward pass
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep)

        # Importance sampling ratio
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Advantages
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # ── KL-Cov: unclip high-covariance tokens ──
        covariances = compute_token_covariances(
            per_token_logps.detach(), advantages.squeeze(1), mask)
        high_cov_mask = select_high_covariance_tokens(
            covariances, mask, self.top_k_fraction)
        normal_mask = mask * (1.0 - high_cov_mask)

        # Normal tokens: clipped surrogate
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        clipped_loss = -torch.min(per_token_loss1, per_token_loss2)

        # High-cov tokens: unclipped + KL penalty
        unclipped_loss = -coef_1 * advantages
        kl_penalty = self.beta_cov * log_ratio.abs()
        penalized_loss = unclipped_loss + kl_penalty

        per_token_loss = clipped_loss * normal_mask + penalized_loss * high_cov_mask

        # Global KL penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregate loss
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()

        # ── Metrics ──
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            mean_kl = ((per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        n_high = high_cov_mask.sum().item()
        n_total = mask.sum().item()
        self._metrics[mode]["klcov/high_cov_fraction"].append(n_high / max(n_total, 1))
        if n_high > 0:
            self._metrics[mode]["klcov/mean_kl_penalty"].append(
                (kl_penalty * high_cov_mask).sum().item() / n_high)

        return loss
