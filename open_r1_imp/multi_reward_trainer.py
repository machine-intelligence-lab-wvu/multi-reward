"""
MultiRewardTrainer — Extends TRL's GRPOTrainer with multi-reward GRPO.

Overrides two key methods:
  1. _generate_and_score_completions() — adds GDPO per-channel normalization
     and self-certainty computation (reasoning-only in thinking mode)
  2. _compute_loss() — adds KL-Cov regularization and entropy bonus

Built on HF Open-R1's GRPOTrainer (from TRL), with self-certainty logic
ported from Intuitor's INTUITORTrainer.
"""

import math
import logging
from typing import Any, Union

import torch
import torch.nn.functional as F
from trl import GRPOTrainer

from open_r1_imp.utils import (
    gdpo_normalize,
    compute_self_certainty_per_token,
    aggregate_self_certainty,
    penalize_short_completions,
    penalize_no_reasoning,
    build_reasoning_mask,
    compute_token_covariances,
    select_high_covariance_tokens,
    THINK_END_ID,
)

logger = logging.getLogger(__name__)


class MultiRewardTrainer(GRPOTrainer):
    """GRPOTrainer with multi-reward GDPO normalization, KL-Cov, and thinking mode.

    Key extensions over base GRPOTrainer:
      - Self-certainty as an additional reward channel
      - GDPO per-channel normalization (cluster+format channel + SC channel)
      - KL-Cov token-level regularization
      - Thinking mode: SC on reasoning tokens only, rewards on answer only
      - Min completion length filter to prevent template collapse
    """

    def __init__(self, *args, **kwargs):
        # Extract multi-reward config before passing to parent
        training_args = kwargs.get("args", args[2] if len(args) > 2 else None)

        self.w_cluster_format = getattr(training_args, "w_cluster_format", 0.5)
        self.w_selfcert = getattr(training_args, "w_selfcert", 0.5)
        self.w_cluster = getattr(training_args, "w_cluster", 0.5)
        self.w_format = getattr(training_args, "w_format", 0.1)
        self.beta_cov = getattr(training_args, "beta_cov", 0.0)
        self.top_k_fraction = getattr(training_args, "top_k_fraction", 0.02)
        self.entropy_bonus = getattr(training_args, "entropy_bonus", 0.0)
        self.thinking = getattr(training_args, "thinking", False)
        self.min_completion_tokens = getattr(training_args, "min_completion_tokens", 50)
        self.grad_corr_enabled = getattr(training_args, "grad_corr_enabled", False)
        self.grad_corr_every_steps = getattr(training_args, "grad_corr_every_steps", 20)
        self.grad_corr_num_tensors = getattr(training_args, "grad_corr_num_tensors", 16)

        super().__init__(*args, **kwargs)

        self._grad_corr_probe_params = None
        self._grad_corr_probe_names = None
        self._grad_corr_failed = False
        self._last_rewards_per_func = None

        # Override TRL's reward_weights with our per-function weights.
        # TRL uses this tensor to combine reward_funcs into result["advantages"];
        # we want w_cluster and w_format from the YAML to be the source of truth.
        self.reward_weights = torch.tensor(
            [self.w_cluster, self.w_format], dtype=torch.float32
        )

        logger.info(
            f"MultiRewardTrainer: w_cf={self.w_cluster_format}, w_sc={self.w_selfcert}, "
            f"w_cluster={self.w_cluster}, w_format={self.w_format}, "
            f"beta_cov={self.beta_cov}, thinking={self.thinking}, "
            f"min_tokens={self.min_completion_tokens}, grad_corr={self.grad_corr_enabled}"
        )

    def _calculate_rewards(self, *args, **kwargs):
        """Capture per-function rewards before TRL combines them.

        TRL's _generate_and_score_completions uses rewards_per_func as a local
        variable and does not include it in the result dict, so we intercept
        it here for the cluster-only gradcorr probe.
        """
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rewards_per_func.detach().clone()
        return rewards_per_func

    def _get_grad_corr_probe_params(self):
        """Select a bounded set of parameter tensors for gradient-correlation probes.

        Using all tensors can be very expensive under ZeRO/FSDP. By default we
        probe the last N trainable tensors, which typically include upper layers
        and heads where task-specific updates concentrate.
        """
        if self._grad_corr_probe_params is not None:
            return self._grad_corr_probe_params

        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable:
            self._grad_corr_probe_params = []
            self._grad_corr_probe_names = []
            return self._grad_corr_probe_params

        n_tensors = int(self.grad_corr_num_tensors)
        if n_tensors > 0 and n_tensors < len(trainable):
            trainable = trainable[-n_tensors:]

        self._grad_corr_probe_names = [n for n, _ in trainable]
        self._grad_corr_probe_params = [p for _, p in trainable]

        logger.info(
            "Grad-corr probe tensors: %d (requested=%s)",
            len(self._grad_corr_probe_params),
            self.grad_corr_num_tensors,
        )
        return self._grad_corr_probe_params

    @staticmethod
    def _grad_stats_from_pairs(grads_a, grads_b):
        """Compute dot products / norms / cosine from two grad tensor lists."""
        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for ga, gb in zip(grads_a, grads_b):
            if ga is None or gb is None:
                continue
            dot += torch.sum(ga * gb).item()
            norm_a_sq += torch.sum(ga * ga).item()
            norm_b_sq += torch.sum(gb * gb).item()

        norm_a = math.sqrt(max(norm_a_sq, 0.0))
        norm_b = math.sqrt(max(norm_b_sq, 0.0))
        denom = max(norm_a * norm_b, 1e-12)
        cosine = dot / denom
        return dot, norm_a, norm_b, cosine

    # ════════════════════════════════════════════════════════════
    # SELF-CERTAINTY COMPUTATION
    # ════════════════════════════════════════════════════════════

    def _compute_self_certainty(
        self, model, input_ids, attention_mask, completion_ids, completion_mask,
        logits_to_keep, batch_size=None,
    ) -> torch.Tensor:
        """Compute per-sample self-certainty scores.

        If thinking mode, uses reasoning_mask (tokens inside <think>...</think>).
        Otherwise uses full completion_mask.

        Returns:
            (B,) self-certainty scores
        """
        batch_size = batch_size or input_ids.size(0)
        all_sc_per_token = []

        # Chunked forward pass (no grad) for self-certainty
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for start in range(0, input_ids.size(0), batch_size):
                end = min(start + batch_size, input_ids.size(0))
                chunk_ids = input_ids[start:end]
                chunk_attn = attention_mask[start:end]

                logits = model(input_ids=chunk_ids, attention_mask=chunk_attn).logits
                # Use raw logits (no temperature) matching Intuitor's approach
                logits = logits[:, -(logits_to_keep + 1):-1, :]
                sc = compute_self_certainty_per_token(logits)
                all_sc_per_token.append(sc)
                del logits

        if was_training:
            model.train()

        per_token_sc = torch.cat(all_sc_per_token, dim=0)  # (B, S)

        # Choose mask: reasoning only (thinking mode) or full completion
        if self.thinking:
            reasoning_mask = build_reasoning_mask(completion_ids, completion_mask)
            sc_mask = reasoning_mask
        else:
            sc_mask = completion_mask

        # Aggregate per-token SC to per-sample
        self_certainty = aggregate_self_certainty(per_token_sc, sc_mask)

        return self_certainty

    # ════════════════════════════════════════════════════════════
    # FORWARD WITH ENTROPY (for _compute_loss extensions)
    # ════════════════════════════════════════════════════════════

    def _forward_with_entropy(self, model, input_ids, attention_mask, logits_to_keep):
        """Forward pass returning per-token logps AND entropy.

        TRL's _get_per_token_logps only returns logps. We need entropy too
        for the entropy bonus in _compute_loss. This method replicates TRL's
        logps computation and adds chunk-wise entropy calculation.

        Returns:
            (per_token_logps, entropies) — both (B, S) tensors
        """
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        completion_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]

        # Temperature scaling (matching TRL's _get_per_token_logps)
        logits = logits / self.temperature

        # Per-token logps via log_softmax + gather
        log_probs = logits.log_softmax(dim=-1)
        per_token_logps = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

        # Entropy: H(π) = -sum(p * log(p)) = logsumexp - E_π[logits]
        # Computed chunk-wise to avoid materializing full (B, S, V) probs
        B, S, V = logits.shape
        lse = logits.logsumexp(dim=-1)
        CHUNK = 32768
        weighted_sum = torch.zeros(B, S, device=logits.device, dtype=logits.dtype)
        for v_start in range(0, V, CHUNK):
            v_end = min(v_start + CHUNK, V)
            chunk_logits = logits[:, :, v_start:v_end]
            chunk_probs = torch.exp(chunk_logits - lse.unsqueeze(-1))
            weighted_sum += (chunk_probs * chunk_logits).sum(dim=-1)
            del chunk_probs, chunk_logits
        entropies = lse - weighted_sum

        del log_probs, lse, weighted_sum
        return per_token_logps, entropies

    # ════════════════════════════════════════════════════════════
    # OVERRIDE: _generate_and_score_completions
    # ════════════════════════════════════════════════════════════

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions and compute GDPO multi-channel advantages.

        Extends base GRPOTrainer by:
          1. Computing self-certainty from model logits
          2. Extracting per-function rewards for GDPO channels
          3. Z-scoring each channel independently within prompt groups
          4. Combining channels with configurable weights
        """
        # Call parent to get completions, rewards, and standard advantages
        result = super()._generate_and_score_completions(inputs)

        device = result["advantages"].device
        mode = "train" if self.model.training else "eval"

        # ── Compute self-certainty ──
        prompt_ids = result["prompt_ids"]
        prompt_mask = result["prompt_mask"]
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        self_certainty = self._compute_self_certainty(
            self.model, input_ids, attention_mask,
            completion_ids, completion_mask,
            logits_to_keep, batch_size=4,
        )

        # ── Build group indices ──
        B = self_certainty.shape[0]
        G = self.num_generations
        group_idx = torch.arange(B, device=device) // G

        # ── Penalize short completions and no-reasoning ──
        self_certainty = penalize_short_completions(
            self_certainty, completion_mask, group_idx, self.min_completion_tokens)
        if self.thinking:
            reasoning_mask = build_reasoning_mask(completion_ids, completion_mask)
            self_certainty = penalize_no_reasoning(
                self_certainty, reasoning_mask, group_idx)

        # ── GDPO per-channel normalization ──
        # Channel 1: cluster+format (from TRL's reward_funcs)
        # TRL already computed rewards as weighted sum in result["advantages"]
        # But we need the raw per-function rewards for GDPO.
        # The parent stores rewards_per_func internally — we reconstruct from
        # the standard advantages (which are z-scored rewards).
        # Actually, we use TRL's already-computed advantages as the cluster channel.
        trl_advantages = result["advantages"]  # already z-scored by TRL

        # Channel 2: self-certainty (z-scored per group)
        ch_sc = gdpo_normalize(self_certainty.detach(), group_idx)

        # Combine channels via GDPO weighting
        gdpo_advantages = self.w_cluster_format * trl_advantages + self.w_selfcert * ch_sc

        # Keep per-channel advantages for optional gradient-correlation probes.
        result["advantages_cf"] = trl_advantages.detach()
        result["advantages_sc"] = ch_sc.detach()

        # Optional: isolate cluster-only channel from TRL per-reward outputs.
        # Expected key from TRL internals: rewards_per_func with shape (B, R).
        # We assume reward_funcs order from config (default: cluster_voting, boxed_format).
        # If unavailable, cluster-only diagnostics are skipped safely.
        cluster_adv = None
        rewards_per_func = self._last_rewards_per_func
        if rewards_per_func is not None:
            try:
                if isinstance(rewards_per_func, (list, tuple)):
                    rewards_per_func = torch.stack([
                        x if torch.is_tensor(x) else torch.tensor(x, device=device)
                        for x in rewards_per_func
                    ], dim=-1)
                if torch.is_tensor(rewards_per_func) and rewards_per_func.ndim == 2 and rewards_per_func.size(1) >= 1:
                    cluster_raw = rewards_per_func[:, 0]
                    cluster_adv = gdpo_normalize(cluster_raw.detach(), group_idx)
                    result["advantages_cluster"] = cluster_adv
            except Exception as e:
                logger.debug("Could not derive cluster-only advantages: %s", e)

        # Replace TRL's advantages with GDPO advantages
        result["advantages"] = gdpo_advantages

        # ── Log multi-reward metrics ──
        self._metrics[mode]["rewards/self_certainty"].append(self_certainty.mean().item())
        self._metrics[mode]["advantages/ch_sc_std"].append(
            ch_sc.std(correction=0).item() if ch_sc.numel() > 1 else 0.0)
        self._metrics[mode]["advantages/gdpo_combined_std"].append(
            gdpo_advantages.std(correction=0).item() if gdpo_advantages.numel() > 1 else 0.0)

        # Reward-channel alignment in advantage space (cheap diagnostic).
        # Mixed-precision safe: compute cosine diagnostics in fp32.
        cf_center = (trl_advantages - trl_advantages.mean()).float()
        sc_center = (ch_sc - ch_sc.mean()).float()
        denom = (cf_center.norm() * sc_center.norm()).item()
        adv_cos = torch.dot(cf_center, sc_center).item() / max(denom, 1e-12)
        self._metrics[mode]["advantages/ch_cf_sc_cosine"].append(adv_cos)

        if cluster_adv is not None:
            cl_center = (cluster_adv - cluster_adv.mean()).float()
            denom_cl = (cl_center.norm() * sc_center.norm()).item()
            adv_cl_sc_cos = torch.dot(cl_center, sc_center).item() / max(denom_cl, 1e-12)
            self._metrics[mode]["advantages/ch_cluster_sc_cosine"].append(adv_cl_sc_cos)

        if self.thinking:
            # reasoning_mask already computed above (line 166)
            self._metrics[mode]["reasoning/frac_with_think"].append(
                (reasoning_mask.sum(dim=-1) > 0).float().mean().item())
            self._metrics[mode]["reasoning/avg_reasoning_tokens"].append(
                reasoning_mask.sum(dim=-1).float().mean().item())

        return result

    # ════════════════════════════════════════════════════════════
    # OVERRIDE: _compute_loss
    # ════════════════════════════════════════════════════════════

    def _compute_loss(self, model, inputs):
        """Compute GRPO loss with KL-Cov regularization and entropy bonus.

        Extends base GRPOTrainer._compute_loss by:
          1. KL-Cov: high-covariance tokens get unclipped loss + KL penalty
          2. Entropy bonus: -α * H(π) encourages exploration
        """
        # If no extensions active, use base implementation (faster)
        if self.beta_cov == 0.0 and self.entropy_bonus == 0.0:
            return super()._compute_loss(model, inputs)

        # ── Replicate base computation with extensions ──
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Forward pass — get logits, then compute logps + entropy from them
        per_token_logps, entropies = self._forward_with_entropy(
            model, input_ids, attention_mask, logits_to_keep,
        )

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

        advantages_cf = inputs.get("advantages_cf")
        advantages_sc = inputs.get("advantages_sc")
        advantages_cluster = inputs.get("advantages_cluster")

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # ── KL-Cov extension ──
        if self.beta_cov > 0.0:
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
        else:
            # Standard clipped surrogate (no KL-Cov)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL penalty
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # ── Entropy bonus extension ──
        if self.entropy_bonus > 0.0 and entropies is not None:
            per_token_loss = per_token_loss - self.entropy_bonus * entropies

        # Aggregate loss
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        normalizer = self.current_gradient_accumulation_steps
        loss = loss / normalizer

        # ── Optional gradient-correlation probe between channels ──
        mode = "train" if self.model.training else "eval"
        should_probe = (
            self.grad_corr_enabled
            and self.model.training
            and not self._grad_corr_failed
            and advantages_cf is not None
            and advantages_sc is not None
            and bool(getattr(self.accelerator, "sync_gradients", True))
            and ((int(self.state.global_step) + 1) % max(int(self.grad_corr_every_steps), 1) == 0)
        )

        if should_probe:
            try:
                probe_params = self._get_grad_corr_probe_params()
                if probe_params:
                    adv_cf = advantages_cf.unsqueeze(1) if advantages_cf.dim() == 1 else advantages_cf
                    adv_sc = advantages_sc.unsqueeze(1) if advantages_sc.dim() == 1 else advantages_sc

                    # Use the same clipped surrogate form for each channel.
                    cf_l1 = coef_1 * adv_cf
                    cf_l2 = coef_2 * adv_cf
                    sc_l1 = coef_1 * adv_sc
                    sc_l2 = coef_2 * adv_sc
                    cf_token_loss = -torch.min(cf_l1, cf_l2)
                    sc_token_loss = -torch.min(sc_l1, sc_l2)

                    cf_loss = ((cf_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
                    sc_loss = ((sc_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
                    cf_loss = cf_loss / normalizer
                    sc_loss = sc_loss / normalizer

                    grads_cf = torch.autograd.grad(
                        cf_loss, probe_params, retain_graph=True, allow_unused=True
                    )
                    grads_sc = torch.autograd.grad(
                        sc_loss, probe_params, retain_graph=True, allow_unused=True
                    )

                    dot, norm_cf, norm_sc, cos = self._grad_stats_from_pairs(grads_cf, grads_sc)

                    self._metrics[mode]["gradcorr/cf_sc_dot"].append(dot)
                    self._metrics[mode]["gradcorr/cf_norm"].append(norm_cf)
                    self._metrics[mode]["gradcorr/sc_norm"].append(norm_sc)
                    self._metrics[mode]["gradcorr/cf_sc_cosine"].append(cos)
                    self._metrics[mode]["gradcorr/probe_tensors"].append(float(len(probe_params)))

                    # Third backward (cluster-only vs self-certainty) commented out
                    # to keep activation memory in budget for the LoRA gradcorr probe.
                    # Re-enable if you need to disambiguate cluster_voting from boxed_format.
                    # if advantages_cluster is not None:
                    #     adv_cl = (
                    #         advantages_cluster.unsqueeze(1)
                    #         if advantages_cluster.dim() == 1
                    #         else advantages_cluster
                    #     )
                    #     cl_l1 = coef_1 * adv_cl
                    #     cl_l2 = coef_2 * adv_cl
                    #     cl_token_loss = -torch.min(cl_l1, cl_l2)
                    #     cl_loss = ((cl_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
                    #     cl_loss = cl_loss / normalizer
                    #
                    #     grads_cl = torch.autograd.grad(
                    #         cl_loss, probe_params, retain_graph=True, allow_unused=True
                    #     )
                    #     dot_cl, norm_cl, norm_sc2, cos_cl_sc = self._grad_stats_from_pairs(grads_cl, grads_sc)
                    #     self._metrics[mode]["gradcorr/cluster_sc_dot"].append(dot_cl)
                    #     self._metrics[mode]["gradcorr/cluster_norm"].append(norm_cl)
                    #     self._metrics[mode]["gradcorr/sc_norm_clusterprobe"].append(norm_sc2)
                    #     self._metrics[mode]["gradcorr/cluster_sc_cosine"].append(cos_cl_sc)
            except RuntimeError as e:
                self._grad_corr_failed = True
                logger.warning("Gradient-correlation probe disabled after runtime error: %s", e)

        # ── Metrics ──
        if self.beta != 0.0:
            mean_kl = ((per_token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        if entropies is not None:
            mean_entropy = ((entropies * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            self._metrics[mode]["entropy"].append(
                self.accelerator.gather(mean_entropy).nanmean().item())

        if self.beta_cov > 0.0:
            n_high = high_cov_mask.sum().item()
            n_total = mask.sum().item()
            self._metrics[mode]["klcov/high_cov_fraction"].append(n_high / max(n_total, 1))
            if n_high > 0:
                self._metrics[mode]["klcov/mean_kl_penalty"].append(
                    (kl_penalty * high_cov_mask).sum().item() / n_high)

        return loss
