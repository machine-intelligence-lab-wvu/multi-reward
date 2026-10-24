"""KL-Cov: covariance-targeted KL regularization (Cui et al., 2025, arXiv:2505.22617).

For every completion token y_i in the local micro-batch we compute

    Cov(y_i) = (log pi(y_i) - mean log pi) * (A_i - mean A)

and select the top `top_k_fraction` tokens by covariance. Those tokens use the
unclipped surrogate plus a penalty beta_cov * |log pi_theta - log pi_old|; every
other token uses the standard clipped GRPO surrogate.

`KLCovMixin._compute_loss` is a drop-in replacement for TRL's
`GRPOTrainer._compute_loss` (TRL 0.23.1). Apart from the KL-Cov split it mirrors
TRL exactly: the same importance-sampling correction for vLLM, the same
reference-KL term and the same `loss_type` aggregation and gradient-accumulation
scaling. With beta_cov == 0 the base TRL loss is used unchanged.
"""

import torch


def compute_token_covariances(log_probs, advantages, response_mask):
    """Per-token centered cross-product between log-prob and advantage.

    Args:
        log_probs: (B, S) log-probs of the sampled tokens (detached).
        advantages: (B,) per-completion advantages.
        response_mask: (B, S) 1 for completion tokens.
    Returns:
        (B, S) covariance contributions (0 on padding).
    """
    B, S = log_probs.shape
    adv_per_token = advantages.unsqueeze(1).expand(B, S)
    total_tokens = response_mask.sum().clamp(min=1)
    mean_lp = (log_probs * response_mask).sum() / total_tokens
    mean_adv = (adv_per_token * response_mask).sum() / total_tokens
    centered_lp = (log_probs - mean_lp) * response_mask
    centered_adv = (adv_per_token - mean_adv) * response_mask
    return centered_lp * centered_adv


def select_high_covariance_tokens(covariances, response_mask, top_k_fraction=0.02):
    """Mask of the top-k fraction of completion tokens by (signed) covariance."""
    flat_cov = covariances.reshape(-1)
    flat_mask = response_mask.reshape(-1)
    valid_indices = flat_mask.nonzero(as_tuple=True)[0]
    n_valid = valid_indices.shape[0]
    if n_valid == 0:
        return torch.zeros_like(response_mask)
    k = max(1, int(top_k_fraction * n_valid))
    _, topk_local = flat_cov[valid_indices].topk(k, largest=True)
    high_cov_flat = torch.zeros_like(flat_mask)
    high_cov_flat[valid_indices[topk_local]] = 1
    return high_cov_flat.view_as(response_mask)


class KLCovMixin:
    """Adds KL-Cov to a TRL GRPOTrainer subclass. Requires `self.beta_cov` and
    `self.top_k_fraction` to be set before `_compute_loss` is called."""

    def _compute_loss(self, model, inputs):
        if self.beta_cov == 0.0:
            return super()._compute_loss(model, inputs)
        if self.importance_sampling_level != "token" or self.top_entropy_quantile < 1.0 or self.use_liger_loss:
            raise ValueError("KL-Cov supports importance_sampling_level='token', "
                             "top_entropy_quantile=1.0 and use_liger_loss=False only.")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # KL-Cov split: top-k covariance tokens get the unclipped surrogate + |log ratio| penalty.
        covariances = compute_token_covariances(per_token_logps.detach(), advantages, completion_mask)
        high_cov_mask = select_high_covariance_tokens(covariances, completion_mask, self.top_k_fraction)
        adv = advantages.unsqueeze(1)
        clipped_loss = -torch.min(coef_1 * adv, coef_2 * adv)
        penalized_loss = -coef_1 * adv + self.beta_cov * log_ratio.abs()
        per_token_loss = clipped_loss * (1 - high_cov_mask) + penalized_loss * high_cov_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics (same keys as TRL, plus KL-Cov diagnostics).
        mode = "train" if self.model.training else "eval"
        n_tokens = completion_mask.sum().clamp(min=1.0)
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / n_tokens
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())
        mean_entropy = (entropies * completion_mask).sum() / n_tokens
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        n_high = high_cov_mask.sum()
        self._metrics[mode]["klcov/high_cov_fraction"].append((n_high / n_tokens).item())
        penalty = (self.beta_cov * log_ratio.abs() * high_cov_mask).sum() / n_high.clamp(min=1)
        self._metrics[mode]["klcov/mean_kl_penalty"].append(penalty.item())
        self._last_loss_terms = dict(per_token_logps=per_token_logps, coef_1=coef_1, coef_2=coef_2,
                                     high_cov_mask=high_cov_mask, completion_mask=completion_mask)
        return loss
