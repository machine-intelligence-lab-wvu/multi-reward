"""MultiRewardTrainer: cluster voting + self-certainty with GDPO normalization, and KL-Cov.

Advantage construction (per prompt group of G completions):

  channel cf : TRL combines the TRL reward functions (cluster_voting, boxed_format)
               with weights (w_cluster, w_format) and z-scores the sum per group.
  channel sc : self-certainty of each completion under the current policy,
               z-scored per group (population std).
  A = w_cluster_format * cf + w_selfcert * sc

With `gdpo_norm_after_combine` (GDPO ablation) the raw rewards are summed first
and a single per-group z-score is applied to the total.

The loss is TRL's GRPO loss, with KL-Cov when `beta_cov > 0` (see klcov.py).
"""

import logging

import torch
from trl import GRPOTrainer

from open_r1_imp.klcov import KLCovMixin
from open_r1_imp.utils import compute_self_certainty, gdpo_normalize

logger = logging.getLogger(__name__)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = (a - a.mean()).float()
    b = (b - b.mean()).float()
    return torch.dot(a, b).item() / max((a.norm() * b.norm()).item(), 1e-12)


class MultiRewardTrainer(KLCovMixin, GRPOTrainer):

    def __init__(self, *args, **kwargs):
        training_args = kwargs["args"]
        self.w_cluster_format = training_args.w_cluster_format
        self.w_selfcert = training_args.w_selfcert
        self.gdpo_norm_after_combine = training_args.gdpo_norm_after_combine
        self.beta_cov = training_args.beta_cov
        self.top_k_fraction = training_args.top_k_fraction
        self.sc_batch_size = training_args.sc_batch_size
        self.grad_corr_enabled = training_args.grad_corr_enabled
        self.grad_corr_every_steps = max(int(training_args.grad_corr_every_steps), 1)
        super().__init__(*args, **kwargs)
        if len(self.reward_funcs) != 2:
            raise ValueError("multi_reward expects reward_funcs: [cluster_voting, boxed_format]")
        # Weights TRL uses to combine the reward functions into the cf channel.
        self.reward_weights = torch.tensor([training_args.w_cluster, training_args.w_format], dtype=torch.float32)
        self._rewards_per_func = None

    def _calculate_rewards(self, *args, **kwargs):
        # Keep the per-function rewards (TRL returns them gathered across processes).
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._rewards_per_func = rewards_per_func.detach()
        return rewards_per_func

    def _local_rewards_per_func(self, local_size: int):
        r = self._rewards_per_func
        if r.size(0) == local_size:
            return r
        rank = self.accelerator.process_index
        return r[rank * local_size:(rank + 1) * local_size]

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"

        self_certainty = compute_self_certainty(
            self.model, result["prompt_ids"], result["prompt_mask"],
            result["completion_ids"], result["completion_mask"], batch_size=self.sc_batch_size)
        B = self_certainty.size(0)
        group_idx = torch.arange(B, device=self_certainty.device) // self.num_generations

        if self.gdpo_norm_after_combine:
            rpf = self._local_rewards_per_func(B)
            cf_raw = (rpf * self.reward_weights.to(rpf.device, rpf.dtype)).nansum(dim=-1)
            advantages = gdpo_normalize(self.w_cluster_format * cf_raw + self.w_selfcert * self_certainty, group_idx)
            ch_cf, ch_sc = cf_raw, self_certainty
        else:
            ch_cf = result["advantages"]                      # TRL's group z-score of the cf channel
            ch_sc = gdpo_normalize(self_certainty, group_idx)
            advantages = self.w_cluster_format * ch_cf + self.w_selfcert * ch_sc

        result["advantages"] = advantages
        result["advantages_cf"] = ch_cf.detach()
        result["advantages_sc"] = ch_sc.detach()

        self._metrics[mode]["rewards/self_certainty"].append(self_certainty.mean().item())
        self._metrics[mode]["advantages/ch_sc_std"].append(ch_sc.std(correction=0).item())
        self._metrics[mode]["advantages/gdpo_combined_std"].append(advantages.std(correction=0).item())
        self._metrics[mode]["advantages/ch_cf_sc_cosine"].append(_cosine(ch_cf, ch_sc))
        return result

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if (self.grad_corr_enabled and self.model.training
                and self.accelerator.sync_gradients
                and (self.state.global_step + 1) % self.grad_corr_every_steps == 0):
            self._log_channel_gradients(inputs)
        return loss

    def _log_channel_gradients(self, inputs):
        """Per-channel gradient norms/cosine of the linearized surrogate w.r.t. per-token
        log-probs (no extra model backward). Used for the GDPO-ablation diagnostic."""
        t = self._last_loss_terms
        logps, mask = t["per_token_logps"], t["completion_mask"]
        adv = inputs["advantages"].unsqueeze(1)
        with torch.no_grad():
            uses_coef1 = (t["coef_1"] * adv <= t["coef_2"] * adv) | t["high_cov_mask"].bool()
        c_eff = torch.where(uses_coef1, t["coef_1"], t["coef_2"])
        ga = self.current_gradient_accumulation_steps

        def grad_of(channel_adv):
            per_tok = -c_eff * channel_adv.unsqueeze(1)
            lin = ((per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean() / ga
            g, = torch.autograd.grad(lin, logps, retain_graph=True)
            return (g.float() * mask).reshape(-1)

        try:
            g_cf, g_sc = grad_of(inputs["advantages_cf"]), grad_of(inputs["advantages_sc"])
        except RuntimeError as e:
            logger.warning("Gradient probe disabled after error: %s", e)
            self.grad_corr_enabled = False
            return
        n_cf, n_sc = g_cf.norm().item(), g_sc.norm().item()
        mode = "train"
        self._metrics[mode]["gradcorr_logps/cf_norm"].append(n_cf)
        self._metrics[mode]["gradcorr_logps/sc_norm"].append(n_sc)
        self._metrics[mode]["gradcorr_logps/cf_sc_cosine"].append(
            torch.dot(g_cf, g_sc).item() / max(n_cf * n_sc, 1e-12))
