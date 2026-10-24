"""IntuitorTrainer: self-certainty as the only reward (Zhao et al., 2025, "Learning to
Reason without External Rewards").

Self-certainty is computed online from the current policy and z-scored within each
prompt group across all processes (sample std, eps=1e-4, as in the INTUITOR code).
The TRL reward functions in the recipe are computed for logging only.
Optional KL-Cov (`beta_cov > 0`) for the INTUITOR + KL-Cov ablation.
"""

import torch
from trl import GRPOTrainer

from open_r1_imp.klcov import KLCovMixin
from open_r1_imp.utils import compute_self_certainty


class IntuitorTrainer(KLCovMixin, GRPOTrainer):

    def __init__(self, *args, **kwargs):
        training_args = kwargs["args"]
        self.beta_cov = training_args.beta_cov
        self.top_k_fraction = training_args.top_k_fraction
        self.sc_batch_size = training_args.sc_batch_size
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"

        self_certainty = compute_self_certainty(
            self.model, result["prompt_ids"], result["prompt_mask"],
            result["completion_ids"], result["completion_mask"], batch_size=self.sc_batch_size)

        # Group statistics over the gathered batch so every rank sees the same mean/std.
        B, G = self_certainty.size(0), self.num_generations
        all_sc = self.accelerator.gather(self_certainty).view(-1, G)
        mean = all_sc.mean(dim=1).repeat_interleave(G)
        std = all_sc.std(dim=1, correction=1).repeat_interleave(G)
        local = self.accelerator.process_index * B + torch.arange(B, device=self_certainty.device)
        result["advantages"] = (self_certainty - mean[local]) / (std[local] + 1e-4)

        self._metrics[mode]["rewards/self_certainty"].append(self_certainty.mean().item())
        self._metrics[mode]["advantages/ch_sc_std"].append(result["advantages"].std(correction=0).item())
        return result
