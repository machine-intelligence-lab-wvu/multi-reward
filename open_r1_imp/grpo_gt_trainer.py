"""GRPOGTTrainer: supervised GRPO with the ground-truth accuracy reward.

Training is TRL's GRPOTrainer unchanged (plus KL-Cov when `beta_cov > 0`, used for
the GRPO-GT + KL-Cov appendix experiment). Self-certainty is logged for comparison
with the unsupervised runs; it does not affect training.
"""

from trl import GRPOTrainer

from open_r1_imp.klcov import KLCovMixin
from open_r1_imp.utils import compute_self_certainty


class GRPOGTTrainer(KLCovMixin, GRPOTrainer):

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
        self._metrics[mode]["rewards/self_certainty"].append(self_certainty.mean().item())
        return result
