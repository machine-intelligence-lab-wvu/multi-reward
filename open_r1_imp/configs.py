"""Configuration dataclasses shared by all three training methods.

A recipe YAML is parsed into (ScriptArguments, TrainConfig, trl.ModelConfig).
`method` selects the trainer:

  multi_reward  cluster-voting + self-certainty with GDPO normalization (ours);
                also the cluster-only baseline (w_selfcert: 0, w_format: 0)
  intuitor      self-certainty only (Zhao et al., 2025)
  grpo_gt       supervised GRPO with the ground-truth accuracy reward

Any TRL `GRPOConfig` field can also be set in the recipe.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import trl
from trl import GRPOConfig

METHODS = ("multi_reward", "intuitor", "grpo_gt")


@dataclass
class ScriptArguments(trl.ScriptArguments):
    dataset_prompt_column: str = field(default="problem", metadata={"help": "Dataset column holding the prompt."})
    reward_funcs: List[str] = field(
        default_factory=lambda: ["cluster_voting", "boxed_format"],
        metadata={"help": "Reward functions from open_r1_imp.rewards: cluster_voting, boxed_format, accuracy."})


@dataclass
class TrainConfig(GRPOConfig):
    method: str = field(default="multi_reward", metadata={"help": f"One of {METHODS}."})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System message prepended to every prompt."})

    # Multi-reward advantage: A = w_cluster_format * z(cluster+format) + w_selfcert * z(self-certainty)
    w_cluster_format: float = field(default=0.5, metadata={"help": "Weight of the cluster+format channel (alpha)."})
    w_selfcert: float = field(default=0.5, metadata={"help": "Weight of the self-certainty channel (beta)."})
    w_cluster: float = field(default=0.5, metadata={"help": "Weight of cluster_voting inside the cf channel."})
    w_format: float = field(default=0.1, metadata={"help": "Weight of boxed_format inside the cf channel."})
    gdpo_norm_after_combine: bool = field(
        default=False,
        metadata={"help": "GDPO ablation: sum the raw channel rewards first, then z-score once "
                          "(combine-then-normalize) instead of normalizing each channel."})

    # KL-Cov
    beta_cov: float = field(default=0.0, metadata={"help": "KL-Cov coefficient; 0 disables KL-Cov."})
    top_k_fraction: float = field(default=0.02, metadata={"help": "Fraction of tokens selected by covariance."})

    # Self-certainty
    sc_batch_size: int = field(default=4, metadata={"help": "Micro-batch size for the self-certainty forward pass."})

    # Diagnostics (GDPO ablation, Appendix): per-channel gradient norms in log-prob space.
    grad_corr_enabled: bool = field(default=False, metadata={"help": "Log per-channel log-prob gradient norms (needs beta_cov > 0)."})
    grad_corr_every_steps: int = field(default=1, metadata={"help": "Probe every N optimizer steps."})

    # Disk usage
    keep_full_checkpoints: int = field(
        default=1, metadata={"help": "Newest N checkpoints keep optimizer state; older ones keep weights only."})

    def __post_init__(self):
        super().__post_init__()
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.grad_corr_enabled and self.beta_cov == 0.0:
            raise ValueError("grad_corr_enabled requires beta_cov > 0 (the probe reuses the KL-Cov loss graph).")
