"""
Extended configuration for Multi-Reward GRPO.

Extends TRL's GRPOConfig and ScriptArguments directly (no Open-R1 dependency).
This makes the package fully self-contained.

Extensions:
  - GDPO channel weights
  - KL-Cov parameters
  - Thinking mode flag
  - EMPO semantic clustering options
  - System prompt and dataset column settings
"""

from dataclasses import dataclass, field
from typing import List, Optional

from trl import GRPOConfig, ScriptArguments


@dataclass
class MultiRewardGRPOConfig(GRPOConfig):
    """Training config with multi-reward extensions.

    Extends trl.GRPOConfig with GDPO, KL-Cov, thinking mode, and wandb settings.
    """

    # System prompt (Open-R1 had this, TRL doesn't)
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt prepended to every conversation."},
    )

    # Needed for base models that ship no chat template (e.g. meta-llama/
    # Llama-3.2-3B). Applied to the tokenizer in utils.get_tokenizer(), so it
    # is used for training AND saved into every checkpoint's tokenizer_config
    # (evaluation of checkpoints then needs no extra handling).
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Jinja chat template to assign to the tokenizer "
                          "(overrides the tokenizer's own, if any)."},
    )

    # GDPO channel weights (Eq. 3-4)
    w_cluster_format: float = field(
        default=0.5,
        metadata={"help": "GDPO weight for cluster+format channel"},
    )
    w_selfcert: float = field(
        default=0.5,
        metadata={"help": "GDPO weight for self-certainty channel"},
    )

    # KL-Cov regularization (Eq. 6-9)
    beta_cov: float = field(
        default=0.0,
        metadata={"help": "KL-Cov penalty coefficient. 0 = disabled."},
    )
    top_k_fraction: float = field(
        default=0.02,
        metadata={"help": "Fraction of high-covariance tokens to penalize"},
    )

    # Entropy bonus
    entropy_bonus: float = field(
        default=0.0,
        metadata={"help": "Entropy bonus for exploration. 0 = disabled."},
    )

    # Thinking mode (Qwen3)
    thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode: SC on reasoning only, rewards on answer only"},
    )

    # Min completion length filter
    min_completion_tokens: int = field(
        default=50,
        metadata={"help": "Completions shorter than this get reward=0. Prevents template collapse."},
    )

    # Self-certainty forward-pass micro-batch (mirrors configs_grpo_gt/intuitor;
    # required so A-5 yamls that set it parse under the multi trainer too)
    sc_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size for the self-certainty forward pass"},
    )

    # HuggingFace Hub push interval (independent of save_steps)
    hub_push_every_n_steps: int = field(
        default=50,
        metadata={"help": "Push checkpoint to HF Hub every N steps. Independent of save_steps."},
    )

    # In-training accuracy evaluation
    eval_accuracy_steps: int = field(
        default=0,
        metadata={"help": "Evaluate pass@1 accuracy every N steps. 0 = disabled. Pauses training."},
    )
    eval_accuracy_dataset: str = field(
        default="math500",
        metadata={"help": "Dataset for in-training accuracy eval (gsm8k, math500, math)"},
    )
    eval_accuracy_samples: int = field(
        default=50,
        metadata={"help": "Number of samples for in-training accuracy eval (keep small for speed)"},
    )

    # Ablation: skip per-group z-score (GDPO normalization) on the SC channel
    # so raw self-certainty is combined with TRL's already-normalized CF
    # advantage. Used to empirically show why separate per-channel normalization
    # is needed before combining rewards.
    disable_gdpo_norm: bool = field(
        default=False,
        metadata={"help": "If true, skip per-group z-score on self-certainty (and cluster logging) advantage."},
    )

    # Ablation (variant B): combine raw per-channel rewards first, then run a
    # single per-group z-score on the sum. Bypasses TRL's per-channel z-score
    # on the cluster+format channel as well. Takes precedence over
    # disable_gdpo_norm if both are set.
    gdpo_norm_after_combine: bool = field(
        default=False,
        metadata={"help": "If true, combine raw cluster+format+SC rewards then z-score once on the sum."},
    )

    # Gradient-correlation diagnostics between reward channels
    grad_corr_enabled: bool = field(
        default=False,
        metadata={"help": "Log gradient cosine between cluster/format and self-certainty channels."},
    )
    grad_corr_every_steps: int = field(
        default=20,
        metadata={"help": "Run gradient-correlation probe every N optimizer steps."},
    )
    grad_corr_num_tensors: int = field(
        default=16,
        metadata={"help": "Number of trailing trainable parameter tensors used for grad-correlation probe. <=0 uses all."},
    )


@dataclass
class MultiRewardScriptArguments(ScriptArguments):
    """Script args with multi-reward extensions.

    Extends trl.ScriptArguments with reward function config, EMPO settings,
    and dataset column specification.
    """

    # Reward functions (list of names from our registry)
    reward_funcs: List[str] = field(
        default_factory=lambda: ["cluster_voting", "boxed_format"],
        metadata={"help": "List of reward function names"},
    )

    # Optional domain normalizer applied to extracted answers BEFORE clustering
    # in cluster_voting. The math normalizer does not strip punctuation or
    # articles ("PARIS." / "The Paris" fragment away from "Paris"), so
    # free-text domains (T2 TriviaQA) set "trivia": SQuAD-style lowercase +
    # strip articles/punctuation. None = unchanged math behavior.
    cluster_answer_normalizer: Optional[str] = field(
        default=None,
        metadata={"help": "None (math default) or 'trivia' (SQuAD-style "
                          "normalization for free-text answers)"},
    )

    # Dataset column
    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the prompt/problem text"},
    )

    # Reward weights within the cluster+format channel
    w_cluster: float = field(
        default=0.5,
        metadata={"help": "Weight for cluster reward within cluster_format channel"},
    )
    w_format: float = field(
        default=0.1,
        metadata={"help": "Weight for format reward within cluster_format channel"},
    )

    # EMPO semantic clustering
    use_semantic_clustering: bool = field(
        default=False,
        metadata={"help": "Use DeBERTa NLI for semantic clustering (EMPO method)"},
    )
    deberta_model: str = field(
        default="microsoft/deberta-v3-large",
        metadata={"help": "DeBERTa model for semantic similarity"},
    )
    nli_threshold: float = field(
        default=0.5,
        metadata={"help": "NLI entailment threshold for merging clusters"},
    )
    entropy_filter_low: float = field(
        default=0.0,
        metadata={"help": "EMPO: skip groups with semantic entropy below this"},
    )
    entropy_filter_high: float = field(
        default=float("inf"),
        metadata={"help": "EMPO: skip groups with semantic entropy above this"},
    )
    empo_open_ended: bool = field(
        default=False,
        metadata={"help": "EMPO: cluster whole completions instead of \\boxed{} answers "
                          "(non-verifiable prompts, e.g. A-5 on Alpaca)"},
    )

    # Soft embedding-consensus reward (open-ended / A-5 v2)
    consensus_embed_model: str = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={"help": "Sentence-embedding model for the soft_consensus reward"},
    )
    consensus_max_length: int = field(
        default=512,
        metadata={"help": "Token truncation for consensus embeddings"},
    )

    # Dataset row filtering
    dataset_filter_empty: Optional[str] = field(
        default=None,
        metadata={"help": "Keep only rows where this column is empty/missing, e.g. "
                          "'input' for Alpaca's no-context subset (~31k rows)"},
    )
