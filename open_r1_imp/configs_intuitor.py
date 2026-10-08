"""
Configuration for INTUITOR training.

Simplified config — no GDPO channel weights, no KL-Cov, no entropy bonus.
Only self-certainty-specific parameters are added to GRPOConfig.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from trl import GRPOConfig, ScriptArguments


@dataclass
class IntuitorGRPOConfig(GRPOConfig):
    """Training config for INTUITOR.

    Extends trl.GRPOConfig with thinking mode and collapse guard settings.
    """

    # System prompt
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

    # Thinking mode (for Qwen3 models)
    thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode: SC computed on reasoning tokens only."},
    )

    # Collapse guards
    min_completion_tokens: int = field(
        default=50,
        metadata={"help": "Completions shorter than this get SC set to group minimum."},
    )

    # Self-certainty computation batch size
    sc_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for chunked self-certainty forward pass."},
    )

    # KL-Cov regularization (optional extension)
    beta_cov: float = field(
        default=0.0,
        metadata={"help": "KL-Cov penalty coefficient. 0 = disabled (standard GRPO)."},
    )
    top_k_fraction: float = field(
        default=0.02,
        metadata={"help": "Fraction of high-covariance tokens to penalize."},
    )

    # HuggingFace Hub push interval
    hub_push_every_n_steps: int = field(
        default=50,
        metadata={"help": "Push checkpoint to HF Hub every N steps."},
    )

    # In-training accuracy evaluation
    eval_accuracy_steps: int = field(
        default=0,
        metadata={"help": "Evaluate pass@1 accuracy every N steps. 0 = disabled."},
    )
    eval_accuracy_dataset: str = field(
        default="math500",
        metadata={"help": "Dataset for in-training accuracy eval (gsm8k, math500, math)"},
    )
    eval_accuracy_samples: int = field(
        default=50,
        metadata={"help": "Number of samples for in-training accuracy eval."},
    )


@dataclass
class IntuitorScriptArguments(ScriptArguments):
    """Script args for INTUITOR.

    Extends trl.ScriptArguments with dataset column and reward function config.
    External reward functions are kept for logging/comparison but don't affect training.
    """

    # Dataset column
    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the prompt/problem text"},
    )

    # Reward functions — logged for comparison, zero-weighted in training
    reward_funcs: List[str] = field(
        default_factory=lambda: ["cluster_voting", "boxed_format"],
        metadata={"help": "Reward functions to log (do NOT affect INTUITOR training)"},
    )
