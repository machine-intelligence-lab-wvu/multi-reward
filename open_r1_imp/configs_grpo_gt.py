"""
Configuration for standard GRPO with ground-truth reward.

Matches the Intuitor paper's GRPOConfig/GRPOScriptArguments structure exactly.
Extends TRL's GRPOConfig with system prompt, eval, and hub push settings
consistent with the other open_r1_imp configs.
No self-certainty, no GDPO, no KL-Cov — just supervised accuracy reward.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from trl import GRPOConfig, ScriptArguments


@dataclass
class GRPOGTConfig(GRPOConfig):
    """Training config for standard GRPO with ground-truth rewards.

    Extends trl.GRPOConfig with system prompt and eval/hub settings
    matching the other open_r1_imp configs.
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
        metadata={"help": "Enable thinking mode: evaluate answer after </think> only."},
    )

    # Self-certainty monitoring batch size (logged only, does not affect training)
    sc_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for chunked self-certainty forward pass (monitoring)."},
    )

    # KL-Cov regularization (targeted entropy-collapse control on top-covariance tokens).
    # Rebuttal run A4_grpo_gt_klcov: apply KL-Cov to the supervised GRPO-GT objective.
    beta_cov: float = field(
        default=0.0,
        metadata={"help": "KL-Cov coefficient. 0.0 = plain GRPO-GT (unchanged). "
                          ">0 penalizes the top-covariance tokens with an unclipped surrogate + KL."},
    )
    top_k_fraction: float = field(
        default=0.02,
        metadata={"help": "Fraction of highest-covariance completion tokens treated as KL-Cov tokens."},
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
class GRPOGTScriptArguments(ScriptArguments):
    """Script args for standard GRPO with ground-truth rewards.

    Matches the Intuitor paper's GRPOScriptArguments structure.
    """

    # Dataset column
    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the prompt/problem text"},
    )

    # Reward functions (accuracy = ground-truth correctness, matching Intuitor paper naming)
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward function names"},
    )

    # ── EMPO semantic clustering ────────────────────────────────────────────
    # These live on MultiRewardScriptArguments too, but this entry point
    # (grpo_gt_train.py) is what launch.sh routes the `empo_cluster` configs to
    # — A3_empo_* and A5_nonverif_ours_*. Without them here,
    # get_multi_reward_funcs' getattr(script_args, "use_semantic_clustering",
    # False) always resolved False, so semantic clustering was silently
    # unreachable from this entry point, and putting the key in the YAML failed
    # to parse (TrlParser uses fail_with_unknown_args=True).
    use_semantic_clustering: bool = field(
        default=False,
        metadata={"help": "Use DeBERTa NLI for semantic answer clustering (empo_cluster)."},
    )
    deberta_model: str = field(
        default="microsoft/deberta-large-mnli",
        metadata={"help": "NLI checkpoint for semantic clustering. Must have an "
                          "entailment label; microsoft/deberta-v3-large is a plain "
                          "encoder with a RANDOM classification head and will not work."},
    )
    nli_threshold: float = field(
        default=0.5,
        metadata={"help": "Bidirectional entailment probability required to merge two answers."},
    )
    empo_open_ended: bool = field(
        default=False,
        metadata={"help": "Cluster whole completions instead of \\boxed{} answers. "
                          "Required for non-verifiable/open-ended prompts (A-5): with "
                          "the default boxed extraction, open-ended completions are all "
                          "unparseable and the reward is identically 0 (no gradient). "
                          "Needs use_semantic_clustering=true to be meaningful."},
    )
