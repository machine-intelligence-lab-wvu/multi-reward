#!/usr/bin/env python3
"""
Generate all NeurIPS-rebuttal training configs into open_r1_imp/recipes/rebuttal/.

Hyperparameters match the existing `recipes/final/` 1.5B suite (LR 3e-6, beta 0.005,
eps 0.2, T 1.0, num_generations 7, max_completion 3072, effective batch 882) but with
a SHORT horizon (max_steps=200) for the rebuttal deadline. Outputs go to /ocean.

Run:  python open_r1_imp/scripts/rebuttal/make_configs.py
"""
import os, yaml

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "recipes", "rebuttal")
OUT_DIR = os.path.abspath(OUT_DIR)
OCEAN = "${MR_SCRATCH}"

SYS = ("You are a helpful AI Assistant, designed to provided well-reasoned and detailed "
       "responses. You FIRST think about the reasoning process step by step and then "
       "provide the user with the answer. Please enclose your final answer in the box: "
       "\\boxed{Your Answer}. Please stop generation immediately after outputing the box.")

def common(model, run_name, max_steps=200, per_device=21, grad_accum=14,
           dataset="DigitalLearningGmbH/MATH-lighteval", prompt_col="problem"):
    return {
        "model_name_or_path": model, "model_revision": "main", "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2", "use_peft": False,
        "dataset_name": dataset, "dataset_prompt_column": prompt_col,
        "system_prompt": SYS,
        "bf16": True, "beta": 0.005, "use_vllm": True, "do_eval": False,
        "learning_rate": 3.0e-06, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1,
        "epsilon": 0.2, "temperature": 1.0, "max_grad_norm": 1.0,
        "max_completion_length": 3072, "num_generations": 7, "num_iterations": 1,
        "num_train_epochs": 1,
        "per_device_train_batch_size": per_device, "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_steps": max_steps,
        "log_completions": False, "log_level": "info", "logging_first_step": True,
        "logging_steps": 1, "logging_strategy": "steps", "report_to": ["wandb"],
        "save_strategy": "steps", "save_steps": 20, "save_total_limit": 50, "seed": 7,
        "push_to_hub": False, "thinking": False, "overwrite_output_dir": False,
        "eval_accuracy_steps": 0, "eval_accuracy_dataset": "math500", "eval_accuracy_samples": 50,
        "output_dir": f"{OCEAN}/{run_name}", "run_name": run_name,
    }

def multi(model, run_name, beta_cov=0.02, entropy_bonus=0.0, **kw):
    c = common(model, run_name, **kw)
    # NOTE: MultiRewardGRPOConfig has NO sc_batch_size field (unlike intuitor/grpo configs) — do not add it.
    c.update({"reward_funcs": ["cluster_voting", "boxed_format"],
              "w_cluster_format": 0.5, "w_selfcert": 0.5, "w_cluster": 0.5, "w_format": 0.1,
              "beta_cov": beta_cov, "top_k_fraction": 0.02, "entropy_bonus": entropy_bonus,
              "min_completion_tokens": 0})
    return c

def intuitor(model, run_name, beta_cov=0.0, **kw):
    c = common(model, run_name, **kw)
    c.update({"reward_funcs": ["cluster_voting", "boxed_format"], "reward_weights": [0.0, 0.0],
              "beta_cov": beta_cov, "min_completion_tokens": 0, "sc_batch_size": 4})
    return c

def grpo_std(model, run_name, reward_funcs, reward_weights, **kw):
    # generic GRPO (grpo_gt_train.py entry) — any registered reward_funcs
    c = common(model, run_name, **kw)
    c.update({"reward_funcs": reward_funcs, "reward_weights": reward_weights, "sc_batch_size": 4})
    return c

Q3 = "Qwen/Qwen3-1.7B-Base"          # A-2 newer-generation (same family)
LL = "meta-llama/Llama-3.2-3B"       # B-6 different family (GATED on HF)
Q25 = "Qwen/Qwen2.5-1.5B"            # A-3/A-4/A-5/B-7 all at 1.5B

CONFIGS = {
    # ---- A-2 : newer generation (Qwen3-1.7B-Base) — full vs INTUITOR vs GRPO-GT ----
    "A2_qwen3_17b_full":     multi(Q3, "a2-qwen3-17b-full", beta_cov=0.02),
    "A2_qwen3_17b_intuitor": intuitor(Q3, "a2-qwen3-17b-intuitor"),
    "A2_qwen3_17b_grpo_gt":  grpo_std(Q3, "a2-qwen3-17b-grpo-gt", ["accuracy"], [1.0]),

    # ---- B-6 : different family (Llama-3.2-3B) — OPTIONAL, gated model, 3B batch ----
    "B6_llama32_3b_full":     multi(LL, "b6-llama32-3b-full", beta_cov=0.02, per_device=14, grad_accum=21),
    "B6_llama32_3b_intuitor": intuitor(LL, "b6-llama32-3b-intuitor", per_device=14, grad_accum=21),
    "B6_llama32_3b_grpo_gt":  grpo_std(LL, "b6-llama32-3b-grpo-gt", ["accuracy"], [1.0], per_device=14, grad_accum=21),

    # ---- A-3 : recent self-rewarding baselines vs ours (1.5B) ----
    # EMPO = semantic-cluster consensus. Faithful to arXiv:2504.05812: reward is PURELY the
    # meaning-cluster probability (size/n_parseable) — no auxiliary format reward. Unparseable
    # (no \boxed{}) answers get 0, which implicitly incentivizes format, as in EMPO.
    "A3_empo_qwen25_15b":        grpo_std(Q25, "a3-empo-qwen25-15b", ["empo_cluster"], [1.0]),
    # Co-rewarding-I — FAITHFUL port of tmlr-group/Co-rewarding (arXiv:2508.00410).
    # Trains on the authors' released Qwen3-32B rephrasings via the paired two-view
    # dataset (make_corewarding_dataset.py). Reward = cross-view swapped-majority
    # binary (their exact logic), no auxiliary rewards. Pairing REQUIRES:
    # shuffle_dataset false + adjacent pairs + a SINGLE training process
    # (launch.sh forces the 2-GPU shape for this id). grad_accum=42 is baked so the
    # per-process generation batch (21x42=882 completions = 126 prompts) covers
    # whole ori/aug pairs; effective batch stays 882 like every other run.
    "A3_corewarding_qwen25_15b": {**grpo_std(
        Q25, "a3-corewarding-qwen25-15b", ["co_rewarding"], [1.0],
        dataset="${MR_DATA_DIR}/corewarding_math_paired",
        prompt_col="problem", grad_accum=42),
        "shuffle_dataset": False},

    # ---- A-4 : RLVR-augmentation batch (1.5B) ----
    # (iv) config-only: GRPO-GT + majority-vote consensus aux reward
    "A4_rlvr_majorityvote_qwen25_15b": grpo_std(Q25, "a4-rlvr-majorityvote-qwen25-15b",
                                                ["accuracy", "cluster_voting"], [1.0, 0.5]),
    # (i) GRPO-GT + KL-Cov — KL-Cov ported into grpo_gt trainer (beta_cov field added).
    "A4_grpo_gt_klcov_qwen25_15b":     {**grpo_std(Q25, "a4-grpo-gt-klcov-qwen25-15b", ["accuracy"], [1.0]),
                                        "beta_cov": 0.02, "top_k_fraction": 0.02},
    # (iii) RLVR + entropy bonus — DEFERRED (entropy_bonus not ported to grpo_gt trainer); config removed.

    # ---- B-7 : targeted (KL-Cov) vs global entropy on OUR objective (1.5B) — config-only ----
    "B7_multireward_globalentropy_qwen25_15b": multi(Q25, "b7-multireward-globalentropy-qwen25-15b",
                                                     beta_cov=0.0, entropy_bonus=0.01),

    # ---- A-5 : non-verifiable / open-ended (1.5B) — DRAFT: dataset + LLM-judge TODO ----
    "A5_nonverif_ours_qwen25_15b":     grpo_std(Q25, "a5-nonverif-ours-qwen25-15b", ["empo_cluster"], [1.0],
                                                dataset="tatsu-lab/alpaca", prompt_col="instruction"),
    "A5_nonverif_intuitor_qwen25_15b": intuitor(Q25, "a5-nonverif-intuitor-qwen25-15b",
                                                dataset="tatsu-lab/alpaca", prompt_col="instruction"),
}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, cfg in CONFIGS.items():
        path = os.path.join(OUT_DIR, name + ".yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, width=1000)
        print("wrote", os.path.relpath(path))
    print(f"\n{len(CONFIGS)} configs written to {OUT_DIR}")
