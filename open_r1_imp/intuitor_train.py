"""
INTUITOR Training — Entry Point.

Implements the INTUITOR method (Zhao et al., 2025) using IntuitorTrainer.
Self-certainty is the sole reward signal; external rewards are logged
for comparison but do not affect training.

Usage:
  # Via SLURM (recommended):
  sbatch open_r1_imp/scripts/submit_intuitor.sh
  sbatch open_r1_imp/scripts/submit_intuitor.sh --export=ALL,MODE=lora

  # Single GPU (no vLLM):
  python open_r1_imp/intuitor_train.py --config open_r1_imp/recipes/config_intuitor_clariden.yaml

  # Multi-GPU with vLLM server:
  accelerate launch --config_file open_r1_imp/accelerate_configs/zero3.yaml \
    --num_processes 3 open_r1_imp/intuitor_train.py \
    --config open_r1_imp/recipes/config_intuitor_clariden.yaml
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_r1_imp.configs_intuitor import IntuitorGRPOConfig, IntuitorScriptArguments
from open_r1_imp.intuitor_trainer import IntuitorTrainer
from open_r1_imp.rewards_multi import get_multi_reward_funcs
from open_r1_imp.utils import get_dataset, get_model, get_tokenizer
from open_r1_imp.hub_callback import HubPushCallback
from open_r1_imp.eval_callback import AccuracyEvalCallback
from open_r1_imp.checkpoint_prune_callback import PruneOptimizerStateCallback

logger = logging.getLogger(__name__)


def _build_reward_funcs(script_args, training_args):
    """Build reward functions for logging (zero-weighted, don't affect training)."""
    # get_multi_reward_funcs reads 'thinking' via getattr — pass training_args
    # so it picks up the thinking flag from IntuitorGRPOConfig
    script_args.thinking = getattr(training_args, "thinking", False)
    multi_rewards = get_multi_reward_funcs(script_args, training_args.num_generations)

    reward_funcs = []
    for name in script_args.reward_funcs:
        if name in multi_rewards:
            reward_funcs.append(multi_rewards[name])
        else:
            raise ValueError(
                f"Unknown reward function: '{name}'. "
                f"Available: {list(multi_rewards.keys())}"
            )

    logger.info(f"Reward functions (logged only, zero-weighted): "
                f"{[f.__name__ for f in reward_funcs]}")
    return reward_funcs


def main(script_args, training_args, model_args):
    """Main training function."""
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"Model: {model_args}")
    logger.info(f"Script args: {script_args}")
    logger.info(f"Training args: {training_args}")

    # Checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected: {last_checkpoint}")

    # Load dataset
    dataset = get_dataset(script_args)

    # Load tokenizer and model
    tokenizer = get_tokenizer(model_args, training_args)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Build reward functions (logged only — zero-weighted)
    reward_funcs = _build_reward_funcs(script_args, training_args)

    # Format dataset into conversation
    def make_conversation(example):
        prompt_column = script_args.dataset_prompt_column
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        if prompt_column not in example:
            raise ValueError(f"Dataset column not found: '{prompt_column}'. "
                           f"Available: {list(example.keys())}")
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Build callbacks
    callbacks = []
    if training_args.save_strategy != "no":
        keep_full = getattr(training_args, "keep_full_checkpoints", 1)
        callbacks.append(PruneOptimizerStateCallback(keep_full=keep_full))
        logger.info(f"Checkpoint pruning enabled: keeping {keep_full} fully-resumable "
                    f"checkpoint(s), stripping optimizer state from older ones")
    if training_args.push_to_hub:
        hub_push_steps = getattr(training_args, "hub_push_every_n_steps", 50)
        run_name = getattr(training_args, "run_name", None)
        callbacks.append(HubPushCallback(
            push_every_n_steps=hub_push_steps,
            run_name=run_name,
        ))
        logger.info(f"Hub push enabled: every {hub_push_steps} steps")

    eval_acc_steps = getattr(training_args, "eval_accuracy_steps", 0)
    if eval_acc_steps > 0:
        eval_dataset_name = getattr(training_args, "eval_accuracy_dataset", "math500")
        eval_samples = getattr(training_args, "eval_accuracy_samples", 50)
        thinking = getattr(training_args, "thinking", False)
        callbacks.append(AccuracyEvalCallback(
            tokenizer=tokenizer,
            eval_every_n_steps=eval_acc_steps,
            dataset_name=eval_dataset_name,
            max_samples=eval_samples,
            max_new_tokens=training_args.max_completion_length,
            thinking=thinking,
        ))
        logger.info(f"Accuracy eval enabled: every {eval_acc_steps} steps on "
                    f"{eval_dataset_name} ({eval_samples} samples)")

    # ── Initialize IntuitorTrainer ──
    trainer = IntuitorTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── Train ──
    logger.info("*** Training with INTUITOR (self-certainty only) ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ── Save model ──
    logger.info("*** Saving model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            dataset_name=script_args.dataset_name,
            tags=["intuitor", "self-certainty", "rlif"],
        )
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("INTUITOR training complete!")


if __name__ == "__main__":
    parser = TrlParser((IntuitorScriptArguments, IntuitorGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
