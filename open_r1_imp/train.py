"""Training entry point for all methods (multi_reward, intuitor, grpo_gt).

    accelerate launch --config_file accelerate_configs/zero3.yaml \
        --num_processes 3 --gradient_accumulation_steps 14 \
        -m open_r1_imp.train --config recipes/qwen2.5-1.5b/multi_reward_klcov.yaml

Prefer scripts/train.sh, which also starts the vLLM generation server. Any recipe
field can be overridden on the command line, e.g. `--output_dir /tmp/run --max_steps 10`.
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser, get_peft_config

from open_r1_imp.checkpoint_prune_callback import PruneOptimizerStateCallback
from open_r1_imp.configs import ScriptArguments, TrainConfig
from open_r1_imp.grpo_gt_trainer import GRPOGTTrainer
from open_r1_imp.intuitor_trainer import IntuitorTrainer
from open_r1_imp.multi_reward_trainer import MultiRewardTrainer
from open_r1_imp.rewards import get_reward_funcs
from open_r1_imp.utils import get_dataset, get_model, get_tokenizer

logger = logging.getLogger(__name__)

TRAINERS = {"multi_reward": MultiRewardTrainer, "intuitor": IntuitorTrainer, "grpo_gt": GRPOGTTrainer}


def main(script_args: ScriptArguments, training_args: TrainConfig, model_args: ModelConfig):
    set_seed(training_args.seed)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    training_args.output_dir = os.path.expandvars(training_args.output_dir)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from {last_checkpoint}")

    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args)
    model = get_model(model_args)
    reward_funcs = get_reward_funcs(script_args.reward_funcs, training_args.num_generations)

    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example[script_args.dataset_prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    callbacks = []
    if training_args.save_strategy != "no":
        callbacks.append(PruneOptimizerStateCallback(keep_full=training_args.keep_full_checkpoints))

    trainer = TRAINERS[training_args.method](
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    logger.info(f"*** Training method={training_args.method} beta_cov={training_args.beta_cov} "
                f"loss_type={training_args.loss_type} num_iterations={training_args.num_iterations} ***")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    logger.info(f"Done. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainConfig, ModelConfig))
    main(*parser.parse_args_and_config())
