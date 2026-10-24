"""Every recipe parses, and the KL-Cov recipes state the loss settings used in the paper."""
import contextlib
import io
from pathlib import Path

import pytest
from trl import ModelConfig, TrlParser

from open_r1_imp.configs import ScriptArguments, TrainConfig

RECIPES = sorted(Path(__file__).resolve().parent.parent.glob("recipes/*/*.yaml"))


def parse(path):
    with contextlib.redirect_stderr(io.StringIO()):
        return TrlParser((ScriptArguments, TrainConfig, ModelConfig)).parse_args_and_config(args=["--config", str(path)])


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_recipe(path):
    script, train, model = parse(path)
    assert not Path(train.output_dir).is_absolute() and "$" not in train.output_dir
    assert train.per_device_train_batch_size % train.num_generations == 0
    if train.beta_cov > 0:  # KL-Cov runs used per-sequence aggregation without vLLM IS correction
        assert train.loss_type == "grpo" and train.vllm_importance_sampling_correction is False
    if train.method == "grpo_gt":
        assert script.reward_funcs == ["accuracy"]
    else:
        assert script.reward_funcs == ["cluster_voting", "boxed_format"]


def test_recipe_count():
    assert len(RECIPES) == 22
