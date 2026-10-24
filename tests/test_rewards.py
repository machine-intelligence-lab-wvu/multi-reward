"""Reward functions (open_r1_imp/rewards.py)."""
import pytest

from open_r1_imp.rewards import (boxed_format_reward, cluster_voting_reward, compute_cluster_rewards,
                                 get_reward_funcs, accuracy_reward)


def c(text):  # TRL conversational completion
    return [{"role": "assistant", "content": text}]


def test_cluster_reward_is_cluster_fraction_over_parseable():
    rewards = compute_cluster_rewards([r"\boxed{3}", r"\boxed{3}", r"\boxed{4}", "no answer"])
    assert rewards == pytest.approx([2 / 3, 2 / 3, 1 / 3, 0.0])


def test_cluster_reward_numeric_normalization():
    rewards = compute_cluster_rewards([r"\boxed{0.5}", r"\boxed{\frac{1}{2}}", r"\boxed{1/2}", r"\boxed{1,000}"])
    assert rewards[:3] == pytest.approx([0.75] * 3)


def test_cluster_voting_groups_are_consecutive_chunks():
    comps = [c(r"\boxed{1}"), c(r"\boxed{1}"), c(r"\boxed{2}"), c(r"\boxed{2}")]
    assert cluster_voting_reward(comps, num_generations=2) == [1.0, 1.0, 1.0, 1.0]
    assert cluster_voting_reward(comps, num_generations=4) == [0.5, 0.5, 0.5, 0.5]


def test_format_reward():
    assert boxed_format_reward([c(r"\boxed{7}"), c(r"\boxed{"), c("7")]) == [1.0, 0.5, 0.0]


def test_accuracy_reward_uses_math_verify():
    sols = [r"$\boxed{\frac{1}{2}}$", r"$\boxed{\frac{1}{2}}$"]
    assert accuracy_reward([c(r"so \boxed{0.5}"), c(r"\boxed{2}")], solution=sols) == [1.0, 0.0]


def test_registry():
    assert len(get_reward_funcs(["cluster_voting", "boxed_format", "accuracy"], 7)) == 3
    with pytest.raises(ValueError):
        get_reward_funcs(["not_a_reward"], 7)
