"""Unit tests for reward components (no torch required)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_boxed():
    from rewards.answer_extraction import extract_boxed_answer
    assert extract_boxed_answer("\\boxed{42}") == "42"
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed_answer("no box") is None
    assert extract_boxed_answer("The answer is \\boxed{3x + 5}") == "3x + 5"
    print("test_boxed: PASS")

def test_normalize():
    from rewards.answer_extraction import normalize_answer, answers_equivalent
    assert normalize_answer("42.0") == "42"
    assert normalize_answer("1/2") == "0.5"
    assert answers_equivalent("42", "42.0")
    assert not answers_equivalent("42", "43")
    print("test_normalize: PASS")

def test_cluster():
    from rewards.cluster_voting import compute_cluster_rewards
    r, i = compute_cluster_rewards(["\\boxed{42}", "\\boxed{42}", "\\boxed{7}", "\\boxed{42}"])
    assert abs(r[0] - 3/4) < 0.01
    assert abs(r[2] - 1/4) < 0.01
    assert i["n_clusters"] == 2
    print("test_cluster: PASS")

def test_cluster_unparseable():
    from rewards.cluster_voting import compute_cluster_rewards
    r, i = compute_cluster_rewards(["\\boxed{42}", "no answer", "\\boxed{42}"])
    assert r[1] == 0.0
    assert i["n_unparseable"] == 1
    print("test_cluster_unparseable: PASS")

def test_format():
    from rewards.format_reward import compute_format_reward
    assert compute_format_reward("\\boxed{42}") == 1.0
    assert compute_format_reward("nothing") == 0.0
    print("test_format: PASS")

def test_self_certainty():
    """Verify KL(U||pi) is non-negative and zero for uniform distribution."""
    import torch
    from rewards.self_certainty import compute_self_certainty_from_logits

    V = 8
    # Uniform logits → KL(U||U) = 0
    uniform_logits = torch.zeros(1, 4, V)
    mask = torch.ones(1, 4)
    score = compute_self_certainty_from_logits(uniform_logits, mask)
    assert abs(score.item()) < 1e-5, f"KL(U||U) should be 0, got {score.item()}"

    # Peaked logits → KL(U||peaked) > 0
    peaked_logits = torch.full((1, 4, V), -10.0)
    peaked_logits[:, :, 0] = 10.0  # strongly peaked on first token
    score_peaked = compute_self_certainty_from_logits(peaked_logits, mask)
    assert score_peaked.item() > 0, f"KL(U||peaked) should be > 0, got {score_peaked.item()}"

    print("test_self_certainty: PASS")


if __name__ == "__main__":
    test_boxed()
    test_normalize()
    test_cluster()
    test_cluster_unparseable()
    test_format()
    test_self_certainty()
    print("\nAll tests passed!")
