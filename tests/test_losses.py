"""Loss-level checks of KL-Cov and self-certainty on a tiny model (CPU).

Needs the tiny test model `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` from the
Hugging Face Hub (downloaded once, ~1 MB); skipped when offline.
"""
import pytest
import torch

pytest.importorskip("trl")
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from open_r1_imp.configs import TrainConfig  # noqa: E402
from open_r1_imp.grpo_gt_trainer import GRPOGTTrainer  # noqa: E402
from open_r1_imp.intuitor_trainer import IntuitorTrainer  # noqa: E402
from open_r1_imp.multi_reward_trainer import MultiRewardTrainer  # noqa: E402
from open_r1_imp.rewards import get_reward_funcs  # noqa: E402
from open_r1_imp.utils import aggregate_self_certainty, compute_self_certainty, compute_self_certainty_per_token  # noqa: E402

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
GA = 7
B, P, C = 4, 6, 10


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline
        pytest.skip(f"tiny model unavailable: {e}")


def make(cls, tokenizer, method, rewards, **kw):
    torch.manual_seed(0)
    args = TrainConfig(output_dir="/tmp/mr_test", method=method, per_device_train_batch_size=4, num_generations=4,
                       gradient_accumulation_steps=GA, max_completion_length=16, use_vllm=False, report_to=[],
                       bf16=False, use_cpu=True, beta=0.005, **kw)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "1+1?"}]] * 8, "solution": ["2"] * 8})
    trainer = cls(model=model, reward_funcs=get_reward_funcs(rewards, 4), args=args,
                  train_dataset=ds, processing_class=tokenizer)
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = GA
    return trainer


def batch(old_shift=None):
    g = torch.Generator().manual_seed(1)
    inputs = dict(prompt_ids=torch.randint(0, 1000, (B, P), generator=g), prompt_mask=torch.ones(B, P, dtype=torch.long),
                  completion_ids=torch.randint(0, 1000, (B, C), generator=g),
                  completion_mask=torch.tensor([[1] * 10, [1] * 7 + [0] * 3, [1] * 4 + [0] * 6, [1] * 9 + [0]]),
                  advantages=torch.tensor([1.2, -0.4, 0.9, -1.7]), old_per_token_logps=None)
    return inputs


def grad(trainer, inputs, **attrs):
    for k, v in attrs.items():
        setattr(trainer, k, v)
    trainer.model.zero_grad()
    inputs = dict(inputs)
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
    am = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
    with torch.no_grad():
        logps = trainer._get_per_token_logps_and_entropies(trainer.model, ids, am, C)[0]
    inputs["ref_per_token_logps"] = logps.clone()
    inputs["num_items_in_batch"] = inputs["completion_mask"].sum() * GA
    if inputs.get("old_per_token_logps") == "shifted":
        inputs["old_per_token_logps"] = logps - 0.05 * torch.randn(logps.shape, generator=torch.Generator().manual_seed(2))
    trainer._compute_loss(trainer.model, inputs).backward()
    return torch.cat([p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]).clone()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, 0).item()


@pytest.mark.parametrize("cls,method,rewards", [
    (MultiRewardTrainer, "multi_reward", ["cluster_voting", "boxed_format"]),
    (IntuitorTrainer, "intuitor", ["cluster_voting", "boxed_format"]),
    (GRPOGTTrainer, "grpo_gt", ["accuracy"]),
])
@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "bnpo"])
def test_klcov_with_zero_penalty_equals_trl_loss(tokenizer, cls, method, rewards, loss_type):
    """With a vanishing penalty the KL-Cov loss must equal TRL's loss for the same loss_type
    (same aggregation and gradient-accumulation scaling) for every trainer."""
    t = make(cls, tokenizer, method, rewards, loss_type=loss_type)
    base = grad(t, batch(), beta_cov=0.0)
    klcov = grad(t, batch(), beta_cov=1e-12)
    assert cos(base, klcov) > 0.99999
    assert klcov.norm() / base.norm() == pytest.approx(1.0, rel=1e-4)


def test_klcov_penalty_changes_gradient(tokenizer):
    t = make(MultiRewardTrainer, tokenizer, "multi_reward", ["cluster_voting", "boxed_format"], loss_type="grpo")
    inputs = dict(batch(), old_per_token_logps="shifted")
    assert not torch.equal(grad(t, inputs, beta_cov=1e-12), grad(t, inputs, beta_cov=0.05))


def test_self_certainty_uses_aligned_logits():
    torch.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"tiny model unavailable: {e}")
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)  # make next-token distributions non-uniform
    inp = batch()
    ids = torch.cat([inp["prompt_ids"], inp["completion_ids"]], 1)
    am = torch.cat([inp["prompt_mask"], inp["completion_mask"]], 1)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am).logits
    expected = aggregate_self_certainty(compute_self_certainty_per_token(logits[:, P - 1:P - 1 + C]), inp["completion_mask"])
    got = compute_self_certainty(model, inp["prompt_ids"], inp["prompt_mask"], inp["completion_ids"], inp["completion_mask"])
    assert torch.allclose(got, expected, atol=1e-5)
