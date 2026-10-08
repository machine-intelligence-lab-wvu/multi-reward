"""
Multi-Reward Unsupervised GRPO Training with KL-Cov (TRL, single GPU).

RTX 4090 (24GB): Qwen3-1.7B + LoRA.

MultiRewardGRPOTrainer (subclass of GRPOTrainer) integrates ALL proposal components:
  1. CLUSTER VOTING (answer-level): EMPO-style via TRL reward_funcs
  2. SELF-CERTAINTY (reasoning-level): KL(U||pi) from forward-pass entropy
  3. GDPO NORMALIZATION: per-channel normalize, then weighted combine
  4. KL-COV (token-level): targeted covariance-aware regularization

TRL handles: rollout generation, tokenization, checkpointing, PEFT, logging.
We override: _compute_loss only.

COMPATIBILITY: Tested against TRL 0.16-0.23+. Pin trl==0.21.3 if unsure.
"""
import os, sys, logging, argparse, math, inspect, glob, json
from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from rewards.answer_extraction import extract_boxed_answer, normalize_answer
from rewards.cluster_voting import compute_cluster_rewards
from rewards.format_reward import compute_format_reward
from callbacks.klcov import compute_token_covariances, select_high_covariance_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiRewardGRPOTrainer(GRPOTrainer):

    def __init__(self, w_cluster_format=0.5, w_selfcert=0.5, beta_cov=0.1,
                 top_k_fraction=0.02, print_rollouts_every=1, entropy_bonus=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.w_cluster_format = w_cluster_format
        self.w_selfcert = w_selfcert
        self.beta_cov = beta_cov
        self.top_k_fraction = top_k_fraction ##for Kl Cov, fraction of tokens to apply penalty to based on covariance ranking
        self.print_rollouts_every = print_rollouts_every
        self.entropy_bonus = entropy_bonus ## weight for entropy bonus to encourage exploration (added to per-token loss)
        self._generation_count = 0
        self._cached_vocab_size = self._resolve_vocab_size()

        G = kwargs.get("args", None)
        if G is not None:
            G = getattr(G, "num_generations", None)
        if G is not None and G <= 2 and w_cluster_format > 0:
            logger.warning(
                f"num_generations={G} ≤ 2: cluster voting will ALWAYS have zero "
                f"variance within groups (all rollouts get the same reward). "
                f"The cluster channel contributes NO gradient signal. "
                f"Use num_generations ≥ 4 for cluster voting to be effective."
            )
        logger.info(f"MultiRewardGRPO: w_cf={w_cluster_format}, w_sc={w_selfcert}, "
                     f"beta_cov={beta_cov}, top_k={top_k_fraction}, "
                     f"entropy_bonus={entropy_bonus}, V={self._cached_vocab_size}")
        
        self._last_rewards_per_func = None  # intercepted from _calculate_rewards

        # Only log these metrics to wandb (filter out TRL's noisy defaults)
        self._tracked_metrics = {
            "loss", "grad_norm", "learning_rate",
            "rewards/cluster_mean", "rewards/self_certainty",
            "advantages/combined_mean", "advantages/combined_std",
            "kl", "entropy",
            "klcov/high_cov_fraction", "klcov/mean_covariance", "klcov/mean_kl_penalty"
        }

        # Validate reward function ordering — raw_cluster_rewards uses index 0
        reward_funcs = kwargs.get("reward_funcs", [])
        if reward_funcs and reward_funcs[0] is not cluster_reward_fn:
            raise ValueError(
                "reward_funcs[0] must be cluster_reward_fn. "
                "GDPO ch_cf relies on rewards_per_func[:, 0] being cluster rewards. "
                f"Got: {reward_funcs[0].__name__}"
            )

    def log(self, logs, *args, **kwargs):
        """Filter logs to only track important metrics in wandb/console."""
        filtered = {k: v for k, v in logs.items()
                    if k in self._tracked_metrics or k in ("epoch", "step")}
        super().log(filtered, *args, **kwargs)

    def _calculate_rewards(self, *args, **kwargs):
        """Override to intercept per-function rewards before TRL combines them.

        TRL's _generate_and_score_completions uses rewards_per_func as a local
        variable and doesn't include it in the output dict. We capture it here
        so _generate_and_score_completions can inject it into the result dict
        for correct sub-batching by split_tensor_dict.
        """
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rewards_per_func.detach().clone()
        return rewards_per_func

    def _forward_with_mean_logprobs(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Forward pass that returns per-token logps, entropy, AND mean log-probs
        across the vocabulary (needed for KL(U||pi) in Eq. 2).

        TRL's _get_per_token_logps_and_entropies only returns entropy H(pi),
        which gives KL(pi||U)=log|V|-H. The paper's Eq. 2 requires
        KL(U||pi)=-log|V|-(1/|V|)*sum_v log pi(v), so we need mean(log_probs).

        CRITICAL: In a causal LM, logits[:, t, :] predicts token at position t+1.
        To get P(completion_token_t | context), we need logits at position t-1.
        TRL uses logits[:, -(C+1):-1, :] to get the C logits that predict each
        completion token. Using [:, -C:, :] is OFF-BY-ONE and causes KL explosion.

        Memory optimization: avoids materializing full (B, S, V) log_probs AND
        probs simultaneously. Computes per_token_logps and mean_log_probs from
        logits + logsumexp directly, then computes entropy chunk-wise.
        Peak memory: logits (B,S,V) + one chunk of probs — saves ~2.4GB vs naive.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # CORRECT indexing: logits[:, -(C+1):-1, :] predicts completion tokens
        logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]

        # CRITICAL: TRL temperature-scales logits before computing log-probs
        # (see _get_per_token_logps_and_entropies line 1022). We MUST match this
        # so that per_token_logps and old_per_token_logps use the same distribution.
        # Without this, the importance sampling ratio exp(logps - old_logps) is garbage.
        logits = logits / self.temperature

        B, S, V = logits.shape

        # logsumexp once — everything derives from this
        lse = logits.logsumexp(dim=-1)  # (B, S)

        # Per-token log-probs for chosen tokens (no full log_probs materialized)
        completion_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1) - lse

        # Mean log-probs: mean_v(log pi(v)) = mean_v(logits_v) - logsumexp
        # (needed for KL(U||pi) in Eq. 2)
        mean_log_probs = logits.mean(dim=-1) - lse

        # Entropy H(pi) = logsumexp - E_pi[logits], computed chunk-wise over V
        # to avoid materializing full (B, S, V) probs tensor
        CHUNK = 32768  # process ~32K vocab entries at a time
        weighted_sum = torch.zeros(B, S, device=logits.device, dtype=logits.dtype)
        for v_start in range(0, V, CHUNK):
            v_end = min(v_start + CHUNK, V)
            chunk_logits = logits[:, :, v_start:v_end]
            chunk_probs = torch.exp(chunk_logits - lse.unsqueeze(-1))
            weighted_sum += (chunk_probs * chunk_logits).sum(dim=-1)
            del chunk_probs, chunk_logits
        entropies = lse - weighted_sum  # H(pi) = log Z - E_pi[z]
        del lse, weighted_sum

        return per_token_logps, entropies, mean_log_probs

    def _generate_and_score_completions(self, inputs):
        """Override to pre-compute GDPO advantages over the full batch.

        CRITICAL: GDPO z-scoring MUST happen here, before TRL's shuffle+split.
        TRL shuffles completions then splits into sub-batches for gradient
        accumulation. With per_device_train_batch_size=1, each sub-batch has
        ~0-1 completions per prompt group → z-scoring in _compute_loss would
        always return 0 (no variance within groups).

        By computing advantages here over all B completions and injecting them
        into the result dict, TRL's split_tensor_dict slices them correctly
        into each sub-batch.
        """
        result = super()._generate_and_score_completions(inputs)
        self._generation_count += 1

        # Inject raw cluster rewards for logging in _compute_loss
        if self._last_rewards_per_func is not None and self._last_rewards_per_func.shape[1] >= 1:
            result["raw_cluster_rewards"] = self._last_rewards_per_func[:, 0]

        # === PRE-COMPUTE GDPO ADVANTAGES OVER FULL BATCH ===
        B = result["completion_ids"].shape[0]
        G = self.num_generations
        device = result["completion_ids"].device
        group_idx = torch.arange(B, device=device) // G
        result["prompt_group_idx"] = group_idx

        # Channel 1: Cluster rewards z-scored per prompt group
        raw_cluster = result.get("raw_cluster_rewards")
        if raw_cluster is not None:
            ch_cf = self._gdpo_normalize(raw_cluster.to(device), group_idx)
        else:
            ch_cf = self._gdpo_normalize(result["advantages"].to(device), group_idx)

        # Channel 2: Self-certainty KL(U||pi) z-scored per prompt group
        # Requires forward pass for mean_log_probs — done chunked + no_grad
        self_certainty = self._compute_batch_self_certainty(result)
        ch_sc = self._gdpo_normalize(self_certainty, group_idx)

        # Combined GDPO advantages (Eq. 4)
        result["gdpo_advantages"] = self.w_cluster_format * ch_cf + self.w_selfcert * ch_sc

        # Log full-batch advantage stats here (sub-batches in _compute_loss are size 1,
        # so per-sub-batch std is meaningless)
        mode = "train" if self.model.training else "eval"
        gdpo_adv = result["gdpo_advantages"]
        self._metrics[mode]["advantages/combined_std"].append(
            gdpo_adv.std(correction=0).item() if gdpo_adv.numel() > 1 else 0.0)

        logger.info(f"GDPO advantages: mean={gdpo_adv.mean():.6f}, "
                     f"std={gdpo_adv.std():.6f}, "
                     f"ch_cf_std={ch_cf.std():.6f}, ch_sc_std={ch_sc.std():.6f}, "
                     f"sc_raw_min={self_certainty.min():.4f}, sc_raw_max={self_certainty.max():.4f}, "
                     f"cluster_unique={raw_cluster.unique().tolist() if raw_cluster is not None else 'N/A'}")

        if self.print_rollouts_every > 0 and self._generation_count % self.print_rollouts_every == 0:
            try:
                self._log_rollouts(inputs, result)
            except Exception as e:
                logger.warning(f"Rollout logging failed: {e}")

        return result

    def _compute_batch_self_certainty(self, result):
        """Compute KL(U||pi) for all completions in the full batch.

        Done with torch.no_grad() in chunks to avoid OOM on a single GPU.
        This is an extra forward pass but necessary for correct GDPO z-scoring
        before TRL splits the batch. Generation (vLLM) dominates wall time,
        so this adds ~5-10% overhead.
        """
        prompt_ids = result["prompt_ids"]
        prompt_mask = result["prompt_mask"]
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        B = input_ids.shape[0]
        CHUNK = 4  # 4 sequences at a time to stay within 24GB VRAM
        log_V = math.log(self._cached_vocab_size)

        self_certainty = torch.zeros(B, device=input_ids.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for start in range(0, B, CHUNK):
                end = min(start + CHUNK, B)
                chunk_ids = input_ids[start:end]
                chunk_attn = attention_mask[start:end]
                chunk_mask = completion_mask[start:end]

                outputs = self.model(input_ids=chunk_ids, attention_mask=chunk_attn)
                logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
                lse = logits.logsumexp(dim=-1)
                mean_lp = logits.mean(dim=-1) - lse

                kl_uniform = (-log_V - mean_lp) * chunk_mask
                lengths = chunk_mask.sum(dim=-1).clamp(min=1)
                self_certainty[start:end] = kl_uniform.sum(dim=-1) / lengths

                del outputs, logits, lse, mean_lp, kl_uniform

        if was_training:
            self.model.train()

        return self_certainty

    def _log_rollouts(self, inputs, result):
        """Pretty-print a sample of rollouts with their rewards."""
        tokenizer = self.processing_class
        G = self.args.num_generations

        # Decode prompts and completions
        prompt_ids = result.get("prompt_ids", None)
        completion_ids = result.get("completion_ids", None)
        if prompt_ids is None or completion_ids is None:
            return

        # Get rewards from result
        rewards = result.get("rewards", None)  # (B,) or per-reward-fn

        B = completion_ids.shape[0]
        n_prompts = B // G

        # Show first prompt's group (all G rollouts)
        show_prompts = min(2, n_prompts)

        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  ROLLOUTS (generation #{self._generation_count})")
        lines.append("=" * 80)

        for p_idx in range(show_prompts):
            start = p_idx * G
            # Decode prompt (just once per group)
            prompt_text = tokenizer.decode(prompt_ids[start], skip_special_tokens=True)
            # Truncate long prompts for display
            if len(prompt_text) > 300:
                prompt_text = prompt_text[:150] + " ... " + prompt_text[-100:]

            lines.append(f"\n{'─' * 70}")
            lines.append(f"PROMPT [{p_idx}]: {prompt_text}")
            lines.append(f"{'─' * 70}")

            for g in range(min(G, 4)):  # Show up to 4 rollouts per prompt
                idx = start + g
                comp_text = tokenizer.decode(completion_ids[idx], skip_special_tokens=True)
                # Truncate for display
                if len(comp_text) > 500:
                    comp_text = comp_text[:250] + "\n  ... [truncated] ...\n  " + comp_text[-200:]

                # Get per-rollout reward
                reward_str = ""
                if rewards is not None:
                    if rewards.dim() == 1:
                        reward_str = f"  R={rewards[idx].item():.3f}"
                    elif rewards.dim() == 2:
                        reward_str = "  R=[" + ", ".join(
                            f"{rewards[idx, j].item():.3f}" for j in range(rewards.shape[1])
                        ) + "]"

                # Extract boxed answer for display
                from rewards.answer_extraction import extract_boxed_answer
                boxed = extract_boxed_answer(comp_text)
                boxed_str = f"  \\boxed{{{boxed}}}" if boxed else "  [no \\boxed{}]"

                terminated = len(tokenizer.encode(comp_text)) < self.args.max_completion_length
                term_str = "✓ terminated" if terminated else f"✗ hit max ({self.args.max_completion_length})"

                lines.append(f"\n  ROLLOUT [{g}]{reward_str}  {term_str}{boxed_str}")
                lines.append(f"  {comp_text}")

            if G > 4:
                lines.append(f"\n  ... ({G - 4} more rollouts not shown)")

        lines.append(f"\n{'=' * 80}")

        # Print all at once to avoid interleaving with progress bar
        logger.info("\n".join(lines))

    @staticmethod
    def _gdpo_normalize(rewards, group_idx, eps=1e-8):
        """Z-score normalize rewards within prompt groups (Eq. 3).

        Args:
            rewards: (B,) tensor of per-completion rewards
            group_idx: (B,) tensor of prompt group indices — completions with
                the same index belong to the same prompt. This is shuffle-safe:
                TRL's shuffle_sequence_dict reorders completions, but
                prompt_group_idx is shuffled along with them.
        """
        normalized = torch.zeros_like(rewards)
        for gid in group_idx.unique():
            mask = group_idx == gid
            group = rewards[mask]
            if group.numel() <= 1:
                continue
            mu = group.mean()
            sigma = group.std(correction=0)
            if sigma > eps:
                normalized[mask] = (group - mu) / (sigma + eps)
        return normalized

    def _resolve_vocab_size(self):
        model = self.model
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            return model.config.vocab_size
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()
            if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
                return base.config.vocab_size
        if hasattr(self, "processing_class"):
            pc = self.processing_class
            if hasattr(pc, "vocab_size"): return pc.vocab_size
            if hasattr(pc, "__len__"): return len(pc)
        logger.warning("Could not determine vocab_size, using 151936 (Qwen3)")
        return 151936

    def _get_grad_accum_steps(self):
        # TRL >= 0.22: current_gradient_accumulation_steps (issue #3983)
        if hasattr(self, "current_gradient_accumulation_steps"):
            return self.current_gradient_accumulation_steps
        return self.args.gradient_accumulation_steps

    def _aggregate_loss(self, per_token_loss, mask):
        """Match TRL's loss_type aggregation (grpo/dapo/dr_grpo/bnpo)."""
        loss_type = getattr(self, "loss_type", "grpo")
        if loss_type == "grpo":
            return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
        elif loss_type in ("dapo", "cispo", "bnpo"):
            return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
        elif loss_type == "dr_grpo":
            return (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.args.max_completion_length)
        else:
            return ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()

    def _compute_loss(self, model, inputs):
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        trl_advantages = inputs["advantages"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        mask = completion_mask
        if "tool_mask" in inputs:
            mask = completion_mask * inputs["tool_mask"]

        # STEP 1: Forward pass (custom — also returns mean_log_probs for Eq. 2)
        per_token_logps, entropies, mean_log_probs = self._forward_with_mean_logprobs(
            model, input_ids, attention_mask, logits_to_keep,
        )

        # KL-Cov anchor: old_per_token_logps = log π_θ_old (the generation policy).
        # TRL 0.28 provides old_per_token_logps from the scoring pass.
        # Fallback chain: old_per_token_logps → ref_per_token_logps → detached current.
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            ref_logps = inputs.get("ref_per_token_logps")
            if ref_logps is not None:
                old_per_token_logps = ref_logps
            else:
                old_per_token_logps = per_token_logps.detach()

        # STEP 2: Self-certainty via KL(U || pi_theta) per Eq. 2 (for logging)
        log_V = math.log(self._cached_vocab_size)
        kl_uniform_pi = (-log_V - mean_log_probs) * mask
        lengths = mask.sum(dim=-1).clamp(min=1)
        self_certainty = kl_uniform_pi.sum(dim=-1) / lengths

        # STEP 3: Use pre-computed GDPO advantages from _generate_and_score_completions.
        # Z-scoring was done over the FULL batch (all G completions per prompt)
        # before TRL's shuffle+split, so each sub-batch gets correct values.
        raw_cluster = inputs.get("raw_cluster_rewards")
        advantages = inputs.get("gdpo_advantages")
        if advantages is None:
            # Fallback: compute in sub-batch (inaccurate with small sub-batches)
            group_idx = inputs.get("prompt_group_idx")
            if group_idx is None:
                G = self.args.num_generations
                B = self_certainty.shape[0]
                group_idx = torch.arange(B, device=self_certainty.device) // G
            raw_cluster = inputs.get("raw_cluster_rewards")
            if raw_cluster is not None:
                ch_cf = self._gdpo_normalize(raw_cluster.to(self_certainty.device), group_idx)
            else:
                ch_cf = self._gdpo_normalize(trl_advantages.detach(), group_idx)
            ch_sc = self._gdpo_normalize(self_certainty.detach(), group_idx)
            advantages = self.w_cluster_format * ch_cf + self.w_selfcert * ch_sc

        # DEBUG: verify advantages are non-zero
        if advantages.abs().max() < 1e-10:
            gdpo_present = "gdpo_advantages" in inputs
            logger.warning(f"ZERO advantages in _compute_loss! "
                           f"gdpo_key_present={gdpo_present}, "
                           f"adv_shape={advantages.shape}, "
                           f"adv_vals={advantages[:4].tolist()}")

        # STEP 4: KL vs reference
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        if self.beta != 0.0 and ref_per_token_logps is not None:
            per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps) - 1)
        elif self.beta != 0.0:
            per_token_kl = (torch.exp(per_token_logps.detach() - per_token_logps)
                            - (per_token_logps.detach() - per_token_logps) - 1)
        else:
            per_token_kl = torch.zeros_like(per_token_logps)

        # STEP 5: KL-Cov
        covariances = compute_token_covariances(per_token_logps.detach(), advantages, mask)
        high_cov_mask = select_high_covariance_tokens(covariances, mask, self.top_k_fraction)
        normal_mask = mask * (1.0 - high_cov_mask)

        # STEP 6: Per-token loss
        # log(π_θ / π_θ_old) — the true importance sampling ratio.
        # When num_iterations=1: π_θ = π_θ_old, ratio ≈ 1, clipping is a no-op.
        # When num_iterations>1: π_θ drifts, clipping constrains divergence.
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)

        adv = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages

        # Normal tokens: clipped surrogate (Eq. 21)
        clipped_ratio = torch.clamp(ratio,
                                    1.0 - self.epsilon_low,
                                    1.0 + self.epsilon_high)
        clipped_loss = -torch.min(ratio * adv, clipped_ratio * adv)

        # High-cov tokens: unclipped + KL-Cov penalty (Eq. 8-9)
        unclipped_loss = -ratio * adv
        kl_penalty = self.beta_cov * log_ratio.abs()
        penalized_loss = unclipped_loss + kl_penalty

        per_token_loss = clipped_loss * normal_mask + penalized_loss * high_cov_mask
        per_token_loss = per_token_loss + self.beta * per_token_kl

        # Entropy bonus: -alpha * H(pi) encourages exploration.
        # When entropy is low (model too peaked), this provides gradient signal
        # even when advantages are all zero (identical completions).
        if self.entropy_bonus > 0:
            per_token_loss = per_token_loss - self.entropy_bonus * entropies

        loss = self._aggregate_loss(per_token_loss, mask)
        loss = loss / self._get_grad_accum_steps()

        # STEP 7: Metrics
        mode = "train" if model.training else "eval"
        n_high = high_cov_mask.sum().item()
        n_total = mask.sum().item()

        # Rewards
        self._metrics[mode]["rewards/self_certainty"].append(self_certainty.mean().item())
        if raw_cluster is not None:
            self._metrics[mode]["rewards/cluster_mean"].append(raw_cluster.mean().item())

        # Advantages (std is logged in _generate_and_score_completions over the full batch)
        self._metrics[mode]["advantages/combined_mean"].append(advantages.mean().item())

        # KL & entropy
        if self.beta != 0.0:
            masked_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(masked_kl).nanmean().item())
        mean_entropy = (entropies * mask).sum() / mask.sum().clamp(min=1)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item())

        # KL-Cov
        self._metrics[mode]["klcov/high_cov_fraction"].append(n_high / max(n_total, 1))
        self._metrics[mode]["klcov/mean_covariance"].append(
            (covariances * mask).sum().item() / max(n_total, 1))
        if n_high > 0:
            self._metrics[mode]["klcov/mean_kl_penalty"].append(
                (kl_penalty * high_cov_mask).sum().item() / n_high)

        return loss


# ============================================================
# REWARD FUNCTIONS
# ============================================================

def cluster_reward_fn(prompts, completions, **kwargs):
    num_gen = kwargs.get("num_generations", 8)
    N = len(completions)
    texts = []
    for comp in completions:
        if isinstance(comp, list):
            texts.append(comp[0]["content"] if comp else "")
        elif isinstance(comp, dict):
            texts.append(comp.get("content", ""))
        else:
            texts.append(str(comp))
    rewards = [0.0] * N
    for start in range(0, N, num_gen):
        end = min(start + num_gen, N)
        group_rewards, _ = compute_cluster_rewards(texts[start:end])
        for i, r in enumerate(group_rewards):
            rewards[start + i] = r

    print(f"\n\nCluster rewards for first group: {rewards[:num_gen]}")

    return rewards


def format_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        if isinstance(comp, list):
            text = comp[0]["content"] if comp else ""
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        rewards.append(compute_format_reward(text))
    return rewards


# ============================================================
# DATASET
# ============================================================

def prepare_dataset(tokenizer, dataset_name, split="train", max_samples=None):
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        question_key = "question"

    elif dataset_name == "math":
        try: ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=split)
        except: print("Hendrycks-MATH-benchmark not FOUND")
        question_key = "problem"

    elif dataset_name == "numina":
        ds = load_dataset("AI-MO/NuminaMath-CoT", split=split, trust_remote_code=True)
        question_key = "problem"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    records = []
    for ex in ds:
        q = ex.get(question_key, ex.get("question", ""))
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        # CRITICAL: Qwen3 with enable_thinking=True generates <think>...</think> blocks
        # that consume the entire max_completion_length (512 tokens) without ever
        # producing \boxed{} answers. This causes ALL rewards to be 0.0, making
        # training impossible. Use enable_thinking=False so the model directly
        # produces step-by-step reasoning with \boxed{} answers.
        #
        # Once the model learns to produce good answers (rewards > 0), you can
        # optionally switch to enable_thinking=True with a larger
        # max_completion_length (2048+) in a second training phase.
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        records.append({"prompt": prompt})
    return Dataset.from_list(records)


# ============================================================
# POST-TRAINING METRIC PLOTS
# ============================================================

def plot_training_metrics(log_history, output_dir):
    """Generate and save training metric plots from HF Trainer log_history.

    Args:
        log_history: list of dicts from trainer.state.log_history
        output_dir: directory to save plots and raw metrics JSON
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping metric plots. "
                       "Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save raw log history as JSON for later analysis
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(log_history, f, indent=2, default=str)
    logger.info(f"Raw metrics saved → {metrics_file}")

    # Collect all metric keys (excluding non-numeric and meta keys)
    skip_keys = {"epoch", "step", "total_flos", "train_runtime",
                 "train_samples_per_second", "train_steps_per_second",
                 "train_loss"}
    all_keys = set()
    for entry in log_history:
        for k, v in entry.items():
            if k not in skip_keys and isinstance(v, (int, float)):
                all_keys.add(k)

    if not all_keys:
        logger.warning("No numeric metrics found in log_history — skipping plots.")
        return

    # Group metrics by prefix for multi-panel figures
    groups = defaultdict(list)
    for k in sorted(all_keys):
        prefix = k.split("/")[0] if "/" in k else "general"
        groups[prefix].append(k)

    # Extract step-value pairs for each metric
    def get_series(key):
        steps, vals = [], []
        for entry in log_history:
            if key in entry and "step" in entry:
                steps.append(entry["step"])
                vals.append(entry[key])
        return steps, vals

    # Plot each group as a separate figure
    plot_files = []
    for group_name, keys in groups.items():
        n = len(keys)
        cols = min(n, 2)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
        fig.suptitle(f"Training Metrics: {group_name}", fontsize=14, fontweight="bold")

        for idx, key in enumerate(keys):
            ax = axes[idx // cols][idx % cols]
            steps, vals = get_series(key)
            if steps:
                ax.plot(steps, vals, linewidth=1.2)
                ax.set_title(key, fontsize=10)
                ax.set_xlabel("Step")
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{key} (no data)", fontsize=10)

        # Hide unused subplots
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"metrics_{group_name}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(fname)

    # Also make a single summary figure with key metrics
    summary_keys = [
        "loss", "grad_norm",
        "rewards/cluster_mean", "rewards/self_certainty", 
        "advantages/combined_mean", "advantages/combined_std",
        "kl", "entropy",
        "klcov/high_cov_fraction", "klcov/mean_covariance", "klcov/mean_kl_penalty",
        "learning_rate",
    ]
    available = [k for k in summary_keys if any(k in e for e in log_history)]
    if available:
        n = len(available)
        cols = min(n, 2)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
        fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
        for idx, key in enumerate(available):
            ax = axes[idx // cols][idx % cols]
            steps, vals = get_series(key)
            if steps:
                ax.plot(steps, vals, linewidth=1.2, color="tab:blue")
                ax.set_title(key, fontsize=10)
                ax.set_xlabel("Step")
                ax.grid(True, alpha=0.3)
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        plt.tight_layout()
        summary_file = os.path.join(output_dir, "training_summary.png")
        fig.savefig(summary_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(summary_file)

    logger.info(f"Saved {len(plot_files)} metric plots → {output_dir}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Reward GRPO with KL-Cov (TRL)")
    p.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math", "numina"])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--w_cluster", type=float, default=0.5)
    p.add_argument("--w_format", type=float, default=0.1)
    p.add_argument("--w_cluster_format", type=float, default=0.5)
    p.add_argument("--w_selfcert", type=float, default=0.5)
    p.add_argument("--beta_cov", type=float, default=0.1)
    p.add_argument("--top_k_fraction", type=float, default=0.02)
    p.add_argument("--entropy_bonus", type=float, default=0.01,
                    help="Entropy bonus coefficient. Encourages exploration when model "
                         "is too peaked (low entropy → identical completions → zero advantages).")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--loss_type", default="grpo", choices=["grpo", "dapo", "dr_grpo", "bnpo"])
    p.add_argument("--beta_kl", type=float, default=0.04)
    p.add_argument("--output_dir", default="./outputs/multi_reward_grpo_trl")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--run_name", default="multi_reward_grpo_trl")
    p.add_argument("--print_rollouts_every", type=int, default=1,
                    help="Print sample rollouts every N generation steps (0=disable)")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info(f"Args: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_dataset(tokenizer, args.dataset, max_samples=args.max_train_samples)
    logger.info(f"Training samples: {len(train_dataset)}")

    # Determine run number: output_dir/run_0, run_1, etc.
    os.makedirs(args.output_dir, exist_ok=True)
    existing_runs = glob.glob(os.path.join(args.output_dir, "run_*"))
    run_numbers = [int(r.split("_")[-1]) for r in existing_runs
                   if r.split("_")[-1].isdigit()]
    run_number = max(run_numbers) + 1 if run_numbers else 0
    run_output_dir = os.path.join(args.output_dir, f"run_{run_number}")
    logger.info(f"Saving to {run_output_dir}")

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules="all-linear", task_type="CAUSAL_LM", lora_dropout=0.05,
    )

    grpo_kwargs = dict(
        output_dir=run_output_dir, run_name=args.run_name,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature, top_p=1.0,
        beta=args.beta_kl,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        report_to="wandb" if args.use_wandb else "none",
        reward_weights=[args.w_cluster, args.w_format],
    )
    if "loss_type" in inspect.signature(GRPOConfig.__init__).parameters:
        grpo_kwargs["loss_type"] = args.loss_type
    # Enable TRL's built-in completion logging if available
    if "log_completions" in inspect.signature(GRPOConfig.__init__).parameters:
        grpo_kwargs["log_completions"] = args.print_rollouts_every > 0

    training_args = GRPOConfig(**grpo_kwargs)

    trainer = MultiRewardGRPOTrainer(
        w_cluster_format=args.w_cluster_format,
        w_selfcert=args.w_selfcert,
        beta_cov=args.beta_cov,
        top_k_fraction=args.top_k_fraction,
        print_rollouts_every=args.print_rollouts_every,
        entropy_bonus=args.entropy_bonus,
        model=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[cluster_reward_fn, format_reward_fn],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    final_dir = os.path.join(run_output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved → {final_dir}")

    # Plot training metrics
    plot_training_metrics(trainer.state.log_history, run_output_dir)


if __name__ == "__main__":
    main()
