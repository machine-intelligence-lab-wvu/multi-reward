# NeurIPS 2026 Rebuttal — Run Plan (Bridges-2)

**Deadline:** all results posted **before Aug 3** (Phase 2 end). Today = Jul 24 → ~10 days.
**Compute:** ~1,879 GPU-SU left; full program below ≈ **530 SU + evals** → not the binding constraint.
**Hardware:** default job shape = **2×H100-80** (GPU 0 = `trl vllm-serve`, GPU 1 = training; grad-accum 42 keeps effective batch 882; ~3× the 3-GPU wall-clock but schedules far faster on the jammed queue). `GPUS=4` restores the wide shape (GPU 0 vLLM + 3×ZeRO-3, grad-accum 14); **B6 (3B) requires GPUS=4**; `A3_corewarding` is forced to 2-GPU (view pairing needs one training process). flash-attn needs Ampere+ → **H100/L40S only, never V100** (no H200 exists on Bridges-2; `GPU_TYPE=l40s-48` targets the L40S pool).
**All estimates below are PROVISIONAL** until the 20-step calibration (job 42585008) returns real min/step.

Priority order (from the AC meta-review: G2/G3/G4 need *experiments*, not clarifications). Protect coverage in this order if compute/time is cut: **A-2 → A-3 → A-4 → A-5**, then Tier B.

---

## Priority 1 — A-2: newer generation (Qwen3-1.7B-Base)  ·  covers G2, x-W1, x-Q2, 21m6-Wk
The single most-named experiment (1xmu named Qwen3). Full vs INTUITOR vs GRPO-GT, MATH, 200 steps.

| Config id | Method / entry | Readiness | ~wall | ~SU |
|---|---|---|---|---|
| `A2_qwen3_17b_full` | multi-reward + KL-Cov (multi_reward_train) | ✅ ready | ~7.5h | 30 |
| `A2_qwen3_17b_intuitor` | INTUITOR (intuitor_train) | ✅ ready | ~7.5h | 30 |
| `A2_qwen3_17b_grpo_gt` | GRPO-GT accuracy (grpo_gt_train) | ✅ ready | ~7.5h | 30 |

```bash
bash open_r1_imp/scripts/rebuttal/launch.sh A2_qwen3_17b_full
bash open_r1_imp/scripts/rebuttal/launch.sh A2_qwen3_17b_intuitor
bash open_r1_imp/scripts/rebuttal/launch.sh A2_qwen3_17b_grpo_gt
```

## Priority 2 — A-3: recent self-rewarding baseline head-to-head (1.5B)  ·  covers G3, G1, q-W1, x-W2, x-W3
| Config id | Method | Readiness | ~wall | ~SU |
|---|---|---|---|---|
| `A3_empo_qwen25_15b` | EMPO (semantic-cluster consensus) | ✅ ready — faithful¹ | ~7.5h | 30 |
| `A3_corewarding_qwen25_15b` | Co-rewarding-I (faithful port) | ✅ ready² | ~22h @2-GPU | ~45 |

¹ Faithful to the official repo (`QingyangZhang/EMPO`): reward = `cluster_size / n_generations`, intrinsic entropy filter `entropy ≥ log(G)` zeroes all-disagree groups. DeBERTa NLI downloads on first use (numeric answers use the fast string path).
² **Faithful port of `tmlr-group/Co-rewarding` (arXiv:2508.00410), Co-rewarding-I:** uses the authors' **released** Qwen3-32B rephrasings (`Co-rewarding-I/data/math/train_pairs.parquet`, downloaded to `${MR_DATA_DIR}/corewarding_src/`), interleaved into a paired two-view dataset by `scripts/rebuttal/make_corewarding_dataset.py` (15,000 rows = 7,500 uid pairs). The `co_rewarding` reward ports their logic verbatim: `majority_vote` (their exact function), last-`\boxed{}` extraction (their `last_boxed_only_string`+`remove_boxed`), swapped pseudo-labels (ori scored against aug-majority and vice versa), binary reward `1.0 iff ans == pseudo != ''`. **Pairing requirements** (all enforced): `shuffle_dataset: false`, pairs adjacent, `grad_accum=42` baked, and launch.sh **forces the 2-GPU / single-training-process shape** so both views of a uid share one reward batch (a runtime warning counts any missing counterparts — must be 0). Documented deviations: 7 rollouts/view (suite standard; theirs 8), our shared system prompt (controlled comparison); their lr 3e-6 / KL 0.005 / max_completion 3072 already match ours.

```bash
bash open_r1_imp/scripts/rebuttal/launch.sh A3_empo_qwen25_15b
bash open_r1_imp/scripts/rebuttal/launch.sh A3_corewarding_qwen25_15b
```

## Priority 3 — A-4: RLVR-augmentation batch (1.5B)  ·  covers G3, G4, q-W1/W2/Q1/Q3, x-Q1
| Config id | What | Readiness | ~wall | ~SU |
|---|---|---|---|---|
| `A4_rlvr_majorityvote_qwen25_15b` | GRPO-GT + majority-vote aux reward | ✅ ready | ~7.5h | 30 |
| `A4_grpo_gt_klcov_qwen25_15b` | GRPO-GT + KL-Cov | ✅ **ready (KL-Cov ported)** | ~7.5h | 30 |
| ~~`A4_rlvr_entropy`~~ | GRPO-GT + entropy bonus | ❌ removed (deferred; entropy not ported) | — | — |

KL-Cov is now ported into `grpo_gt_trainer.py` (`beta_cov`/`top_k_fraction`; `beta_cov=0` = unchanged base behavior). The entropy-bonus port for `A4_rlvr_entropy` was intentionally **not** done — leave that config unused unless you decide to add it.

```bash
bash open_r1_imp/scripts/rebuttal/launch.sh A4_rlvr_majorityvote_qwen25_15b
bash open_r1_imp/scripts/rebuttal/launch.sh A4_grpo_gt_klcov_qwen25_15b
```

## Priority 4 — A-5: non-verifiable / open-ended (1.5B)  ·  covers G4 (deepest), G2-onlyMATH, q-Q4, x-Q3
| Config id | What | Readiness | ~wall | ~SU |
|---|---|---|---|---|
| `A5_nonverif_ours_qwen25_15b` | ours: semantic-cluster (empo) on open-ended prompts | ⚠️ **DRAFT⁴** | ~7.5h | 30 |
| `A5_nonverif_intuitor_qwen25_15b` | single-reward baseline on same prompts | ⚠️ **DRAFT⁴** | ~7.5h | 30 |

⁴ Two open items — see **Code work #3**: (a) final open-ended **dataset** (placeholder = `tatsu-lab/alpaca`), and (b) an **LLM-judge eval harness** (none exists in `evaluation/tasks/`). Training is config-only; scoring is the new part. Costliest/deepest — if it must be cut, scope honestly to the AC (fallback C-10).

---

## Tier B — depth (do if time/GPUs allow)

| Config id | What | Covers | Readiness | ~SU |
|---|---|---|---|---|
| `B7_multireward_globalentropy_qwen25_15b` | targeted (KL-Cov) vs global entropy on OUR objective | q-Q1, x-Q1 | ✅ ready | 30 |
| `B6_llama32_3b_full/intuitor/grpo_gt` | **different family** (Llama-3.2-3B) | G2, x-W1/Q2 | ⚠️ gated model⁵ | 48 ea |

⁵ Llama-3.2-3B is **gated on HF** (accept Meta license + `HF_TOKEN`), and its base is weaker at math / boxed-format adherence — run as *supporting* different-family evidence, not headline. Qwen3-1.7B (A-2) covers "newer generation" but is the same family; B-6 is what actually closes the "different family" axis.

```bash
bash open_r1_imp/scripts/rebuttal/launch.sh B7_multireward_globalentropy_qwen25_15b
bash open_r1_imp/scripts/rebuttal/launch.sh B6_llama32_3b_full      # needs HF_TOKEN with Llama access
```

---

## Code work status
1. ✅ **DONE — Co-rewarding-I faithful port** (`A3_corewarding`): verbatim reward-logic port + authors' released rephrase data + paired dataset + pairing enforcement (see A-3 note ²). Unit-tested (swapped-majority, empty-pseudo, missing-counterpart cases). Path chosen: **in-stack faithful port** over running their verl repo — verl's pinned deps would have broken our venv, and their released `train_pairs.parquet` removes the paraphrase-generation risk entirely.
2. ✅ **DONE — KL-Cov in GRPO-GT** (`A4_grpo_gt_klcov`): `beta_cov`/`top_k_fraction` added to `GRPOGTConfig`; KL-Cov `_compute_loss` ported into `grpo_gt_trainer.py` (base behavior unchanged when `beta_cov=0`). Entropy-bonus port (`A4_rlvr_entropy`) intentionally skipped.
3. ⚠️ **TODO — A-5 dataset + LLM-judge** (`A5_*`): pick the open-ended dataset; write an `evaluation/tasks/llm_judge.py`. (~1 day) — only remaining code item.

**Runnable now (11 of 12 rebuttal runs):** all A-2 (3) + A-3 (both) + A-4 (`majorityvote`, `grpo_gt_klcov`) + B-7 + B-6 (3, if HF token). Only **A-5** still needs code (dataset + judge). `A4_rlvr_entropy` is deferred by choice.

---

## Evaluation
After each run, sweep its checkpoints (step 20…200) with the existing tasks — GSM8K, MATH500 (A-2); +MMLU-Pro/LCB/CRUXEval where relevant; LLM-judge for A-5. Eval is 1 GPU, inference-only (~cheap). Port `submit_eval.sh` to Bridges-2 (same module/venv/path swap as `submit_bridges2.sh`).

## Prereqs before launching (once)
- ✅ venv + flash-attn + math-verify installed.
- Calibration job 42585008 → confirms pipeline works on Bridges-2 + real min/step (updates all estimates).
- `wandb login` (or set `WANDB_MODE=offline`); HF cache on `/ocean` (set).
- Pre-download Qwen3-1.7B-Base + (for B-6) accept Llama license & export `HF_TOKEN`.
- Revoke the leaked GitHub + HF tokens.

## Rough totals (provisional)
- **Tier A (required):** 10 runs ≈ 300 SU + evals ≈ **~360 SU**
- **Tier B:** B-7 + 3×B-6 ≈ **~174 SU**
- Grand total well under the 1,879-SU budget → the constraints are the **Aug 3 clock** and the **3 code items**, not compute.
