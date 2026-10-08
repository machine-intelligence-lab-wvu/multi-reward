"""Regenerate the per-run input CSVs for early_stopping_analysis.py from
eval_results_final/single_multi_reward_ablation/ (wandb exports + SUMMARY.csv).

Outputs evaluation/early_stopping_inputs/{intuitor,cluster_only,multi_reward}_{train,eval}.csv
  *_train.csv: step,length,entropy   (training-time signals)
  *_eval.csv : step,gsm8k,math500    (greedy pass@1, %)
"""

import csv
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..",
                   "eval_results_final", "single_multi_reward_ablation")
OUT = os.path.join(os.path.dirname(__file__), "early_stopping_inputs")

RUNS = {  # output name -> (wandb run prefix, SUMMARY.csv ckpt prefix)
    "cluster_only": ("cluster-only-noklcov-qwen25-1.5b", "cluster_only"),
    "multi_reward": ("multi-reward-final-noklcov-qwen25-1.5b", "multi_reward_final"),
    "intuitor": ("run2_intuitor-qwen25-15b (betakl-0.005)", "intuitor"),
}


def wandb_series(path, metric_suffix):
    """-> {run prefix: {step: value}} for columns '<run> - <metric_suffix>'."""
    out = {w: {} for w, _ in RUNS.values()}
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        cols = {w: f"{w} - {metric_suffix}" for w, _ in RUNS.values()}
        for row in r:
            step = float(row["train/global_step"])
            for w, c in cols.items():
                v = row.get(c, "").strip()
                if v and v.lower() != "nan":
                    out[w][step] = float(v)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    length = wandb_series(
        os.path.join(SRC, "cluster_self-certainty_multi-reward_mean_completion.csv"),
        "train/completions/mean_length")
    entropy = wandb_series(
        os.path.join(SRC, "cluster_self-certainty_multi-reward_mean_entropy.csv"),
        "train/entropy")

    # The ablation SUMMARY stops at ckpt 280; trajectory_postfix_1.5b extends
    # intuitor/multi_reward to 340 (overlapping rows agree; first source wins).
    evals = {k: {} for k in RUNS}
    for summary in [os.path.join(SRC, "SUMMARY.csv"),
                    os.path.join(SRC, "..", "trajectory_postfix_1.5b", "SUMMARY.csv")]:
        with open(summary, newline="") as fh:
            for row in csv.DictReader(fh):
                m = re.match(r"(.*)_ckpt(\d+)$", row["run"])
                if not m:
                    continue
                for name, (_, pref) in RUNS.items():
                    if m.group(1) == pref:
                        evals[name].setdefault(
                            int(m.group(2)),
                            (float(row["gsm8k_pass@1"]) * 100,
                             float(row["math500_pass@1"]) * 100))
    evals = {k: [(s, g, m5) for s, (g, m5) in v.items()] for k, v in evals.items()}

    for name, (w, _) in RUNS.items():
        with open(os.path.join(OUT, f"{name}_train.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["step", "length", "entropy"])
            for step in sorted(length[w]):
                wr.writerow([step, length[w][step], entropy[w].get(step, "")])
        with open(os.path.join(OUT, f"{name}_eval.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["step", "gsm8k", "math500"])
            for s, g, m5 in sorted(evals[name]):
                wr.writerow([s, f"{g:.2f}", f"{m5:.2f}"])
        print(f"{name}: {len(length[w])} train pts, {len(evals[name])} eval ckpts")


if __name__ == "__main__":
    main()
