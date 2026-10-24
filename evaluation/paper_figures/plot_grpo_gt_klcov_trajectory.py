#!/usr/bin/env python3
"""GRPO-GT vs GRPO-GT + KL-Cov on Qwen2.5-1.5B (appendix figure).

  python evaluation/paper_figures/plot_grpo_gt_klcov_trajectory.py \
      --results_dir results/qwen2.5-1.5b --out figures/trajectory_grpo_gt_klcov.pdf

The base model scores 1.7 / 9.2 (GSM8K / MATH500) under this zero-shot boxed
protocol, far below the plotted range, so no base reference line is drawn.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared palette so these appendix figures match the main-text ones exactly.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent / "paper_figures"))
from palette import style as _style

# (display label, color, linestyle, marker) — GRPO-GT keeps the paper's black,
# the KL-Cov-augmented arm uses the blue reserved for auxiliary variants.
METHODS = ("grpo_gt_noklcov", "grpo_gt_klcov")
OLD_NAMES = {"a4_noklcov": "grpo_gt_noklcov", "a4_klcov": "grpo_gt_klcov"}
STYLES = {k: _style(k)[:4] for k in METHODS}
TASKS = [("gsm8k", "GSM8K"), ("math500", "MATH500")]


def collect(root: Path):
    """-> data[method][task] = [(step, acc%), ...]."""
    data = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "_ckpt" not in d.name:
            continue
        method, step = d.name.rsplit("_ckpt", 1)
        method = OLD_NAMES.get(method, method)
        if method not in METHODS:
            continue
        for f in d.glob("*.json"):
            task = f.stem.split("__")[1]
            acc = json.load(open(f))["accuracy"] * 100.0
            data.setdefault(method, {}).setdefault(task, []).append((int(step), acc))
    for m in data:
        for t in data[m]:
            data[m][t].sort()
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/qwen2.5-1.5b")
    ap.add_argument("--out", default="figures/trajectory_grpo_gt_klcov.pdf")
    args = ap.parse_args()

    data = collect(Path(args.results_dir))

    fig, axes = plt.subplots(1, len(TASKS), figsize=(9.0, 3.4), sharex=True)
    for ax, (task, title) in zip(axes, TASKS):
        for method in METHODS:
            label, color, ls, marker, lw, alpha, z = _style(method)
            pts = data[method][task]
            steps, accs = zip(*pts)
            ax.plot(steps, accs, linestyle=ls, marker=marker, markersize=4,
                    linewidth=lw * 0.75, label=label, color=color, alpha=alpha,
                    zorder=z)
        # star: per-panel peak across both arms
        best = max(((s, a, m) for m in STYLES for s, a in data[m][task]),
                   key=lambda x: x[1])
        ax.scatter([best[0]], [best[1]], marker="*", s=170, color="gold",
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(f"{best[1]:.1f} (step {best[0]})", xy=(best[0], best[1]),
                    xytext=(6, 7), textcoords="offset points", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training step", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(list(range(20, 201, 40)))
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("pass@1 (%)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {out} and {png}")


if __name__ == "__main__":
    main()
