#!/usr/bin/env python3
"""Plot the A-2 (Qwen3-1.7B-Base) greedy pass@1 trajectories for the rebuttal
appendix. Mirrors the paper's trajectory-panel conventions (colors/markers from
evaluation/plot_trajectory.py METHOD_STYLES; grid/legend style from
summarize_passk_trajectory.py; stars mark the per-panel peak among
unsupervised methods, as in Figure fig:trajectory).

Usage:
  python evaluation/plot_trajectory_a2.py \
      --results_dir eval-rebuttal-results \
      --out rebuttal-paper_Multi_Reward_NeurIPS_2026/figures/trajectory_panel_a2_qwen3.pdf
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

# (display label, color, linestyle, marker) — identical roles/colors to the
# 1.5B/3B/7B trajectory figures.
STYLES = {k: _style(k)[:4] for k in ("a2_grpo_gt", "a2_intuitor", "a2_full")}
UNSUPERVISED = ("a2_intuitor", "a2_full")
TASKS = [("gsm8k", "GSM8K"), ("math500", "MATH500")]


def collect(root: Path):
    """-> data[method][task] = [(step, acc%), ...] and base[task] = acc%."""
    data, base = {}, {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d.name.startswith("a2_") or d.name.endswith("_base")):
            continue
        for f in d.glob("*.json"):
            task = f.stem.split("__")[1]
            acc = json.load(open(f))["accuracy"] * 100.0
            if d.name.endswith("_base"):
                if d.name == "qwen3_17b_base":
                    base[task] = acc
                continue
            if "_ckpt" not in d.name:
                continue
            method, step = d.name.rsplit("_ckpt", 1)
            data.setdefault(method, {}).setdefault(task, []).append((int(step), acc))
    for m in data:
        for t in data[m]:
            data[m][t].sort()
    return data, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="eval-rebuttal-results")
    ap.add_argument("--out", default="rebuttal-paper_Multi_Reward_NeurIPS_2026/figures/trajectory_panel_a2_qwen3.pdf")
    args = ap.parse_args()

    data, base = collect(Path(args.results_dir))

    fig, axes = plt.subplots(1, len(TASKS), figsize=(9.0, 3.4), sharex=True)
    for ax, (task, title) in zip(axes, TASKS):
        # base-model reference
        if task in base:
            ax.axhline(base[task], color="#888888", linestyle="--", linewidth=1.0,
                       alpha=0.8, zorder=1)
            ax.annotate(f"base {base[task]:.1f}", xy=(0.985, base[task]),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=7, color="#666666")
        for method in ("a2_grpo_gt", "a2_intuitor", "a2_full"):
            label, color, ls, marker, lw, alpha, z = _style(method)
            pts = data[method][task]
            steps, accs = zip(*pts)
            ax.plot(steps, accs, linestyle=ls, marker=marker, markersize=4,
                    linewidth=lw * 0.75, label=label, color=color, alpha=alpha,
                    zorder=z)
        # star: peak among unsupervised methods
        best = max(((s, a, m) for m in UNSUPERVISED for s, a in data[m][task]),
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
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
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
