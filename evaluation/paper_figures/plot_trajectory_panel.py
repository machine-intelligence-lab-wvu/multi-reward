"""Accuracy-vs-training-step trajectory figure (main paper figure and 7B appendix).

Reads eval JSONs written by scripts/eval.sh / evaluation/run_task.py.

  # main figure: Qwen2.5-1.5B (top row) and Qwen2.5-3B (bottom row)
  python evaluation/paper_figures/plot_trajectory_panel.py --combined \
      --base results/qwen2.5-1.5b --base-3b results/qwen2.5-3b \
      --exclude intuitor_klcov --out figures/trajectory_panel.pdf
  # one model, three tasks
  python evaluation/paper_figures/plot_trajectory_panel.py --panel \
      --base results/qwen2.5-7b --model-label Qwen2.5-7B --panel-tasks gsm8k,math500 --out figures/trajectory_7b.pdf
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Method colors/linestyles come from the shared palette so every figure in the
# paper maps a method to the same appearance (see evaluation/paper_figures/palette.py).
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from palette import label_color as _label_color, line_kw as _line_kw



DEFAULT_BASE = Path(__file__).resolve().parent.parent

# Run names (recipe file names, as written by scripts/eval.sh) in plotting order.
# Older run names are accepted through palette.ALIASES. Missing methods are skipped.
ORDER = ["grpo_gt", "intuitor", "intuitor_klcov", "multi_reward", "multi_reward_klcov",
         "multi_reward_final", "multi_reward_final_klcov05", "grpo_gt_3b", "intuitor_3b",
         "multi_reward_final_3b", "multi_reward_final_klcov05_3b",
         "grpo_gt_7b", "intuitor_7b", "multi_reward_final_klcov05_7b"]
STYLE_BY_METHOD = {k: _label_color(k) for k in ORDER}


def collect(base: Path, task: str):
    """Return {method: [(step, acc), ...]} sorted by step."""
    out = defaultdict(list)
    pat = re.compile(r"^(?P<method>.+)_ckpt(?P<step>\d+)$")
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        method, step = m["method"], int(m["step"])
        if method not in STYLE_BY_METHOD:
            continue
        f = d / f"{d.name}__{task}.json"
        if not f.exists():
            continue
        acc = json.loads(f.read_text())["accuracy"]
        out[method].append((step, acc))
    for method in out:
        out[method].sort(key=lambda t: t[0])
    return out


def _parse_exclude(exclude: str):
    return {m.strip() for m in exclude.split(",") if m.strip()}


def _plot_single_task(base: Path, task: str, out_path: Path, model_label: str, excluded):
    data = collect(base, task)
    if not data:
        raise SystemExit(f"no JSONs found under {base}")

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=150)
    for method in ORDER:
        if method in excluded:
            continue
        pts = data.get(method, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        best = max(ys)
        best_step = xs[ys.index(best)]
        label, color = STYLE_BY_METHOD[method]
        ax.plot(xs, ys, color=color, **_line_kw(method, markersize=5), label=f"{label} (peak {best:.3f} @ step {best_step})")

    ax.set_xlabel("Training step", fontsize=16)
    ax.set_ylabel(f"{task} accuracy (pass@1, greedy)", fontsize=16)
    ax.set_title(f"Trajectory accuracy vs training step - {model_label} ({task})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=13)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def _plot_panel(base: Path, tasks, out_path: Path, model_label: str, excluded):
    title_fs       = 22
    axlabel_fs     = 19
    tick_fs        = 16
    legend_fs      = 18
    star_legend_fs = 13

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0 * len(tasks), 6.6), dpi=170,
                             sharex=True, gridspec_kw={"wspace": 0.22})
    if len(tasks) == 1:
        axes = [axes]

    legend_items = {}

    for i, task in enumerate(tasks):
        ax = axes[i]
        data = collect(base, task)
        if not data:
            ax.set_title(f"{task} (no data)")
            ax.axis("off")
            continue

        # Per-subplot global peak across methods.
        best_acc = -1.0
        best_step = None
        best_label = None
        best_color = None

        for method in ORDER:
            if method in excluded:
                continue
            pts = data.get(method, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label, color = STYLE_BY_METHOD[method]
            line, = ax.plot(xs, ys, color=color, **_line_kw(method, markersize=6), label=label)
            if label not in legend_items:
                legend_items[label] = line
            # Exclude GRPO-GT from the peak comparison (it uses ground truth).
            if method.startswith("grpo_gt"):
                continue
            local_best = max(ys)
            if local_best > best_acc:
                best_acc = local_best
                best_step = xs[ys.index(local_best)]
                best_label = label
                best_color = color

        # Faded dotted vertical line at step 160 (behind data lines).
        ax.axvline(160, color="#888888", linestyle=":", linewidth=2.0,
                   alpha=0.55, zorder=0)

        # Star at the per-subplot peak; small in-axes legend (bottom-left) for it.
        if best_step is not None:
            star = ax.plot([best_step], [best_acc], marker="*", markersize=18,
                                color="gold", markeredgecolor="black",
                           markeredgewidth=1.0, linestyle="None",
                           zorder=5)[0]
            # Smaller marker just for the legend, so the in-axes star block is compact.
            star_legend_handle = plt.Line2D([], [], marker="*", markersize=11,
                                            color="gold", markeredgecolor="black",
                                            markeredgewidth=0.8, linestyle="None")
            ax.legend(handles=[star_legend_handle], labels=[best_label],
                      loc="best", fontsize=star_legend_fs,
                      frameon=True, handletextpad=0.3, borderpad=0.35,
                      labelspacing=0.2)

        ax.set_title(task.upper(), fontsize=title_fs)
        ax.set_xlabel("Training step", fontsize=axlabel_fs)
        if i == 0:
            ax.set_ylabel("Accuracy (pass@1, greedy)", fontsize=axlabel_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        # y=0.055 reproduces the published "tight2" crop (~1487x788 pt) rather
        # than leaving the ~50 pt legend gap the default anchor produces.
        # "upper center" at y=0 puts the legend's top edge at the bottom of the
        # figure box, so it can never collide with the x tick labels / xlabels;
        # bbox_inches="tight" then crops away the unused band.
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(5, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, 0.0), fontsize=legend_fs)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    sibling = out_path.with_suffix(".pdf" if out_path.suffix == ".png" else ".png")
    fig.savefig(sibling, bbox_inches="tight")
    print(f"wrote {out_path} and {sibling}")


def _plot_combined_panel(bases, tasks, out_path: Path, model_labels, excluded):
    """2-row panel: rows = (1.5B, 3B), cols = tasks. Single shared legend at bottom,
    per-subplot peak-star legend inside each subplot, contrastive vline at step 160."""
    title_fs       = 24
    axlabel_fs     = 21
    rowlabel_fs    = 22
    tick_fs        = 17
    legend_fs      = 20
    star_legend_fs = 14

    n_rows, n_cols = len(bases), len(tasks)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7.0 * n_cols, 5.8 * n_rows), dpi=170,
                             sharex=True,
                             gridspec_kw={"wspace": 0.22, "hspace": 0.22})
    if n_rows == 1:
        axes = [axes]

    legend_items = {}

    for r, base in enumerate(bases):
        for c, task in enumerate(tasks):
            ax = axes[r][c]
            data = collect(base, task)
            if not data:
                ax.set_title(f"{task} (no data)")
                ax.axis("off")
                continue

            best_acc = -1.0
            best_step = None
            best_label = None
            best_color = None

            for method in ORDER:
                if method in excluded:
                    continue
                pts = data.get(method, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                label, color = STYLE_BY_METHOD[method]
                line, = ax.plot(xs, ys, color=color, **_line_kw(method, markersize=6), label=label)
                if label not in legend_items:
                    legend_items[label] = line
                if method.startswith("grpo_gt"):
                    continue
                local_best = max(ys)
                if local_best > best_acc:
                    best_acc = local_best
                    best_step = xs[ys.index(local_best)]
                    best_label = label
                    best_color = color

            # Contrastive vertical line at step 160.
            ax.axvline(160, color="#1a1a1a", linestyle=(0, (4, 3)), linewidth=2.4,
                       alpha=0.85, zorder=1)

            if best_step is not None:
                ax.plot([best_step], [best_acc], marker="*", markersize=18,
                        color="gold", markeredgecolor="black",
                        markeredgewidth=0.9, linestyle="None", zorder=9)
                star_legend_handle = plt.Line2D([], [], marker="*", markersize=11,
                                                color="gold", markeredgecolor="black",
                                                markeredgewidth=0.8, linestyle="None")
                ax.legend(handles=[star_legend_handle], labels=[best_label],
                          loc="best", fontsize=star_legend_fs,
                          frameon=True, handletextpad=0.3, borderpad=0.35,
                          labelspacing=0.2)

            if r == 0:
                ax.set_title(task.upper(), fontsize=title_fs)
            if r == n_rows - 1:
                ax.set_xlabel("Training step", fontsize=axlabel_fs)
            if c == 0:
                ax.set_ylabel(f"{model_labels[r]}\nAccuracy (pass@1, greedy)",
                              fontsize=axlabel_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, None)

    if legend_items:
        labels = list(legend_items.keys())
        handles = [legend_items[k] for k in labels]
        # y=0.055 reproduces the published "tight2" crop (~1487x788 pt) rather
        # than leaving the ~50 pt legend gap the default anchor produces.
        # "upper center" at y=0 puts the legend's top edge at the bottom of the
        # figure box, so it can never collide with the x tick labels / xlabels;
        # bbox_inches="tight" then crops away the unused band.
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(5, len(labels)), frameon=True,
                   bbox_to_anchor=(0.5, 0.0),
                   prop={"weight": "bold", "size": legend_fs})

    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.07)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    sibling = out_path.with_suffix(".pdf" if out_path.suffix == ".png" else ".png")
    fig.savefig(sibling, bbox_inches="tight")
    print(f"wrote {out_path} and {sibling}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="Trajectory root directory containing method_ckptXXX folders")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--out",  default=None)
    ap.add_argument("--model-label", default="Qwen2.5-1.5B")
    ap.add_argument("--panel", action="store_true",
                    help="Create one figure with multiple tasks in a row and one shared legend")
    ap.add_argument("--panel-tasks", default="gsm8k,math500,mmlu_pro",
                    help="Comma-separated tasks for --panel mode")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated method keys to skip, e.g. intuitor_klcov,intuitor_klcov_3b")
    ap.add_argument("--combined", action="store_true",
                    help="Produce a 2-row combined panel for two model bases")
    ap.add_argument("--base-3b", default=None,
                    help="Second trajectory root (typically the 3B run) for --combined")
    ap.add_argument("--model-label-3b", default="Qwen2.5-3B",
                    help="Row label for the second base in --combined")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    excluded = _parse_exclude(args.exclude)

    if args.combined:
        if not args.base_3b:
            raise SystemExit("--combined requires --base-3b")
        base2 = Path(args.base_3b).resolve()
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        out_path = Path(args.out) if args.out else base / "plots" / "trajectory_panel_combined.png"
        # Main-text figure only: "Multi-reward + KL-Cov (ours)" instead of the beta_cov value,
        # which isn't explained in this figure's caption.
        for k in ("multi_reward_klcov", "multi_reward_final_klcov05", "multi_reward_final_klcov05_3b"):
            if k in STYLE_BY_METHOD:
                STYLE_BY_METHOD[k] = ("Multi-reward + KL-Cov (ours)", STYLE_BY_METHOD[k][1])
        _plot_combined_panel([base, base2], tasks, out_path,
                             [args.model_label, args.model_label_3b], excluded)
        return

    if args.panel:
        tasks = [t.strip() for t in args.panel_tasks.split(",") if t.strip()]
        out_path = Path(args.out) if args.out else base / "plots" / "trajectory_panel_gsm8k_math500_mmlu_pro.png"
        _plot_panel(base, tasks, out_path, args.model_label, excluded)
        return

    out_path = Path(args.out) if args.out else base / "plots" / f"trajectory_{args.task}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_single_task(base, args.task, out_path, args.model_label, excluded)


if __name__ == "__main__":
    main()
