"""GDPO-normalization ablation figures (Appendix): per-channel GDPO vs combine-then-normalize.

  # accuracy trajectories (GSM8K / MATH500 / MMLU-Pro) from scripts/eval.sh output
  python evaluation/paper_figures/plot_gdpo_ablation.py accuracy --base results/qwen2.5-1.5b \\
      --out figures/gdpo_ablation_accuracy.pdf
  # log-prob-space gradient norm of each reward channel, logged during training
  python evaluation/paper_figures/plot_gdpo_ablation.py gradients --train-dir outputs \\
      --out figures/gdpo_ablation_gradients.pdf
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import style  # noqa: E402
from train_logs import load_metric  # noqa: E402

METHODS = ("gdpo_ablation_per_channel", "gdpo_ablation_combine_then_norm")
TASKS = (("gsm8k", "GSM8K"), ("math500", "MATH500"), ("mmlu_pro", "MMLU-Pro"))
CHANNELS = (("gradcorr_logps/cf_norm", "cluster + format channel"), ("gradcorr_logps/sc_norm", "self-certainty channel"))


def collect_accuracy(base: Path):
    data = defaultdict(lambda: defaultdict(list))
    pat = re.compile(r"^(?P<m>.+)_ckpt(?P<s>\d+)$")
    for d in sorted(base.iterdir()):
        m = pat.match(d.name)
        if not (d.is_dir() and m and m["m"] in METHODS):
            continue
        for task, _ in TASKS:
            f = d / f"{d.name}__{task}.json"
            if f.exists():
                data[m["m"]][task].append((int(m["s"]), 100 * json.loads(f.read_text())["accuracy"]))
    return data


def _line(ax, method, pts, **kw):
    label, color, ls, marker, lw, alpha, z = style(method)
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, lw=lw * 0.75, alpha=alpha, zorder=z, label=label, **kw)


def plot_accuracy(base: Path, out: Path):
    data = collect_accuracy(base)
    if not data:
        raise SystemExit(f"no GDPO-ablation eval JSONs under {base}")
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 3.4))
    for ax, (task, title) in zip(axes, TASKS):
        for method in METHODS:
            if data[method][task]:
                _line(ax, method, data[method][task], markersize=4)
        ax.axvline(160, color="#999999", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("Training step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@1 (%)")
    _finish(fig, axes, out)


def plot_gradients(train_dir: Path, out: Path, stride: int):
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(9, 3.4))
    found = False
    for ax, (key, title) in zip(axes, CHANNELS):
        for method in METHODS:
            pts = load_metric(train_dir, method, key, stride=stride)
            if pts:
                found = True
                _line(ax, method, pts, markersize=0)
        ax.set_yscale("log"); ax.axvline(160, color="#999999", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("Training step"); ax.grid(alpha=0.3)
    if not found:
        raise SystemExit(f"no gradcorr_logps metrics in trainer_state.json under {train_dir}")
    axes[0].set_ylabel(r"$\|\nabla_{\log\pi}\mathcal{L}\|_2$")
    _finish(fig, axes, out)


def _finish(fig, axes, out: Path):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["accuracy", "gradients"])
    ap.add_argument("--base", default="results/qwen2.5-1.5b")
    ap.add_argument("--train-dir", default="outputs")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.what == "accuracy":
        plot_accuracy(Path(args.base), Path(args.out))
    else:
        plot_gradients(Path(args.train_dir), Path(args.out), args.stride)


if __name__ == "__main__":
    main()
