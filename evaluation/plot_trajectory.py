"""Plot GSM8K pass@1 vs. training step across methods.

Reads {output_dir}/{method}_ckpt{step}/{method}_ckpt{step}__gsm8k.json for
every method and step in TRAJECTORY_METHODS, and emits:
  - {output_dir}/SUMMARY_trajectory.csv   (long format)
  - paper/figures/gsm8k_pass1_trajectory.pdf
  - paper/figures/gsm8k_pass1_trajectory.png   (for README)

Usage:
  python3 evaluation/plot_trajectory.py \
      --output_dir evaluation_results/trajectory_v1 \
      --figure_dir paper/figures
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


# Map method short name → (display label, color, linestyle)
METHOD_STYLES = {
    "grpo_gt":              ("GRPO-GT (ground truth)",        "#000000", "-",  "o"),
    "intuitor":             ("INTUITOR (SC only)",            "#d62728", "-",  "s"),
    "intuitor_klcov":       ("INTUITOR + KL-Cov",             "#ff7f0e", "--", "s"),
    "multi_reward":         ("CIRCA w/o KL-Cov",             "#1f77b4", "-",  "^"),
    "multi_reward_klcov05": ("CIRCA ($\\beta_{cov}$=0.05)",     "#2ca02c", "-",  "^"),
    "multi_reward_klcov02": ("CIRCA ($\\beta_{cov}$=0.02)",     "#17becf", "--", "^"),
    # 3B / 7B trajectory runs (post-fix)
    "grpo_gt_3b":                    ("GRPO-GT (ground truth)",       "#000000", "-",  "o"),
    "intuitor_3b":                   ("INTUITOR (SC only)",           "#d62728", "-",  "s"),
    "multi_reward_final_3b":         ("CIRCA w/o KL-Cov",            "#1f77b4", "-",  "^"),
    "multi_reward_final_klcov05_3b": ("CIRCA ($\\beta_{cov}$=0.05)",    "#2ca02c", "-",  "^"),
    "grpo_gt_7b":                    ("GRPO-GT (ground truth)",       "#000000", "-",  "o"),
    "intuitor_7b":                   ("INTUITOR (SC only)",           "#d62728", "-",  "s"),
    "multi_reward_final_klcov05_7b": ("CIRCA ($\\beta_{cov}$=0.05)",    "#2ca02c", "-",  "^"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="evaluation_results/trajectory_v1")
    p.add_argument("--figure_dir", default="paper/figures")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--exclude", default="",
                   help="Comma-separated list of method short-names to drop from the plot")
    p.add_argument("--figure_suffix", default="",
                   help="String appended to the output filename stem, e.g. '_3b' -> gsm8k_pass1_trajectory_3b.pdf")
    p.add_argument("--model_label", default="Qwen2.5-1.5B",
                   help="Model label used in the plot title")
    return p.parse_args()


def _collect(root: Path, task: str):
    """Walk {root}/{method}_ckpt{step}/*.json and collect (method, step, pass@1)."""
    out = defaultdict(list)  # method -> [(step, pass@1), ...]
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        m = re.match(r"^(.*)_ckpt(\d+)$", run_dir.name)
        if not m:
            continue
        method, step = m.group(1), int(m.group(2))
        j = run_dir / f"{run_dir.name}__{task}.json"
        if not j.exists():
            continue
        try:
            d = json.loads(j.read_text())
            acc = d.get("pass@1", d.get("accuracy"))
            if acc is not None:
                out[method].append((step, float(acc)))
        except Exception as e:
            print(f"  skip {j}: {e}")
    for m in out:
        out[m].sort()
    return out


def _write_csv(data, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "pass@1"])
        for method, points in data.items():
            for step, acc in points:
                w.writerow([method, step, f"{acc:.4f}"])


def _plot(data, task: str, fig_dir: Path, suffix: str = "", model_label: str = "Qwen2.5-1.5B"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PDF/PNG plots.")
        print("  pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Draw in a stable order so colours match our legend order
    order = list(METHOD_STYLES.keys())
    for method in order:
        if method not in data:
            continue
        label, color, linestyle, marker = METHOD_STYLES[method]
        steps = [p[0] for p in data[method]]
        accs  = [p[1] * 100 for p in data[method]]  # to percent
        ax.plot(steps, accs, linestyle=linestyle, marker=marker, markersize=4,
                linewidth=1.8, label=label, color=color, alpha=0.9)
    # Plot any extras not in METHOD_STYLES
    for method in data:
        if method in METHOD_STYLES:
            continue
        steps = [p[0] for p in data[method]]
        accs  = [p[1] * 100 for p in data[method]]
        ax.plot(steps, accs, linewidth=1.5, label=method, alpha=0.7)

    ax.set_xlabel("Training step")
    ax.set_ylabel(f"{task.upper()} pass@1 (%)")
    ax.set_title(f"{task.upper()} accuracy over training ({model_label})")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="best", framealpha=0.95)
    ax.set_ylim(bottom=0)

    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / f"{task}_pass1_trajectory{suffix}.pdf"
    png = fig_dir / f"{task}_pass1_trajectory{suffix}.png"
    fig.tight_layout()
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"Saved: {pdf}\nSaved: {png}")


def main():
    args = parse_args()
    root = Path(args.output_dir)
    if not root.exists():
        print(f"{root} does not exist"); return

    data = _collect(root, args.task)
    if not data:
        print(f"No trajectory JSONs found under {root} for task={args.task}"); return

    # Apply exclude list
    excluded = {m.strip() for m in args.exclude.split(",") if m.strip()}
    if excluded:
        before = set(data.keys())
        data = {m: v for m, v in data.items() if m not in excluded}
        dropped = before - set(data.keys())
        if dropped:
            print(f"Excluded methods: {sorted(dropped)}")

    # Summary
    print(f"\nMethods found: {len(data)}")
    for m in sorted(data):
        steps = [p[0] for p in data[m]]
        pretty = METHOD_STYLES.get(m, (m,))[0]
        print(f"  {pretty:40s}  {len(steps):3d} pts  range=[{min(steps)}, {max(steps)}]")

    _write_csv(data, root / "SUMMARY_trajectory.csv")
    print(f"Saved: {root / 'SUMMARY_trajectory.csv'}")

    _plot(data, args.task, Path(args.figure_dir),
          suffix=args.figure_suffix, model_label=args.model_label)


if __name__ == "__main__":
    main()
