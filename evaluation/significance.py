"""Statistical rigor pass for ICLR-TODO T0.2.

Computes, from the per-item records saved in eval-rebuttal-results/:
  * Wilson 95% confidence intervals for every run/checkpoint/task accuracy;
  * exact McNemar tests (two-sided binomial on discordant pairs) for paired
    method comparisons at matched checkpoints.

No GPU, no checkpoints needed - only the eval JSONs already in the repo.

Usage:
    python evaluation/significance.py [--results-dir eval-rebuttal-results]
                                      [--out-csv eval-rebuttal-results/significance_all.csv]

Outputs a full CSV plus a markdown summary (printed to stdout) for the
checkpoints referenced in the paper.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Paired comparisons: (run_a, run_b, label). Positive McNemar direction = a beats b.
COMPARISONS = [
    ("a2_full", "a2_intuitor", "Qwen3-1.7B: ours vs INTUITOR"),
    ("a2_full", "a2_grpo_gt", "Qwen3-1.7B: ours vs GRPO-GT (sup.)"),
    ("a2_intuitor", "a2_grpo_gt", "Qwen3-1.7B: INTUITOR vs GRPO-GT (sup.)"),
    ("a4_klcov", "a4_noklcov", "Qwen2.5-1.5B: GRPO-GT+KL-Cov vs GRPO-GT"),
]

RUN_RE = re.compile(r"^(?P<run>.+?)_ckpt(?P<ckpt>\d+)$")


def wilson_ci(correct, n, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _comb(n, k):
    """math.comb fallback for Python < 3.8 (exact big-int)."""
    if hasattr(math, "comb"):
        return math.comb(n, k)
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value.

    b = items A correct & B wrong; c = items A wrong & B correct.
    Under H0 discordant pairs are Binomial(n, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k), capped at 1. Exact big-int arithmetic.
    tail = sum(_comb(n, i) for i in range(k + 1))
    p = 2 * tail / (2 ** n)
    return min(1.0, p)


def load_results(results_dir):
    """-> {(run, ckpt, task): {'acc': float, 'n': int, 'items': {id: 0/1}}}
    ckpt is int, or -1 for base models (no _ckptNNN suffix)."""
    out = {}
    for d in sorted(Path(results_dir).iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        run, ckpt = (m.group("run"), int(m.group("ckpt"))) if m else (d.name, -1)
        for f in sorted(d.glob("*.json")):
            task = f.stem.split("__")[-1]
            data = json.loads(f.read_text())
            recs = data.get("records")
            if not isinstance(recs, list):
                continue
            items = {r["id"]: int(round(float(r["pass@1"]))) for r in recs}
            out[(run, ckpt, task)] = {
                "acc": float(data["accuracy"]),
                "n": len(items),
                "items": items,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="eval-rebuttal-results")
    ap.add_argument("--out-csv", default="eval-rebuttal-results/significance_all.csv")
    args = ap.parse_args()

    res = load_results(args.results_dir)

    # ---- 1. Wilson CIs for every run/ckpt/task --------------------------------
    ci_rows = []
    for (run, ckpt, task), r in sorted(res.items()):
        lo, hi = wilson_ci(sum(r["items"].values()), r["n"])
        ci_rows.append({
            "run": run, "ckpt": ckpt, "task": task, "n": r["n"],
            "acc": round(100 * r["acc"], 2),
            "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2),
        })

    # ---- 2. McNemar for paired comparisons at matched (ckpt, task) -----------
    mc_rows = []
    by_run = defaultdict(dict)
    for (run, ckpt, task), r in res.items():
        by_run[run][(ckpt, task)] = r
    for run_a, run_b, label in COMPARISONS:
        keys = sorted(set(by_run[run_a]) & set(by_run[run_b]))
        for (ckpt, task) in keys:
            ra, rb = by_run[run_a][(ckpt, task)], by_run[run_b][(ckpt, task)]
            ids = sorted(set(ra["items"]) & set(rb["items"]))
            if len(ids) != len(ra["items"]) or len(ids) != len(rb["items"]):
                print(f"WARNING: id mismatch {run_a} vs {run_b} @ ckpt{ckpt} {task}: "
                      f"{len(ids)} shared of {len(ra['items'])}/{len(rb['items'])}")
            b = sum(1 for i in ids if ra["items"][i] and not rb["items"][i])
            c = sum(1 for i in ids if not ra["items"][i] and rb["items"][i])
            p = mcnemar_exact(b, c)
            mc_rows.append({
                "comparison": label, "task": task, "ckpt": ckpt, "n": len(ids),
                "acc_a": round(100 * sum(ra["items"][i] for i in ids) / len(ids), 2),
                "acc_b": round(100 * sum(rb["items"][i] for i in ids) / len(ids), 2),
                "a_only": b, "b_only": c,
                "p_exact": f"{p:.2e}" if p < 1e-3 else round(p, 4),
            })

    # ---- write CSV ------------------------------------------------------------
    out = Path(args.out_csv)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# Wilson 95% CIs"])
        w.writerow(ci_rows[0].keys())
        for r in ci_rows:
            w.writerow(r.values())
        w.writerow([])
        w.writerow(["# Exact McNemar (paired, discordant-pair binomial)"])
        w.writerow(mc_rows[0].keys())
        for r in mc_rows:
            w.writerow(r.values())
    print(f"Wrote {out} ({len(ci_rows)} CI rows, {len(mc_rows)} McNemar rows)\n")

    # ---- markdown summary: final checkpoints ----------------------------------
    print("## Wilson 95% CIs at step 200 (and base models)\n")
    print("| run | task | acc % | 95% CI |")
    print("|---|---|---|---|")
    for r in ci_rows:
        if r["ckpt"] in (200, -1):
            print(f"| {r['run']} | {r['task']} | {r['acc']} | [{r['ci_lo']}, {r['ci_hi']}] |")

    print("\n## Exact McNemar at step 200\n")
    print("| comparison | task | acc A | acc B | A-only | B-only | p (exact) |")
    print("|---|---|---|---|---|---|---|")
    for r in mc_rows:
        if r["ckpt"] == 200:
            print(f"| {r['comparison']} | {r['task']} | {r['acc_a']} | {r['acc_b']} "
                  f"| {r['a_only']} | {r['b_only']} | {r['p_exact']} |")


if __name__ == "__main__":
    main()
