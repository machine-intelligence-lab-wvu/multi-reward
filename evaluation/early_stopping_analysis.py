"""ICLR-TODO T0.1: label-free early-stopping baseline analysis.

Question it answers (Reviewer weakness #2): could one just train a single-signal
method (INTUITOR / cluster-only) and stop before collapse using LABEL-FREE
training signals (mean completion length, token entropy), instead of using the
multi-reward + KL-Cov framework?

The analysis needs NO checkpoints and NO GPU. It consumes, per run:
  1. a training-series CSV:  step, length[, entropy]   (from wandb export)
  2. an eval-trajectory CSV: step, gsm8k[, math500]    (greedy pass@1 per ckpt)

It applies stopping rules offline, maps each rule's trigger step to the last
already-evaluated checkpoint before the trigger, and reports the accuracy that
a practitioner following that rule would have ended up with - versus the
oracle peak and versus continuing to the end of the horizon.

Stopping rules (all label-free):
  len_drop  : stop when length < (1 - tau) * rolling-median(length)
              [catches INTUITOR-style token collapse: length crashes down]
  len_rise  : stop when length > (1 + tau) * rolling-median(length)
              [catches cluster-only answer drift: length climbs]
  two_sided : first of len_drop or len_rise
  ent_floor : stop when entropy < e_min nats (requires entropy column)

The paper's argument this feeds (Discussion / rebuttal-proof paragraph):
  (a) the two single-signal failures move the monitors in OPPOSITE directions,
      so any one-sided rule silently fails on one family;
  (b) even the best rule freezes the model at its early peak, forfeiting the
      later gains that the stable method keeps accruing;
  (c) tau / e_min / burn-in cannot be validated without labels.

Usage (per run):
  python evaluation/early_stopping_analysis.py \
      --train-csv intuitor_train.csv --eval-csv intuitor_eval.csv \
      --name INTUITOR [--tau 0.15 0.25 0.40] [--window 20] [--e-min 0.5] \
      [--burn-in 60]

CSV column names are matched case-insensitively; use --length-col/--entropy-col
/--step-col to override. Rows with missing values are skipped.

Optional wandb pull (needs `pip install wandb` + login):
  python evaluation/early_stopping_analysis.py --wandb entity/project/run_id \
      --length-key <wandb metric key> --entropy-key <key> --eval-csv ... --name ...
"""

import argparse
import csv
import sys
from statistics import median


def read_csv(path, want, override=None):
    """Read CSV -> list of dicts with keys from `want` (case-insensitive match)."""
    rows = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in r.fieldnames}
        m = {}
        for k in want:
            src = (override or {}).get(k) or next(
                (cols[c] for c in cols if k in c), None)
            if src is None and k != "entropy" and k != "math500":
                sys.exit(f"{path}: no column matching '{k}' (have {list(cols)}); "
                         f"use --{k}-col")
            m[k] = src
        for row in r:
            out = {}
            ok = True
            for k, src in m.items():
                if src is None:
                    out[k] = None
                    continue
                v = row.get(src, "").strip()
                if v == "" or v.lower() in ("nan", "none"):
                    if k in ("step", "length"):
                        ok = False
                        break
                    out[k] = None
                else:
                    out[k] = float(v)
            if ok:
                rows.append(out)
    rows.sort(key=lambda d: d["step"])
    return rows


def rolling_median(vals, i, w):
    lo = max(0, i - w)
    return median(vals[lo:i]) if i > lo else None


def find_trigger(series, key, w, cond):
    vals = [r[key] for r in series]
    for i in range(1, len(series)):
        med = rolling_median(vals, i, w)
        if med and med > 0 and cond(vals[i], med):
            return series[i]["step"]
    return None


def pick_ckpt(eval_rows, trigger):
    """Last evaluated checkpoint at or before the trigger (None -> final ckpt)."""
    if trigger is None:
        return eval_rows[-1]
    prior = [r for r in eval_rows if r["step"] <= trigger]
    return prior[-1] if prior else eval_rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv")
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tau", nargs="+", type=float, default=[0.15, 0.25, 0.40])
    ap.add_argument("--window", type=int, default=20,
                    help="rolling-median window (logged points)")
    ap.add_argument("--e-min", type=float, default=0.5,
                    help="entropy floor in nats for ent_floor rule")
    ap.add_argument("--burn-in", type=float, default=0,
                    help="arm rules only after this training step (skip the "
                         "initial format-learning transient)")
    ap.add_argument("--step-col"), ap.add_argument("--length-col")
    ap.add_argument("--entropy-col"), ap.add_argument("--gsm8k-col")
    ap.add_argument("--math500-col")
    ap.add_argument("--wandb", help="entity/project/run_id (alternative to --train-csv)")
    ap.add_argument("--length-key", default=None, help="wandb metric key for length")
    ap.add_argument("--entropy-key", default=None, help="wandb metric key for entropy")
    args = ap.parse_args()

    if args.wandb:
        import wandb  # noqa: requires login
        run = wandb.Api().run(args.wandb)
        keys = ["_step", args.length_key] + ([args.entropy_key] if args.entropy_key else [])
        train = [{"step": h["_step"], "length": h.get(args.length_key),
                  "entropy": h.get(args.entropy_key) if args.entropy_key else None}
                 for h in run.scan_history(keys=keys)
                 if h.get(args.length_key) is not None]
        train.sort(key=lambda d: d["step"])
    elif args.train_csv:
        train = read_csv(args.train_csv, ["step", "length", "entropy"],
                         {"step": args.step_col, "length": args.length_col,
                          "entropy": args.entropy_col})
    else:
        sys.exit("need --train-csv or --wandb")

    if args.burn_in:
        train = [r for r in train if r["step"] >= args.burn_in]

    evals = read_csv(args.eval_csv, ["step", "gsm8k", "math500"],
                     {"step": args.step_col, "gsm8k": args.gsm8k_col,
                      "math500": args.math500_col})

    def acc(row):
        g = row.get("gsm8k")
        m = row.get("math500")
        parts = [f"GSM8K {g:.1f}" if g is not None else None,
                 f"M500 {m:.1f}" if m is not None else None]
        return f"step {int(row['step'])}: " + ", ".join(p for p in parts if p)

    peak = max(evals, key=lambda r: (r.get("gsm8k") or 0))
    final = evals[-1]
    print(f"\n## {args.name}  (rules armed after step {args.burn_in:g})")
    print(f"oracle peak : {acc(peak)}   (label-based -- not available in RLIF)")
    print(f"end of run  : {acc(final)}\n")
    print("| rule | tau | trigger step | selected ckpt & accuracy |")
    print("|---|---|---|---|")
    w = args.window
    for tau in args.tau:
        for rule, cond in [
            ("len_drop", lambda v, m, t=tau: v < (1 - t) * m),
            ("len_rise", lambda v, m, t=tau: v > (1 + t) * m),
            ("two_sided", lambda v, m, t=tau: v < (1 - t) * m or v > (1 + t) * m),
        ]:
            trig = find_trigger(train, "length", w, cond)
            sel = pick_ckpt(evals, trig)
            print(f"| {rule} | {tau} | {trig if trig is not None else 'never'} "
                  f"| {acc(sel)} |")
    if any(r.get("entropy") is not None for r in train):
        ent = [r for r in train if r.get("entropy") is not None]
        trig = next((r["step"] for r in ent if r["entropy"] < args.e_min), None)
        sel = pick_ckpt(evals, trig)
        print(f"| ent_floor | <{args.e_min} nats | "
              f"{trig if trig is not None else 'never'} | {acc(sel)} |")


if __name__ == "__main__":
    main()
