"""Re-score saved CRUXEval-O JSONs using the current `_extract_answer` +
`python_objects_equal`. Doesn't touch generation; only re-runs scoring on
the completions already saved in each *__cruxeval_o.json file.

Usage:
    PYTHONPATH=. python evaluation/rescore_cruxeval.py \
        --root ${MR_OUTPUT_DIR}/eval_results_final/code_eval

By default writes the recomputed accuracy back into the same JSON (under
`accuracy_recomputed`) and updates `pass@1` / `accuracy` so summarize.py
picks up the new number. Pass --dry-run to only print, don't write.
"""

import argparse
import json
from pathlib import Path

from evaluation.tasks.cruxeval_o import _extract_answer
from evaluation.tasks.sandbox import python_objects_equal


def rescore_file(path: Path, dry_run: bool):
    data = json.loads(path.read_text())
    records = data.get("records", [])
    if not records:
        return None
    n = 0
    correct_old = 0
    correct_new = 0
    for r in records:
        n += 1
        gold = r.get("gold")
        comps = r.get("completions") or []
        if not comps:
            continue
        c = comps[0]
        # Old score (as saved)
        if r.get("per_sample_scores", [0.0])[0] >= 1.0:
            correct_old += 1
        # New score
        pred = _extract_answer(c)
        new_score = 0.0
        if pred is not None:
            try:
                if python_objects_equal(pred, gold):
                    new_score = 1.0
            except Exception:
                pass
        if new_score >= 1.0:
            correct_new += 1
        # Mutate record so the file is internally consistent
        r["per_sample_scores"] = [new_score]
        r["pass@1"] = new_score
        r["pass@k"] = new_score
        r["maj@k"] = new_score
    acc_old = correct_old / n if n else 0.0
    acc_new = correct_new / n if n else 0.0
    print(f"  {path.parent.name:<45}  old={acc_old:.4f}  new={acc_new:.4f}  Δ={acc_new-acc_old:+.4f}")
    if dry_run:
        return acc_new
    data["accuracy"] = acc_new
    data["pass@1"] = acc_new
    data["correct"] = correct_new
    data["correct_pass1"] = correct_new
    data["accuracy_old_extractor"] = acc_old
    path.write_text(json.dumps(data, indent=2, default=str))
    return acc_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*_ckpt*/*__cruxeval_o.json"))
    if not files:
        raise SystemExit(f"no *__cruxeval_o.json under {root}")
    print(f"found {len(files)} cruxeval_o json files")
    for f in files:
        rescore_file(f, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
