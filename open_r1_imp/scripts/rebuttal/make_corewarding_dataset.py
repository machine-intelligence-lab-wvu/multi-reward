#!/usr/bin/env python3
"""
Build the paired two-view dataset for the faithful Co-rewarding-I run
(A3_corewarding_qwen25_15b).

Source: the authors' RELEASED rephrasings — Co-rewarding repo
Co-rewarding-I/data/math/train_pairs.parquet (7,500 MATH train problems,
rephrased by Qwen3-32B; columns idx / original_question / rephrased_question).

Output: an interleaved dataset with one row per (question, view):
    uid | view ('ori'|'aug') | problem | solution
ordered [uid0-ori, uid0-aug, uid1-ori, uid1-aug, ...] so that with
`shuffle_dataset: false` and a single training process, both views of a uid
always land in the same TRL reward batch (required by co_rewarding_reward).

Run:  python open_r1_imp/scripts/rebuttal/make_corewarding_dataset.py
"""
import os
import pandas as pd

SRC = "${MR_DATA_DIR}/corewarding_src/train_pairs.parquet"
OUT_DIR = "${MR_DATA_DIR}/corewarding_math_paired"

def main():
    df = pd.read_parquet(SRC)
    rows = []
    for r in df.itertuples(index=False):
        ori = str(r.original_question).strip()
        aug = str(r.rephrased_question).strip()
        if not ori or not aug:
            continue
        sol = str(r.solution)
        rows.append({"uid": int(r.idx), "view": "ori", "problem": ori, "solution": sol})
        rows.append({"uid": int(r.idx), "view": "aug", "problem": aug, "solution": sol})
    out = pd.DataFrame(rows)
    assert len(out) % 2 == 0
    # sanity: adjacency of pairs
    for i in range(0, min(len(out), 20), 2):
        assert out.iloc[i]["uid"] == out.iloc[i + 1]["uid"]
        assert out.iloc[i]["view"] == "ori" and out.iloc[i + 1]["view"] == "aug"
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
    print(f"wrote {len(out)} rows ({len(out)//2} pairs) -> {OUT_DIR}/train.parquet")

if __name__ == "__main__":
    main()
