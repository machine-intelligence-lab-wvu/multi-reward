"""Prepare MATH for VERL. Usage: python data/prepare_math.py --output_dir data/math --model Qwen/Qwen3-1.7B"""
import argparse, os, re, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

def extract_math_answer(sol):
    idx = sol.rfind("\\boxed{")
    if idx == -1: return sol.strip()
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(sol)):
        if sol[i] == "{": depth += 1
        elif sol[i] == "}":
            if depth == 0: return sol[start:i].strip()
            depth -= 1
    return sol[start:].strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/math")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", default="lighteval/MATH")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for split in ["train", "test"]:
        try: ds = load_dataset(args.dataset, split=split, trust_remote_code=True)
        except: ds = load_dataset(args.dataset, "all", split=split, trust_remote_code=True)
        records = []
        for i, ex in enumerate(ds):
            q = ex.get("problem", ex.get("question", ""))
            s = ex.get("solution", ex.get("answer", ""))
            records.append({"data_source": args.dataset,
                "prompt": [{"role":"system","content":"Please reason step by step, and put your final answer within \\boxed{}."},
                           {"role":"user","content":q}],
                "ability": "math",
                "reward_model": {"style":"rule","ground_truth": extract_math_answer(s)},
                "extra_info": {"index":i,"question":q,"level":ex.get("level",""),"subject":ex.get("type","")}})
        df = pd.DataFrame(records)
        df.to_parquet(f"{args.output_dir}/{split}.parquet", index=False)
        print(f"Saved {len(records)} to {args.output_dir}/{split}.parquet")

if __name__ == "__main__": main()
