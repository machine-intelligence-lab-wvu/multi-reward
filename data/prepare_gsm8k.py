"""Prepare GSM8K for VERL. Usage: python data/prepare_gsm8k.py --output_dir data/gsm8k --model Qwen/Qwen3-1.7B"""
import argparse, os, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data/gsm8k")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--enable_thinking", action="store_true")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for split in ["train", "test"]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        records = []
        for i, ex in enumerate(ds):
            ans = ex["answer"].split("####")[-1].strip()
            records.append({"data_source": "openai/gsm8k",
                "prompt": [{"role":"system","content":"Please reason step by step, and put your final answer within \\boxed{}."},
                           {"role":"user","content":ex["question"]}],
                "ability": "math",
                "reward_model": {"style":"rule","ground_truth":ans},
                "extra_info": {"index":i,"question":ex["question"]}})
        df = pd.DataFrame(records)
        df.to_parquet(f"{args.output_dir}/{split}.parquet", index=False)
        print(f"Saved {len(records)} to {args.output_dir}/{split}.parquet")

if __name__ == "__main__": main()
