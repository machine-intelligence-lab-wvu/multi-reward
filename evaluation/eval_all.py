"""
Evaluate trained model on reasoning benchmarks.
Usage: python evaluation/eval_all.py --model_path ./outputs/multi_reward_grpo_4090
"""
import argparse, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_4090_imp"))
from rewards.answer_extraction import extract_boxed_answer, normalize_answer, answers_equivalent

def load_benchmark(name):
    from datasets import load_dataset
    if name == "gsm8k":
        return [{"question": x["question"], "answer": x["answer"].split("####")[-1].strip()}
                for x in load_dataset("openai/gsm8k", "main", split="test")]
    elif name == "math500":
        return [{"question": x["problem"], "answer": extract_boxed_answer(x["solution"]) or ""}
                for x in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    raise ValueError(f"Unknown: {name}")

def generate(model_path, questions, n=1, temp=0.6, max_tok=1024):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=2048)
    prompts = []
    for q in questions:
        msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": q}]
        try: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    return [[o.text for o in out.outputs]
            for out in llm.generate(prompts, SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tok, n=n))]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--output_dir", default="eval_results")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for bench in args.benchmarks:
        print(f"\nEvaluating {bench}...")
        data = load_benchmark(bench)
        qs, gts = [d["question"] for d in data], [d["answer"] for d in data]
        resps = generate(args.model_path, qs, args.n_samples, args.temperature)
        correct = sum(answers_equivalent(extract_boxed_answer(r[0]), g) for r, g in zip(resps, gts))
        p1 = correct / len(gts)
        maj_correct = 0
        for rs, g in zip(resps, gts):
            ans = [normalize_answer(extract_boxed_answer(r)) for r in rs]
            ans = [a for a in ans if a is not None]
            if ans and answers_equivalent(Counter(ans).most_common(1)[0][0], g): maj_correct += 1
        mv = maj_correct / len(gts)
        results[bench] = {"pass@1": p1, f"majority@{args.n_samples}": mv}
        print(f"  pass@1={p1:.4f}  majority@{args.n_samples}={mv:.4f}")
    with open(f"{args.output_dir}/results.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__": main()
