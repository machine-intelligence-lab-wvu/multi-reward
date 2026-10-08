"""
Paper-aligned evaluation script for GSM8K and MATH500.

Protocol implemented:
- Chat-style prompting (same style as training)
- Greedy decoding only (temperature=0, 1 completion)
- Benchmarks: GSM8K and MATH500

Examples:
  python evaluation/eval.py \
      --model_path outputs/multi_reward_scratch_4090/run_0/checkpoint-400

  python evaluation/eval.py \
      --model_path outputs/multi_reward_scratch_4090/run_0/checkpoint-400 \
      --benchmarks gsm8k math500 \
      --max_samples 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_4090_imp"))

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from rewards.answer_extraction import (
    answers_equivalent,
    extract_boxed_answer,
)


# SYSTEM_PROMPT = (
#     "You are a helpful AI Assistant, designed to provided well-reasoned "
#     "and detailed responses. You FIRST think about the reasoning process "
#     "step by step and then provide the user with the answer. Please enclose "
#     "your final answer in the box: \\boxed{Your Answer}. Please stop "
#     "generation immediately after outputing the box."
# )

# SYSTEM_PROMPT = ("You are a helpful AI Assistant, designed to provided well-reasoned. You FIRST think about the reasoning process step by step and then provide the user with the answer. Solve the problem step by step and show your reasoning clearly. At the end, on a separate final line, write exactly: Final Answer: \\boxed{...}. Do not give the boxed answer before the end.")


SYSTEM_PROMPT = (
    "You are a helpful AI Assistant, designed to provided well-reasoned "
    "You FIRST think about the reasoning process step by step and then provide the user with the answer. Solve the problem step by step and show your reasoning clearly."
    "On the final line only, write exactly: Final Answer: \\boxed{...}. "
    "Do not output any text after that final line."
)



def is_lora_adapter_checkpoint(path):
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def resolve_base_model_for_adapter(adapter_path, base_model_override=None):
    if base_model_override:
        return base_model_override

    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    base_model = adapter_cfg.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "Could not determine base model from adapter_config.json. "
            "Please pass --base_model explicitly."
        )
    return base_model



def _extract_gt_answer(example):
    answer = example.get("answer", "")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    solution = example.get("solution", "")
    extracted = extract_boxed_answer(solution) if isinstance(solution, str) else None
    return extracted or ""


def load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    data = []
    for x in ds:
        gt = x["answer"].split("####")[-1].strip()
        data.append({"question": x["question"], "answer": gt})
    return data


def load_math500():
    """Load MATH-500 test set from nlile/hendrycks-MATH-benchmark (500 problems)."""
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    data = []
    for x in ds:
        question = x.get("problem", "")
        answer = x.get("answer", "")
        if not answer:
            sol = x.get("solution", "")
            extracted = extract_boxed_answer(sol)
            answer = extracted if extracted else ""
        data.append({"question": question, "answer": answer})
    print(f"Loaded MATH-500: {len(data)} problems")
    return data


def load_benchmark(name):
    if name == "gsm8k":
        return load_gsm8k()
    if name == "math500":
        return load_math500()
    raise ValueError(f"Unknown benchmark: {name}")


def build_prompts(tokenizer, questions):
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def evaluate_one_benchmark(llm, tokenizer, name, data, max_new_tokens, min_new_tokens=0):
    questions = [x["question"] for x in data]
    answers = [x["answer"] for x in data]
    prompts = build_prompts(tokenizer, questions)

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    correct = 0
    records = []
    for idx, (out, gt, q) in enumerate(zip(outputs, answers, questions)):
        text = out.outputs[0].text if out.outputs else ""
        pred = extract_boxed_answer(text)
        is_correct = answers_equivalent(pred, gt)
        if is_correct:
            correct += 1

        records.append(
            {
                "id": idx,
                "question": q,
                "ground_truth": gt,
                "predicted_boxed": pred,
                "correct": bool(is_correct),
                "completion": text,
            }
        )

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    return {
        "benchmark": name,
        "metric": "accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }


@torch.no_grad()
def evaluate_one_benchmark_hf(
    model,
    tokenizer,
    name,
    data,
    max_new_tokens,
    min_new_tokens=0,
    log_every=10,
):
    questions = [x["question"] for x in data]
    answers = [x["answer"] for x in data]
    prompts = build_prompts(tokenizer, questions)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    records = []
    start_time = time.time()
    print(f"  Running {name} on {len(data)} samples...")
    for idx, (prompt, gt, q) in enumerate(zip(prompts, answers, questions)):
        sample_start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        pred = extract_boxed_answer(text)
        is_correct = answers_equivalent(pred, gt)
        if is_correct:
            correct += 1

        records.append(
            {
                "id": idx,
                "question": q,
                "ground_truth": gt,
                "predicted_boxed": pred,
                "correct": bool(is_correct),
                "completion": text,
            }
        )

        done = idx + 1
        if done % max(1, log_every) == 0 or done == len(data):
            elapsed = time.time() - start_time
            avg_sec = elapsed / done
            eta_sec = avg_sec * (len(data) - done)
            last_sec = time.time() - sample_start
            print(
                f"  [{name}] {done}/{len(data)} "
                f"acc={correct / done:.4f} "
                f"last={last_sec:.1f}s avg={avg_sec:.1f}s eta={eta_sec/60:.1f}m"
            )

    total = len(data)
    accuracy = correct / total if total > 0 else 0.0
    return {
        "benchmark": name,
        "metric": "accuracy",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-aligned evaluation for GSM8K and MATH500")
    parser.add_argument("--model_path", required=True, help="Model path or LoRA checkpoint path")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional base model for LoRA checkpoints. If omitted, inferred from adapter_config.json",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
        help="Benchmarks to evaluate",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap per benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print progress every N samples during HF/LoRA evaluation",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=64,
        help="Minimum generated tokens to encourage non-trivial reasoning before final boxed answer",
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--save_records", action="store_true", help="Save all per-example generations")
    parser.add_argument(
        "--save_records_limit",
        type=int,
        default=20,
        help=(
            "When --save_records is not set, save up to this many completion records per benchmark. "
            "Set 0 to disable completion saving."
        ),
    )
    parser.add_argument(
        "--eval_base_model",
        action="store_true",
        help="Also evaluate the base model separately (when evaluating a LoRA checkpoint)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    using_adapter = is_lora_adapter_checkpoint(args.model_path)
    if using_adapter:
        base_model = resolve_base_model_for_adapter(args.model_path, args.base_model)
        print(f"Detected LoRA checkpoint. Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.model_path)
        model.eval()
        backend = "transformers+peft"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        backend = "vllm"

    final = {
        "protocol": {
            "prompting": "chat_template",
            "decoding": "greedy",
            "temperature": 0.0,
            "num_completions": 1,
            "backend": backend,
        },
        "model_path": args.model_path,
        "benchmarks": {},
    }

    for benchmark in args.benchmarks:
        print(f"\nEvaluating {benchmark}...")
        data = load_benchmark(benchmark)
        if args.max_samples is not None:
            data = data[: args.max_samples]
        print(f"  Loaded {len(data)} samples")

        benchmark_results = {}

        if using_adapter:
            print(f"  Evaluating LoRA checkpoint on {benchmark}...")
            result = evaluate_one_benchmark_hf(
                model=model,
                tokenizer=tokenizer,
                name=benchmark,
                data=data,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=max(0, args.min_new_tokens),
                log_every=max(1, args.log_every),
            )

            if args.save_records:
                pass
            elif args.save_records_limit > 0:
                preview = result.get("records", [])[: args.save_records_limit]
                result["records_preview"] = preview
                result["records_preview_count"] = len(preview)
                result = {k: v for k, v in result.items() if k != "records"}
            else:
                result = {k: v for k, v in result.items() if k != "records"}

            benchmark_results["checkpoint"] = result
            print(
                f"  {benchmark} (checkpoint): accuracy={result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']})"
            )

            # Evaluate base model if requested
            if args.eval_base_model:
                print(f"  Evaluating base model on {benchmark}...")
                base_model_name = resolve_base_model_for_adapter(args.model_path, args.base_model)
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                base_model_obj.eval()

                base_result = evaluate_one_benchmark_hf(
                    model=base_model_obj,
                    tokenizer=tokenizer,
                    name=benchmark,
                    data=data,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=max(0, args.min_new_tokens),
                    log_every=max(1, args.log_every),
                )

                if args.save_records:
                    pass
                elif args.save_records_limit > 0:
                    preview = base_result.get("records", [])[: args.save_records_limit]
                    base_result["records_preview"] = preview
                    base_result["records_preview_count"] = len(preview)
                    base_result = {k: v for k, v in base_result.items() if k != "records"}
                else:
                    base_result = {k: v for k, v in base_result.items() if k != "records"}

                benchmark_results["base_model"] = base_result
                print(
                    f"  {benchmark} (base model): accuracy={base_result['accuracy']:.4f} "
                    f"({base_result['correct']}/{base_result['total']})"
                )
                del base_model_obj
                torch.cuda.empty_cache()
        else:
            result = evaluate_one_benchmark(
                llm=llm,
                tokenizer=tokenizer,
                name=benchmark,
                data=data,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=max(0, args.min_new_tokens),
            )

            if args.save_records:
                pass
            elif args.save_records_limit > 0:
                preview = result.get("records", [])[: args.save_records_limit]
                result["records_preview"] = preview
                result["records_preview_count"] = len(preview)
                result = {k: v for k, v in result.items() if k != "records"}
            else:
                result = {k: v for k, v in result.items() if k != "records"}

            benchmark_results["model"] = result
            print(
                f"  {benchmark}: accuracy={result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']})"
            )

        final["benchmarks"][benchmark] = benchmark_results

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_path.rstrip("/").replace("/", "__").replace(".", "")
    output_path = os.path.join(args.output_dir, f"eval_paper_{model_tag}_{stamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
