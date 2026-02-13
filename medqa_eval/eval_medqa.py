#!/usr/bin/env python3
"""MedQA Baseline Evaluation for MedGemma 27B (4-bit quantized).

Phase 1A of the Sentinel-X research roadmap: sanity-check evaluation on the
MedQA benchmark (1,273 4-choice MCQ). Google reports 87.7% accuracy.

All output is written to a single JSONL file (one JSON object per line):
  - Line 1:  {"type": "run_header", ...}   run metadata
  - Lines 2+: {"type": "question", ...}    one per question (appended live)
  - Last line: {"type": "summary", ...}    aggregate stats

Usage:
    # Sanity check (50 questions, ~5-10 min)
    python medqa_eval/eval_medqa.py --max-samples 50

    # Full evaluation (1,273 questions, ~2-3 hours)
    python medqa_eval/eval_medqa.py
"""

import argparse
import gc
import json
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedGemma on the MedQA benchmark"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        help="HuggingFace model ID (default: unsloth/medgemma-27b-text-it-unsloth-bnb-4bit)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of questions (default: all 1,273)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="medqa_eval/results",
        help="Directory for result JSON files (default: medqa_eval/results)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5000,
        help="Max new tokens per generation (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1, safe for 24GB VRAM)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def get_vram_info() -> dict:
    """Return current GPU VRAM usage in GB."""
    if not torch.cuda.is_available():
        return {"available": False}
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved
    return {
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
    }


def preflight_vram_check(min_free_gb: float = 15.0) -> None:
    """Abort early if insufficient VRAM is available."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script requires a GPU.")
        sys.exit(1)

    info = get_vram_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: {info['free_gb']:.1f} GB free / {info['total_gb']:.1f} GB total"
    )

    if info["free_gb"] < min_free_gb:
        print(
            f"ERROR: Need at least {min_free_gb} GB free VRAM, "
            f"but only {info['free_gb']:.1f} GB available."
        )
        sys.exit(1)


def cleanup_gpu(model=None, tokenizer=None) -> None:
    """Two-pass GPU memory cleanup (handles BitsAndBytes quantized models)."""
    vram_before = get_vram_info()
    print(f"\nGPU cleanup — VRAM before: {vram_before['allocated_gb']:.2f} GB allocated")

    # Move model to CPU first (may fail for quantized models)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass  # quantized models can't always be moved

    # Delete references
    del model
    del tokenizer

    # First pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Second pass (catches BitsAndBytes stragglers)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    vram_after = get_vram_info()
    print(f"GPU cleanup — VRAM after:  {vram_after['allocated_gb']:.2f} GB allocated")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_medqa(max_samples: int | None = None):
    """Load the MedQA test split from HuggingFace."""
    print("Loading MedQA dataset from openlifescienceai/medqa ...")
    dataset = load_dataset("openlifescienceai/medqa", split="test")
    total = len(dataset)
    print(f"  Total test questions: {total}")

    if max_samples is not None and max_samples < total:
        dataset = dataset.select(range(max_samples))
        print(f"  Using first {max_samples} questions")

    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load pre-quantized model (no BitsAndBytesConfig needed)."""
    print(f"\nLoading tokenizer: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "32GiB"},
    )
    model.eval()

    vram = get_vram_info()
    print(f"  Model loaded — VRAM: {vram['allocated_gb']:.2f} GB allocated")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following multiple-choice medical question. "
    "Think step by step, then provide your final answer as \"The answer is (X)\" "
    "where X is the letter A, B, C, or D."
)


def build_prompt(question: str, options: dict, tokenizer) -> str:
    """Build a chat-formatted prompt for a single MedQA question."""
    option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    user_content = f"{question}\n\n{option_text}"

    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Primary: "The answer is (X)" or "The answer is X"
_PRIMARY_RE = re.compile(r"[Tt]he answer is\s*\(?([A-D])\)?")
# Fallback: last standalone A-D letter in the response
_FALLBACK_RE = re.compile(r"\b([A-D])\b")


def extract_answer(response: str) -> tuple[str, str]:
    """Extract the chosen answer letter from model response.

    Returns (letter, extraction_method) where method is one of:
      "primary", "fallback", "FAILED_EXTRACTION"
    """
    m = _PRIMARY_RE.search(response)
    if m:
        return m.group(1), "primary"

    matches = _FALLBACK_RE.findall(response)
    if matches:
        return matches[-1], "fallback"

    return "FAILED_EXTRACTION", "FAILED_EXTRACTION"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON object as one line to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def run_inference(
    model,
    tokenizer,
    dataset,
    temperature: float,
    max_new_tokens: int,
    results_accumulator: list[dict] | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """Run model on each MedQA question and collect results.

    If results_accumulator is provided, results are appended to it in real-time
    (enables partial saves on Ctrl+C).  Each question result is also appended
    as a JSON line to output_path.
    """
    results = results_accumulator if results_accumulator is not None else []
    total = len(dataset)
    correct = 0
    failed_extractions = 0
    total_output_tokens = 0

    do_sample = temperature > 0

    for i, example in enumerate(dataset):
        data = example["data"]
        question = data["Question"]
        options = data["Options"]
        correct_answer = data["Correct Option"]

        # Build prompt and tokenize
        prompt = build_prompt(question, options, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        print(f"  Q {i + 1}/{total} ...", end="", flush=True)
        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        elapsed = time.time() - t0

        # Decode only the generated tokens (strip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_tokens = len(generated_ids)
        total_output_tokens += output_tokens

        # Extract answer
        model_answer, extraction_method = extract_answer(response)
        is_correct = model_answer == correct_answer
        mark = "ok" if is_correct else "WRONG"
        print(f" {elapsed:.1f}s  [{mark}]  model={model_answer} correct={correct_answer}")

        if is_correct:
            correct += 1
        if extraction_method == "FAILED_EXTRACTION":
            failed_extractions += 1

        record = {
            "type": "question",
            "question_id": i,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "model_response": response,
            "extraction_method": extraction_method,
            "output_tokens": output_tokens,
            "time_seconds": round(elapsed, 2),
        }

        results.append(record)

        # Append to JSONL output (one line per question, written live)
        if output_path is not None:
            append_jsonl(output_path, record)

        # Free intermediate tensors
        del inputs, outputs, generated_ids
        torch.cuda.empty_cache()

        # Progress logging every 10 questions
        if (i + 1) % 10 == 0 or (i + 1) == total:
            acc = correct / (i + 1) * 100
            vram = get_vram_info()
            print(
                f"  [{i + 1:>4}/{total}]  "
                f"acc={acc:.1f}%  "
                f"failed_extractions={failed_extractions}  "
                f"vram={vram['allocated_gb']:.1f}GB  "
                f"last={elapsed:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Results & reporting
# ---------------------------------------------------------------------------

def build_summary(results: list[dict], args: argparse.Namespace, total_time: float) -> dict:
    """Compute aggregate summary from in-memory results."""
    total_questions = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    failed = sum(1 for r in results if r["extraction_method"] == "FAILED_EXTRACTION")
    total_tokens = sum(r["output_tokens"] for r in results)
    avg_tokens = total_tokens / total_questions if total_questions else 0
    avg_time = total_time / total_questions if total_questions else 0

    return {
        "type": "summary",
        "model_id": args.model_id,
        "dataset": "openlifescienceai/medqa",
        "split": "test",
        "total_questions": total_questions,
        "correct": correct,
        "accuracy": round(correct / total_questions, 4) if total_questions else 0,
        "failed_extractions": failed,
        "avg_tokens_per_response": round(avg_tokens, 1),
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_question": round(avg_time, 2),
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }


def print_summary(s: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("  MedQA Evaluation Summary")
    print("=" * 60)
    print(f"  Model:              {s['model_id']}")
    print(f"  Dataset:            {s['dataset']} ({s['split']})")
    print(f"  Questions:          {s['total_questions']}")
    print(f"  Correct:            {s['correct']}")
    print(f"  Accuracy:           {s['accuracy'] * 100:.1f}%")
    print(f"  Failed extractions: {s['failed_extractions']}")
    print(f"  Avg tokens/resp:    {s['avg_tokens_per_response']:.0f}")
    print(f"  Total time:         {s['total_time_seconds']:.0f}s")
    print(f"  Avg time/question:  {s['avg_time_per_question']:.1f}s")
    print(f"  Temperature:        {s['temperature']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Signal handling for graceful Ctrl+C
# ---------------------------------------------------------------------------

_partial_results: list[dict] = []
_args: argparse.Namespace | None = None
_start_time: float = 0.0
_output_path: Path | None = None


def _signal_handler(signum, frame):
    """Append summary to the JSONL file on interrupt, then exit."""
    sig_name = signal.Signals(signum).name
    print(f"\n\nReceived {sig_name} — saving partial summary ...")

    if _partial_results and _args is not None:
        elapsed = time.time() - _start_time
        summary = build_summary(_partial_results, _args, elapsed)
        summary["partial"] = True
        if _output_path is not None:
            append_jsonl(_output_path, summary)
            print(f"  Partial summary appended to: {_output_path}")
        print_summary(summary)

    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _partial_results, _args, _start_time, _output_path

    args = parse_args()
    _args = args

    print("=" * 60)
    print("  MedQA Baseline Evaluation")
    print("=" * 60)

    # Pre-flight
    preflight_vram_check(min_free_gb=15.0)

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    model = None
    tokenizer = None
    try:
        # Load dataset
        dataset = load_medqa(args.max_samples)

        # Load model
        model, tokenizer = load_model(args.model_id)

        # Prepare single JSONL output file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"medqa_eval_{ts}.jsonl"
        _output_path = output_path

        # Write run header as the first line
        append_jsonl(output_path, {
            "type": "run_header",
            "model_id": args.model_id,
            "timestamp": ts,
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "total_questions": len(dataset),
        })

        print(f"\nStarting evaluation ({len(dataset)} questions) ...")
        print(f"  Output: {output_path}\n")
        _start_time = time.time()

        _partial_results.clear()  # reset for this run
        results = run_inference(
            model,
            tokenizer,
            dataset,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            results_accumulator=_partial_results,
            output_path=output_path,
        )

        total_time = time.time() - _start_time

        # Append summary as the final line
        summary = build_summary(results, args, total_time)
        append_jsonl(output_path, summary)

        print(f"\nResults saved: {output_path}")
        print_summary(summary)

    finally:
        # GPU cleanup — always runs
        cleanup_gpu(model, tokenizer)


if __name__ == "__main__":
    main()
