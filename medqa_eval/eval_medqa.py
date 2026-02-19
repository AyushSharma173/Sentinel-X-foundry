#!/usr/bin/env python3
"""MedQA Baseline Evaluation for MedGemma 27B (4-bit quantized) — vLLM backend.

Phase 1A of the Sentinel-X research roadmap: sanity-check evaluation on the
MedQA benchmark (1,273 4-choice MCQ). Google reports 87.7% accuracy.

Uses vLLM offline mode for batch inference with continuous batching and
PagedAttention.  All prompts are submitted at once; vLLM handles scheduling
and KV-cache management internally.

All output is written to a single JSONL file (one JSON object per line):
  - Line 1:  {"type": "run_header", ...}   run metadata
  - Lines 2+: {"type": "question", ...}    one per question (appended live)
  - Last line: {"type": "summary", ...}    aggregate stats

Usage:
    # Sanity check (50 questions, ~30 min on RTX 4090)
    python medqa_eval/eval_medqa.py --max-samples 50

    # Full evaluation (1,273 questions)
    python medqa_eval/eval_medqa.py
"""

import argparse
import json
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import os

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedGemma on the MedQA benchmark (vLLM backend)"
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
        default=512,
        help="Max new tokens per generation (default: 1536)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory for vLLM (default: 0.85; leaves headroom for BnB dequant buffers)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum sequence length (prompt + generation) (default: 2048)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graphs (required for BnB 4-bit on 24GB GPUs to avoid OOM during graph capture)",
    )
    parser.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        default=False,
        help="Disable chunked prefill",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Max concurrent sequences in vLLM scheduler (default: 16; lower to reduce sampler warmup VRAM)",
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


def preflight_vram_check(min_free_gb: float = 10.0) -> None:
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

def load_vllm_model(args: argparse.Namespace) -> LLM:
    """Load model via vLLM for offline batch inference."""
    print(f"\nLoading model via vLLM: {args.model_id} ...")
    llm = LLM(
        model=args.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=not args.no_chunked_prefill,
        trust_remote_code=True,
        # kv_cache_dtype="fp8"
    )
    vram = get_vram_info()
    print(f"  Model loaded — VRAM: {vram['allocated_gb']:.2f} GB allocated")
    return llm


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT = (
#     "You are a medical expert. Answer the following multiple-choice medical question. "
#     "Think step by step, then provide your final answer as \"The answer is (X)\" "
#     "where X is the letter A, B, C, or D."
# )


# NEW:
SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following multiple-choice medical question. "
    "Reason through it in 2-3 concise sentences, then state your final answer as "
    "\"The answer is (X)\" where X is the letter A, B, C, or D."
)


def build_messages(question: str, options: dict) -> list[dict]:
    """Build chat messages for a single MedQA question."""
    option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}\n\n{option_text}"}]


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


def run_inference_vllm(
    llm: LLM,
    dataset,
    args: argparse.Namespace,
    results_accumulator: list[dict],
    output_path: Path,
) -> list[dict]:
    """Run batch inference via vLLM on all MedQA questions.

    All prompts are submitted at once; vLLM handles batching internally.
    Results are written to JSONL as they are post-processed.
    """
    total = len(dataset)

    # Build all conversations and parallel metadata
    conversations = []
    metadata = []
    for i, example in enumerate(dataset):
        data = example["data"]
        question = data["Question"]
        options = data["Options"]
        correct_answer = data["Correct Option"]

        messages = build_messages(question, options)
        conversations.append(messages)
        metadata.append({
            "question_id": i,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
        })

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Single batch call — vLLM handles scheduling and batching
    print(f"\n  Submitting {total} prompts to vLLM for batch inference ...")
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    total_time = time.time() - t0
    amortized_time = total_time / total if total else 0

    print(f"\n  Batch inference complete: {total_time:.1f}s total, {amortized_time:.1f}s/question")

    # Post-process outputs
    correct = 0
    failed_extractions = 0

    for output, meta in zip(outputs, metadata):
        response = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)

        model_answer, extraction_method = extract_answer(response)
        is_correct = model_answer == meta["correct_answer"]
        mark = "ok" if is_correct else "WRONG"

        if is_correct:
            correct += 1
        if extraction_method == "FAILED_EXTRACTION":
            failed_extractions += 1

        print(
            f"  Q {meta['question_id'] + 1}/{total}  [{mark}]  "
            f"model={model_answer} correct={meta['correct_answer']}"
        )

        record = {
            "type": "question",
            "question_id": meta["question_id"],
            "question": meta["question"],
            "options": meta["options"],
            "correct_answer": meta["correct_answer"],
            "model_answer": model_answer,
            "is_correct": is_correct,
            "model_response": response,
            "extraction_method": extraction_method,
            "output_tokens": output_tokens,
            "time_seconds": round(amortized_time, 2),
        }

        results_accumulator.append(record)
        append_jsonl(output_path, record)

    acc = correct / total * 100 if total else 0
    print(
        f"\n  Processed {total} results: "
        f"acc={acc:.1f}%  failed_extractions={failed_extractions}"
    )

    return results_accumulator


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
    print("  MedQA Baseline Evaluation (vLLM)")
    print("=" * 60)

    # Pre-flight
    preflight_vram_check(min_free_gb=10.0)

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Load dataset
    dataset = load_medqa(args.max_samples)

    # Load model via vLLM
    llm = load_vllm_model(args)

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
        "inference_engine": "vllm",
        "timestamp": ts,
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "total_questions": len(dataset),
    })

    print(f"\nStarting evaluation ({len(dataset)} questions) ...")
    print(f"  Output: {output_path}\n")
    _start_time = time.time()

    _partial_results.clear()
    results = run_inference_vllm(
        llm,
        dataset,
        args,
        results_accumulator=_partial_results,
        output_path=output_path,
    )

    total_time = time.time() - _start_time

    # Append summary as the final line
    summary = build_summary(results, args, total_time)
    append_jsonl(output_path, summary)

    print(f"\nResults saved: {output_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
