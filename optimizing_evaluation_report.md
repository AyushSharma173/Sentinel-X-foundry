# Optimizing Your Experiment Workflows: A Comprehensive Guide

**For Sentinel-X Foundry — MedGemma 27B Evaluation, SFT & GRPO Fine-Tuning**

> This report covers everything you need to dramatically speed up benchmark evaluation,
> prompt iteration, and fine-tuning. Written for someone who wants to understand not just
> *what* to do, but *why* it works — from GPU fundamentals through concrete action items.

---

## Table of Contents

1. [Your Current Setup Analysis](#1-your-current-setup-analysis)
2. [GPU Fundamentals for ML](#2-gpu-fundamentals-for-ml)
3. [MedGemma Model Deep Dive](#3-medgemma-model-deep-dive)
4. [Inference Speed Optimization](#4-inference-speed-optimization)
5. [Batch Processing Strategies](#5-batch-processing-strategies)
6. [GPU Upgrade Options](#6-gpu-upgrade-options)
7. [Fine-Tuning Speed Guide](#7-fine-tuning-speed-guide)
8. [GRPO/RL Training Speed](#8-grporl-training-speed)
9. [Benchmark Evaluation at Scale](#9-benchmark-evaluation-at-scale)
10. [Concrete Recommendations](#10-concrete-recommendations)

---

## 1. Your Current Setup Analysis

### What You Have

| Component | Current Value | Notes |
|-----------|--------------|-------|
| GPU | RTX 4090 (24 GB VRAM) | Consumer card, excellent for its class |
| Model | MedGemma 27B, 4-bit (BnB NF4) | `unsloth/medgemma-27b-text-it-unsloth-bnb-4bit` |
| Batch size | 1 | One question at a time |
| max_new_tokens | 5,000 | Very generous — most answers are much shorter |
| Framework | HuggingFace `model.generate()` | No batched inference engine |
| Speed | ~65 seconds/question | ~20 questions/hour |
| Full MedQA run | ~23 hours (1,273 questions) | Extremely slow |

### Where the Bottlenecks Are

**Bottleneck 1: Sequential inference (batch_size=1)**
Your `run_inference()` loop in `eval_medqa.py` processes one question at a time. The GPU
does one forward pass, waits for the full response to generate token-by-token, then starts
the next question. Between questions the GPU is briefly idle during Python overhead, tensor
cleanup, and progress logging. With batch_size=1, the GPU's parallel compute capacity is
drastically underutilized.

**Bottleneck 2: max_new_tokens=5,000**
You're telling the model "you may generate up to 5,000 tokens." For a multiple-choice
medical question with chain-of-thought reasoning, responses are typically 200-600 tokens.
A 5,000-token ceiling means:
- The KV-cache is pre-allocated for the maximum possible sequence, wasting VRAM
- If the model doesn't hit an EOS token for some reason, you waste enormous time
- Reduces headroom for batching (more reserved memory = fewer concurrent sequences)

**Bottleneck 3: No inference engine (raw HuggingFace generate)**
`model.generate()` is a straightforward autoregressive loop: generate one token, append it,
generate the next. It doesn't use continuous batching, PagedAttention, or any of the
optimizations that modern inference engines like vLLM provide. For single requests it's
okay; for 1,273 sequential requests, you're leaving 5-20x performance on the table.

**Bottleneck 4: BitsAndBytes 4-bit (in HF)**
BitsAndBytes NF4 quantization in HuggingFace is great for fitting the model in memory.
But it's slower than optimized quantization formats (AWQ, GPTQ with Marlin kernels)
because BnB dequantizes on-the-fly during each forward pass without the fused kernel
optimizations that formats like Marlin provide.

### The Opportunity

With the optimizations in this report, you can realistically achieve:

| Scenario | Speed | Improvement |
|----------|-------|-------------|
| Current (HF generate, batch=1) | ~65 s/question | Baseline |
| + Reduce max_new_tokens to 512 | ~20-30 s/question | 2-3x |
| + vLLM with batching (batch=16) | ~3-5 s/question | 13-20x |
| + GPU upgrade (A100 80GB + vLLM) | ~1-2 s/question | 30-60x |

**Bottom line:** The full MedQA evaluation could go from ~23 hours to under 30 minutes.

---

## 2. GPU Fundamentals for ML

If you already know what VRAM, tensor cores, and memory bandwidth are, skip to Section 3.
This section explains what actually matters when running LLMs.

### VRAM (Video RAM)

**What it is:** The GPU's own dedicated memory (like RAM for your CPU, but on the graphics
card). Your RTX 4090 has 24 GB of VRAM.

**Why it matters:** Everything the model needs must fit in VRAM simultaneously:
- Model weights (the actual neural network parameters)
- KV-cache (stores attention state for every generated token)
- Activations (intermediate computation results during forward passes)
- Input/output buffers

If you exceed VRAM, you get a CUDA Out of Memory error. Your script already handles
this with `max_memory={0: "20GiB", "cpu": "32GiB"}` — reserving 4 GB of headroom.

### Memory Bandwidth

**What it is:** How fast data can flow between VRAM and the GPU's compute cores, measured
in GB/s.

| GPU | Memory Bandwidth |
|-----|-----------------|
| RTX 4090 | 1,008 GB/s |
| A100 80GB (SXM) | 2,039 GB/s |
| H100 80GB (SXM) | 3,350 GB/s |

**Why it matters for LLMs:** Autoregressive text generation (producing one token at a
time) is **memory-bandwidth bound**, not compute-bound. Here's why:

When generating token-by-token, each step requires reading the entire model's weights from
VRAM to compute a single output token. The arithmetic is simple (matrix-vector multiply),
but you have to *read* billions of parameters each time. The bottleneck is how fast you can
read those weights — that's memory bandwidth.

**Key insight:** Doubling your memory bandwidth roughly doubles your token generation speed.
The A100's 2x bandwidth over the 4090 is why it generates tokens ~2x faster per stream,
even though the 4090 has comparable raw FLOPS.

### Tensor Cores

**What they are:** Specialized hardware units on NVIDIA GPUs that do matrix multiplication
extremely fast, especially at lower precisions (FP16, BF16, INT8, INT4).

| GPU | Tensor Core Gen | FP16 Tensor TFLOPS |
|-----|----------------|-------------------|
| RTX 4090 | 4th gen (Ada) | 330 TFLOPS |
| A100 | 3rd gen (Ampere) | 312 TFLOPS |
| H100 | 4th gen (Hopper) | 990 TFLOPS |

**Why they matter:** Tensor cores accelerate the matrix multiplications that make up the
bulk of neural network computation. They're critical for:
- Training (forward + backward pass = lots of large matrix multiplications)
- Batch inference (multiple sequences = larger matrices = better tensor core utilization)
- Single-sequence inference (memory-bandwidth limited, tensor cores are underutilized)

### Compute-Bound vs Memory-Bound

This distinction is critical for understanding which optimizations help:

| Operation | Bound By | What Helps |
|-----------|----------|------------|
| Single-sequence token generation | Memory bandwidth | Faster memory, quantization |
| Batch inference (many sequences) | Compute (tensor cores) | More FLOPS, larger batches |
| Training (forward + backward) | Compute | More FLOPS, tensor cores |
| Prefill (processing the prompt) | Compute | Flash Attention, tensor cores |
| Loading model to GPU | PCIe / NVLink bandwidth | Faster interconnect |

**For your use case:**
- **Evaluation (generate 1,273 answers):** Memory-bandwidth bound per sequence, but
  batching turns it compute-bound (good! tensor cores can help)
- **Fine-tuning:** Compute-bound (tensor cores are fully utilized)
- **GRPO:** Both (generation phase is memory-bound, training phase is compute-bound)

---

## 3. MedGemma Model Deep Dive

### Architecture Overview

MedGemma 27B is based on the Gemma 2 27B architecture:

| Property | Value |
|----------|-------|
| Parameters | 27 billion |
| Architecture | Decoder-only transformer |
| Hidden size | 4,608 |
| Layers | 46 |
| Attention heads | 32 (with 16 KV heads — GQA) |
| Vocabulary | 256,000 tokens |
| Context window | 8,192 tokens |
| Precision (original) | BF16 (2 bytes/param) |

**Grouped Query Attention (GQA):** Instead of having separate key/value heads for each
attention head (32 KV heads for 32 Q heads), Gemma 2 uses 16 KV heads shared across 32
query heads. This halves the KV-cache size and speeds up inference with minimal quality
loss.

### 4-bit Quantization: What's Actually Happening

You're using BitsAndBytes NF4 (Normal Float 4-bit) quantization. Here's what that means:

**Full precision (BF16):** Each parameter stored as 2 bytes → 27B × 2 = **54 GB**
(Doesn't fit on your 24 GB card!)

**4-bit quantization:** Each parameter stored as 0.5 bytes → 27B × 0.5 = **~13.5 GB**
(Fits with room to spare!)

**How NF4 works:**
1. Parameters are grouped into blocks of 64 values
2. Each block stores a single FP32 scale factor (the absolute maximum)
3. Each value is quantized to one of 16 levels (4 bits) chosen to be optimal for
   normally-distributed weights (this is the "NF" — Normal Float — part)
4. **Double quantization:** The scale factors themselves are also quantized (FP32 → FP8),
   saving an additional ~0.4 bits per parameter

**Memory footprint breakdown:**

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (4-bit) | ~14 GB | 27B params × ~4.2 bits (with scales) |
| KV-cache (per sequence) | ~0.2-0.5 GB | Depends on sequence length |
| Activations + buffers | ~2-4 GB | Forward pass intermediates |
| PyTorch/CUDA overhead | ~1-2 GB | Framework allocations |
| **Total (batch=1)** | **~18-20 GB** | Fits on 24 GB with ~4 GB headroom |

**The quantization trade-off:** 4-bit quantization reduces memory by ~4x but:
- Slightly reduces model quality (typically <1% accuracy drop on benchmarks)
- BnB dequantizes on-the-fly (slower than native-precision or fused kernels)
- Not as fast as optimized formats like AWQ/GPTQ with Marlin fused kernels

### What the Model Actually Needs at Inference Time

For each token generated, the GPU must:

1. **Read** all 27B quantized weights from VRAM (~14 GB)
2. **Dequantize** them to FP16/BF16 on-the-fly
3. **Multiply** the input vector (1 × hidden_size) against each weight matrix
4. **Read** the KV-cache for all previous tokens in the sequence
5. **Compute** attention scores and weighted values
6. **Write** the new KV-cache entries for this token
7. **Sample** the next token from the output distribution

Steps 1-2 are the bottleneck — reading 14 GB of weights at 1,008 GB/s means each
token takes at minimum ~14 ms just for the memory reads. In practice, with overheads,
you see ~40-65 ms per token for a 27B 4-bit model on the 4090.

---

## 4. Inference Speed Optimization

### 4.1 vLLM: The Single Biggest Improvement

**What is vLLM?** An inference engine designed for high-throughput LLM serving. Instead of
the simple `model.generate()` loop, vLLM uses optimized CUDA kernels, intelligent memory
management, and batching strategies that can deliver **5-24x higher throughput** than
HuggingFace Transformers.

**Key innovations:**

**PagedAttention** — Instead of pre-allocating a contiguous block of memory for each
sequence's KV-cache (wasteful when sequences vary in length), vLLM splits the KV-cache
into small fixed-size pages (like virtual memory in an OS). Pages are allocated on demand
and can be non-contiguous in physical memory.

- Traditional KV-cache: 60-80% memory waste from fragmentation and over-reservation
- PagedAttention: <4% memory waste (only the last partially-filled page per sequence)
- This means you can fit **many more concurrent sequences** in the same VRAM

**Continuous Batching** — Instead of waiting for an entire batch of requests to finish
before starting the next batch, vLLM performs iteration-level scheduling:
- At each decoding step, it assembles a batch from all active sequences
- When a sequence finishes (hits EOS), its slot is immediately filled with a new request
- No idle GPU time waiting for the slowest sequence in a batch

**How to use vLLM for your evaluation:**

```python
from vllm import LLM, SamplingParams

# Load model once
llm = LLM(
    model="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
    quantization="bitsandbytes",     # supports BnB 4-bit
    gpu_memory_utilization=0.90,      # use 90% of VRAM
    max_model_len=1024,               # max total sequence length
    dtype="float16",
)

# Prepare all prompts at once
prompts = [build_prompt(q, opts, tokenizer) for q, opts in questions]

# Generate ALL responses in one call — vLLM handles batching internally
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,         # much less than 5000!
)

outputs = llm.generate(prompts, sampling_params)

# Extract responses
for output in outputs:
    response = output.outputs[0].text
    answer, method = extract_answer(response)
```

**Expected speedups with vLLM:**

| Configuration | Throughput | vs Current |
|--------------|-----------|------------|
| HF generate, batch=1, 5000 tokens | ~1 q/min | 1x |
| vLLM, BnB 4-bit, batch=16 | ~5-10 q/min | 5-10x |
| vLLM, AWQ 4-bit (Marlin), batch=16 | ~15-20 q/min | 15-20x |

**Important note on quantization format:** vLLM supports BitsAndBytes 4-bit, but for
maximum throughput, you'd get better results with an AWQ-quantized version of MedGemma
(Marlin kernels are ~4x faster than BnB dequantization). Check if an AWQ variant exists
on HuggingFace, or quantize one yourself:

```bash
# If you need to create an AWQ-quantized model
pip install autoawq
python -c "
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained('google/medgemma-27b-text-it')
model.quantize(tokenizer, quant_config={'w_bit': 4, 'q_group_size': 128})
model.save_quantized('medgemma-27b-awq')
"
```

### 4.2 Reducing max_new_tokens

This is the easiest optimization — just change a number.

**Current setting:** `max_new_tokens=5000`

For a multiple-choice question with chain-of-thought:
- Typical response: 200-400 tokens (reasoning + "The answer is (X)")
- Long response: 500-700 tokens
- Extremely long: 800-1000 tokens

**Recommended:** `max_new_tokens=512` (or 768 if you want extra safety margin)

**Why this helps:**
1. The KV-cache is sized for the maximum possible sequence length. At 5,000 tokens,
   each sequence reserves much more KV-cache memory than it will ever use.
2. If the model fails to produce an EOS token (rare but happens), you cap wasted time
   at 512 tokens instead of 5,000.
3. With less reserved memory, you have room for larger batches.

**What if a response gets truncated?** For MCQ evaluation, the answer pattern
"The answer is (X)" almost always appears in the first 300 tokens. Even if the
chain-of-thought is cut short, your regex extraction will still work. You can
verify this: check your existing results for the maximum `output_tokens` value.

### 4.3 Flash Attention 2

**What it is:** An optimized implementation of the attention mechanism that reduces memory
usage and speeds up computation by being smarter about GPU memory hierarchy (registers →
shared memory → global memory).

**Performance:**
- **2x faster** than standard attention (FlashAttention-1)
- Achieves 50-73% of theoretical peak FLOPS on A100
- Most impactful during the **prefill phase** (processing the input prompt) and training

**How to enable it:**

```python
# Method 1: In HuggingFace
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Add this
)

# Method 2: vLLM enables it automatically if installed
pip install flash-attn --no-build-isolation
```

**Note:** Flash Attention 2 primarily speeds up the prefill (prompt processing) phase.
For autoregressive decoding (token-by-token generation), the speedup is smaller because
the attention computation is already small relative to the weight loading bottleneck.
Still, for long prompts (medical questions + options can be 500+ tokens), it helps.

### 4.4 KV-Cache Optimization

The KV-cache stores key and value vectors for all previous tokens so they don't need to
be recomputed. It grows linearly with sequence length.

**KV-cache size per token (MedGemma 27B):**
- KV heads: 16 (GQA)
- Head dimension: 128
- Layers: 46
- Precision: FP16 (2 bytes)
- Per token: 16 × 128 × 46 × 2 × 2 (K and V) = ~0.38 MB

For a 1,000-token sequence: ~380 MB of KV-cache
For a 5,000-token sequence: ~1.9 GB of KV-cache (per sequence!)

**Optimizations:**
1. **Reduce max sequence length** (already covered — use 512, not 5000)
2. **FP8 KV-cache** (vLLM supports this): Halves KV-cache memory with negligible quality
   loss. Enable with `--kv-cache-dtype fp8` in vLLM
3. **PagedAttention** (vLLM): Only allocates KV-cache pages as needed, not upfront

### 4.5 Speculative Decoding

**What it is:** Use a small, fast "draft" model to predict multiple tokens ahead.
Then verify all predicted tokens in parallel with the large model. If the draft model
predicted correctly (which it often does for common patterns), you get multiple tokens
for the cost of one large-model forward pass.

**How it works:**
1. Draft model generates N tokens quickly (e.g., N=5)
2. Large model verifies all N tokens in one parallel forward pass
3. Accept all tokens up to the first incorrect one
4. Repeat from the corrected position

**Typical speedup:** 2-3x for well-matched draft/target pairs.

**For MedGemma 27B**, you could use MedGemma 4B as the draft model:
```bash
# In vLLM
vllm serve medgemma-27b \
    --speculative-model medgemma-4b \
    --num-speculative-tokens 5
```

**Caveat:** Speculative decoding adds complexity and requires VRAM for both models.
On a 24 GB card with 27B 4-bit already using ~18 GB, fitting a 4B draft model is tight.
More practical on 40+ GB cards.

---

## 5. Batch Processing Strategies

### Why Batching Matters

With batch_size=1, each forward pass computes one output token for one sequence.
The GPU reads 14 GB of weights to produce a single token. This is absurdly inefficient.

With batch_size=16, each forward pass computes one output token for 16 sequences
simultaneously. The GPU still reads 14 GB of weights — once — but produces 16 tokens.
That's 16x better utilization of memory bandwidth.

**The math:**
- batch=1: Read 14 GB → 1 token. Throughput: ~15-25 tokens/sec
- batch=16: Read 14 GB → 16 tokens. Throughput: ~200-400 tokens/sec (theoretical)
- Reality is somewhat less due to KV-cache overhead, but 8-12x improvement is achievable

### Optimal Batch Sizes for RTX 4090

The constraint is VRAM. With 24 GB total:
- Model weights: ~14 GB
- CUDA overhead: ~2 GB
- Available for KV-cache: ~8 GB

KV-cache per sequence (512 max tokens): ~195 MB
Maximum concurrent sequences: 8 GB / 195 MB ≈ **~40 sequences**

Practical sweet spot (leaving headroom): **batch_size=8-16**

With reduced max_new_tokens=512, you can comfortably batch 8-16 sequences.

### How to Modify eval_medqa.py for Batched Inference

**Option A: Use vLLM (recommended — simplest, fastest)**

Replace the entire `run_inference` function with a vLLM-based approach:

```python
from vllm import LLM, SamplingParams

def run_inference_vllm(dataset, temperature, max_new_tokens, output_path):
    """Batch inference using vLLM."""
    # Load model with vLLM
    llm = LLM(
        model="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        quantization="bitsandbytes",
        gpu_memory_utilization=0.90,
        max_model_len=1024,
    )

    # Build all prompts upfront
    prompts = []
    metadata = []
    for example in dataset:
        data = example["data"]
        prompt = build_prompt(data["Question"], data["Options"], llm.get_tokenizer())
        prompts.append(prompt)
        metadata.append({
            "question": data["Question"],
            "options": data["Options"],
            "correct_answer": data["Correct Option"],
        })

    # Generate all at once — vLLM handles batching internally
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        model_answer, method = extract_answer(response)
        meta = metadata[i]

        record = {
            "type": "question",
            "question_id": i,
            "question": meta["question"],
            "options": meta["options"],
            "correct_answer": meta["correct_answer"],
            "model_answer": model_answer,
            "is_correct": model_answer == meta["correct_answer"],
            "model_response": response,
            "extraction_method": method,
            "output_tokens": len(output.outputs[0].token_ids),
        }
        results.append(record)
        if output_path:
            append_jsonl(output_path, record)

    return results
```

**Option B: HuggingFace batched generate (if you want to stay with HF)**

```python
def run_inference_batched(model, tokenizer, dataset, temperature,
                          max_new_tokens, batch_size=8, output_path=None):
    """Batched HF inference."""
    results = []
    total = len(dataset)

    # Ensure left-padding for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = dataset.select(range(batch_start, batch_end))

        # Build prompts for batch
        prompts = []
        batch_meta = []
        for example in batch:
            data = example["data"]
            prompt = build_prompt(data["Question"], data["Options"], tokenizer)
            prompts.append(prompt)
            batch_meta.append(data)

        # Tokenize with padding
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(model.device)

        # Generate for entire batch
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )

        # Decode each sequence in the batch
        for j in range(len(prompts)):
            prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum()
            generated = outputs[j][prompt_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            # ... extract answer and record results ...

        del inputs, outputs
        torch.cuda.empty_cache()

    return results
```

### Throughput Estimates

| Setup | Questions/hour | Full MedQA Time |
|-------|---------------|-----------------|
| Current (HF, batch=1, 5K tokens) | ~20 | ~64 hours* |
| HF batch=8, 512 tokens | ~100-150 | ~8-12 hours |
| vLLM BnB, auto-batch, 512 tokens | ~300-500 | ~2.5-4 hours |
| vLLM AWQ/Marlin, auto-batch | ~600-1,000 | ~1.3-2 hours |

*\*Your measured ~65s/q gives ~55 q/hr for 50 questions, but full 1,273 runs would likely see similar per-question times.*

---

## 6. GPU Upgrade Options

### GPU Specifications Comparison

| Spec | RTX 4090 | A100 40GB | A100 80GB | H100 80GB | L40S 48GB |
|------|----------|-----------|-----------|-----------|-----------|
| VRAM | 24 GB | 40 GB | 80 GB | 80 GB | 48 GB |
| Memory BW | 1,008 GB/s | 1,555 GB/s | 2,039 GB/s | 3,350 GB/s | 864 GB/s |
| FP16 Tensor | 330 TFLOPS | 312 TFLOPS | 312 TFLOPS | 990 TFLOPS | 362 TFLOPS |
| Architecture | Ada Lovelace | Ampere | Ampere | Hopper | Ada Lovelace |
| Best for | Inference (small) | Training/Inference | Training (large) | Everything | Inference (mid) |

### Cloud Pricing (February 2026)

| GPU | RunPod On-Demand | RunPod Spot | Vast.ai | Lambda Labs |
|-----|-----------------|-------------|---------|-------------|
| RTX 4090 24GB | $0.59/hr | $0.29/hr | $0.28/hr | — |
| A100 40GB | $0.60/hr | — | $0.52/hr | $1.29/hr |
| A100 80GB (SXM) | $1.49/hr | $0.95/hr | $0.67/hr | $1.79/hr |
| H100 80GB | $2.39/hr | $1.25/hr | $1.55/hr | $2.99/hr |
| L40S 48GB | $0.86/hr | $0.26/hr | $0.47/hr | — |

**Budget picks:**
- **Best value overall:** Vast.ai A100 80GB at $0.67/hr
- **Cheapest usable:** RunPod L40S spot at $0.26/hr (48GB, good for inference)
- **Best for training:** Vast.ai A100 80GB at $0.67/hr (high bandwidth + 80GB VRAM)

### Cost-Per-Experiment Calculations

**Scenario: Full MedQA evaluation (1,273 questions)**

| GPU + Setup | Time | Cost (Vast.ai) | Cost (RunPod) |
|-------------|------|-----------------|---------------|
| RTX 4090, current setup | ~23 hr | — (your own GPU) | — |
| RTX 4090, vLLM + optimized | ~3 hr | — (your own GPU) | — |
| A100 80GB, vLLM + optimized | ~45 min | $0.50 | $1.12 |
| H100 80GB, vLLM + optimized | ~20 min | $0.52 | $0.80 |

**Scenario: SFT fine-tuning (1 epoch, 500 examples, QLoRA)**

| GPU | Time estimate | Cost (Vast.ai) | Cost (RunPod) |
|-----|--------------|-----------------|---------------|
| A100 80GB | ~2-4 hours | $1.34-2.68 | $2.98-5.96 |
| H100 80GB | ~1-2 hours | $1.55-3.10 | $2.39-4.78 |

**Scenario: GRPO training (500 steps, 4 generations/prompt)**

| GPU | Time estimate | Cost (Vast.ai) | Cost (RunPod) |
|-----|--------------|-----------------|---------------|
| A100 80GB | ~6-12 hours | $4.02-8.04 | $8.94-17.88 |
| H100 80GB | ~3-6 hours | $4.65-9.30 | $7.17-14.34 |

### Recommendations by Task

| Task | Minimum GPU | Recommended GPU | Why |
|------|------------|-----------------|-----|
| MedQA evaluation | RTX 4090 (your card) | RTX 4090 with vLLM | Free, fast enough with optimizations |
| Multi-benchmark eval | RTX 4090 | A100 80GB | More VRAM = larger batches = faster |
| Prompt iteration (quick tests) | RTX 4090 | RTX 4090 | Low latency for small runs |
| SFT fine-tuning (QLoRA) | A100 40GB | A100 80GB | Needs VRAM for gradients + optimizer |
| GRPO training | A100 80GB | A100 80GB or H100 | generations_per_prompt eats VRAM |
| Full pipeline (eval → SFT → GRPO) | A100 80GB | H100 80GB | Speed matters for iteration cycles |

### The RTX 4090 Is Surprisingly Capable

Don't dismiss your current card too quickly:
- **Evaluation:** With vLLM optimizations, your 4090 can handle MedQA in 2-3 hours
- **Small SFT runs:** QLoRA with Unsloth can fit on 24 GB for small datasets
- **Prompt iteration:** For testing 50-question subsets, the 4090 is fine

**When to upgrade:** Rent a cloud GPU when you need to:
- Run GRPO (needs 40+ GB VRAM)
- Fine-tune with larger batch sizes or longer contexts
- Run the full evaluation suite across 4+ benchmarks in one session
- Need the run done in under an hour

---

## 7. Fine-Tuning Speed Guide

### Unsloth: Your Secret Weapon

Unsloth is a library that optimizes LLM fine-tuning with custom Triton kernels. It's
specifically designed for QLoRA workflows and delivers:

- **2-5x faster training** vs standard HuggingFace/TRL
- **50-74% less VRAM** usage
- **Zero accuracy loss** (mathematically equivalent, just faster)

**How Unsloth achieves this:**
- Manually derived backpropagation steps rewritten as fused Triton kernels
- Optimized memory layout to minimize GPU memory allocation/deallocation
- Padding-free approach: automatically removes padding tokens, saving 30%+ memory
- Optional sequence packing: fits multiple short examples into one sequence (up to 5x faster)

You're already using an Unsloth-quantized model (`unsloth/medgemma-27b-text-it-unsloth-bnb-4bit`).
For training, use Unsloth's `FastLanguageModel`:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # LoRA rank
    lora_alpha=64,           # scaling factor (typically 2 * r)
    target_modules="all-linear",
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # Unsloth-optimized version
)
```

### QLoRA Memory Math

Let's calculate exactly how much VRAM you need for QLoRA training of MedGemma 27B:

**Model weights (frozen, 4-bit):** ~14 GB

**LoRA adapters (trainable, FP16):**
- With r=32 and "all-linear" targets
- Approximate trainable params: ~200M (typical for r=32 on a 27B model)
- Memory: 200M × 2 bytes = ~0.4 GB

**Optimizer states (AdamW):**
- Per trainable param: 8 bytes (2 for momentum, 2 for variance, 4 for master copy)
- Memory: 200M × 8 = ~1.6 GB

**Gradient storage:**
- Per trainable param: 2 bytes (FP16)
- Memory: 200M × 2 = ~0.4 GB

**Activations (with gradient checkpointing):**
- Heavily depends on batch size and sequence length
- batch=2, seq_len=2048: ~2-6 GB (with checkpointing)
- batch=4, seq_len=2048: ~4-10 GB (with checkpointing)

**Total VRAM estimate:**

| Component | batch=1, seq=2048 | batch=2, seq=2048 | batch=4, seq=2048 |
|-----------|-------------------|-------------------|-------------------|
| Weights (4-bit) | 14 GB | 14 GB | 14 GB |
| LoRA adapters | 0.4 GB | 0.4 GB | 0.4 GB |
| Optimizer | 1.6 GB | 1.6 GB | 1.6 GB |
| Gradients | 0.4 GB | 0.4 GB | 0.4 GB |
| Activations | 2 GB | 4 GB | 8 GB |
| CUDA overhead | 2 GB | 2 GB | 2 GB |
| **Total** | **~20 GB** | **~22 GB** | **~26 GB** |
| Fits on 24GB? | Yes (tight) | Borderline | No |
| Fits on 40GB? | Yes | Yes | Yes |
| Fits on 80GB? | Yes (lots of room) | Yes (lots of room) | Yes (lots of room) |

**Key insight:** Your RTX 4090 (24 GB) can technically do QLoRA training with batch=1
and short sequences, but it's painfully slow. An A100 80GB lets you use batch=4+ with
gradient accumulation, which is 4-8x faster per epoch.

### LoRA Rank Selection

**What is LoRA rank (r)?** It controls the dimensionality of the low-rank adaptation
matrices. Higher rank = more trainable parameters = more capacity to learn = more memory.

**Recommended for medical domain fine-tuning: r=32**

Here's why:

| Rank (r) | Trainable Params | Memory | Best For |
|----------|-----------------|--------|----------|
| 4-8 | ~25-50M | Minimal | Simple instruction tuning, style transfer |
| 16 | ~100M | Low | General domain adaptation |
| **32** | **~200M** | **Moderate** | **Domain-specific tasks (medical, legal, scientific)** |
| 64 | ~400M | Higher | Complex multi-task learning |
| 128 | ~800M | High | Near full fine-tuning capacity |

**Why r=32 for medical:**
- Medical reasoning requires learning specialized vocabulary, clinical decision patterns,
  and domain-specific relationships (drug interactions, diagnostic criteria)
- r=16 often underfits on domain shifts this large
- r=64+ shows diminishing returns in medical benchmarks — the improvement from 32→64 is
  typically <1% accuracy while doubling trainable parameters
- Your roadmap's `LoraConfig(r=32, lora_alpha=64)` is well-chosen

**Alpha/rank ratio:** `lora_alpha = 2 * r` is the standard starting point. This means
the effective learning rate for LoRA weights is scaled by `alpha/r = 2`. Some practitioners
use `alpha = r` (scaling factor = 1) for more conservative adaptation.

### Gradient Checkpointing

**What it does:** Instead of storing all intermediate activations during the forward pass
(needed for backpropagation), gradient checkpointing only stores activations at certain
"checkpoint" layers. The discarded activations are recomputed during the backward pass.

**Trade-off:**
- Saves 50-70% of activation memory
- Costs 20-30% more compute time (recomputation overhead)
- Usually worth it because the memory savings let you use larger batches, which more than
  compensates for the slowdown

**Always enable gradient checkpointing for 27B model training.** It's the difference
between OOM and successful training.

```python
# Standard HuggingFace
model.gradient_checkpointing_enable()

# Unsloth (optimized version, slightly faster than HF's implementation)
use_gradient_checkpointing="unsloth"
```

### Optimal Training Configuration

For MedGemma 27B QLoRA on A100 80GB:

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=2,     # 2 examples per step
    gradient_accumulation_steps=8,      # effective batch = 16
    num_train_epochs=1,                 # 1 epoch for medical SFT
    learning_rate=2e-4,                 # standard for QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=2048,
    fp16=True,                          # or bf16=True on A100
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    dataset_text_field="text",          # or use formatting_func
)
```

**Expected training speed on A100 80GB with Unsloth:**
- ~2,000-4,000 tokens/sec
- ~3,500-5,000 samples/hour (at 2048 token sequences)
- 500 training examples × 1 epoch: ~6-10 minutes
- 500 examples × 3 epochs: ~20-30 minutes

**On your RTX 4090 (if you must):**
- ~800-1,500 tokens/sec (2-3x slower)
- batch_size=1 with gradient_accumulation=16
- 500 examples × 1 epoch: ~20-30 minutes
- Workable for small experiments, not for iteration

---

## 8. GRPO/RL Training Speed

### What Makes GRPO Different

GRPO (Group Relative Policy Optimization) works differently from SFT:

1. **Generation phase:** For each training prompt, generate multiple responses
   (controlled by `generations_per_prompt`, typically 4-8)
2. **Scoring phase:** Score each response with reward functions
3. **Training phase:** Update the model to increase probability of high-scoring
   responses relative to low-scoring ones within each group

The generation phase is what makes GRPO expensive. You're running inference
`generations_per_prompt` times for every training example, and each generation is
autoregressive (slow, token-by-token).

### Memory Requirements

**GRPO memory = SFT memory + generation memory × generations_per_prompt**

For MedGemma 27B with `generations_per_prompt=4`:

| Component | Memory |
|-----------|--------|
| Model weights (4-bit) | 14 GB |
| LoRA adapters | 0.4 GB |
| Optimizer states | 1.6 GB |
| Gradients | 0.4 GB |
| Training activations | 4-8 GB |
| Generation KV-cache (4 sequences) | 1.5-3 GB |
| Reference model logits | 2-4 GB |
| CUDA overhead | 2 GB |
| **Total** | **~26-34 GB** |

**This is why GRPO needs 40+ GB VRAM.** Your RTX 4090 (24 GB) cannot realistically
run GRPO on a 27B model. You need at minimum an A100 40GB, ideally an A100 80GB.

With `generations_per_prompt=8`, memory requirements increase to ~35-45 GB.

### Framework Comparison

| Feature | TRL (HuggingFace) | OpenRLHF | veRL |
|---------|-------------------|----------|------|
| **Speed** | Baseline | **3.1x faster** | ~2x faster |
| **Ease of use** | Easiest (HF ecosystem) | Moderate | Complex (200+ hyperparams) |
| **Memory efficiency** | Good | Better (distributed) | Best (weight sharing) |
| **Multi-GPU** | DeepSpeed/FSDP | Ray-based | FSDP/Megatron |
| **Algorithms** | PPO, GRPO | PPO, GRPO, REINFORCE++ | PPO, GRPO |
| **Documentation** | Excellent | Good | Growing |
| **Unsloth compat** | Yes | Partial | No |
| **Code complexity** | 19K LOC | 8.5K LOC | 32K LOC |

**Recommendation: Start with TRL GRPOTrainer + Unsloth**

- TRL integrates seamlessly with your existing HuggingFace workflow
- Unsloth's GRPO optimizations reduce VRAM by 50-90%
- If speed becomes a bottleneck, graduate to OpenRLHF (3x faster)

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,                  # generations per prompt
    per_device_train_batch_size=1,      # keep low for 27B
    gradient_accumulation_steps=4,
    learning_rate=5e-6,                 # 10x lower than SFT
    max_completion_length=512,
    num_train_epochs=1,
    bf16=True,
    logging_steps=5,
)

# Define reward functions (from your roadmap)
def correctness_reward(completions, correct_answers):
    """Binary reward: does the model get the right answer?"""
    rewards = []
    for completion, answer in zip(completions, correct_answers):
        model_answer, _ = extract_answer(completion)
        rewards.append(1.0 if model_answer == answer else 0.0)
    return rewards

def format_reward(completions):
    """Does the output follow the expected format?"""
    rewards = []
    for completion in completions:
        has_format = bool(re.search(r"[Tt]he answer is\s*\(?[A-D]\)?", completion))
        rewards.append(0.5 if has_format else 0.0)
    return rewards

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    reward_funcs=[correctness_reward, format_reward],
    processing_class=tokenizer,
)

trainer.train()
```

### Practical GRPO Speed Tips

1. **Keep generations_per_prompt low (4, not 8):** Memory scales linearly, and 4
   generations gives enough variance for group relative optimization. 8 is only needed
   if you're seeing high variance in rewards.

2. **Use Unsloth's memory optimizations:**
   ```python
   from unsloth import FastLanguageModel
   # Unsloth automatically optimizes GRPO memory layout
   ```

3. **Reduce max_completion_length:** For MCQ tasks, 512 tokens is plenty. Don't waste
   VRAM generating long completions that will score 0.

4. **Use vLLM for the generation phase:** Some frameworks (OpenRLHF, veRL) use vLLM
   as the inference backend during GRPO's generation step, which is much faster than
   HuggingFace generate.

5. **Gradient accumulation over larger batch sizes:** Using batch=1 with grad_accum=8
   uses much less peak VRAM than batch=8 with grad_accum=1, for the same effective batch.

---

## 9. Benchmark Evaluation at Scale

### Your Benchmark Suite

From the roadmap, you'll eventually evaluate on:

| Benchmark | Questions | Type | Est. Time (current) | Est. Time (optimized) |
|-----------|-----------|------|---------------------|-----------------------|
| MedQA | 1,273 | 4-choice MCQ | ~23 hours | ~1-2 hours |
| PubMedQA | 500 | Yes/No/Maybe | ~9 hours | ~30 min |
| MedMCQA | 4,183 | 4-choice MCQ | ~75 hours | ~3-5 hours |
| MMLU-Medical* | ~1,800 | 4-choice MCQ | ~32 hours | ~1.5-2.5 hours |
| EHRQA (custom) | ~500 | Free-form | ~9 hours | ~30-60 min |
| CT-FHIR Delta | ~200-500 | Classification | ~4-9 hours | ~15-30 min |
| **Total** | **~8,500** | | **~152 hours** | **~7-11 hours** |

*\*MMLU-Medical includes: clinical knowledge, medical genetics, anatomy, professional medicine, college medicine, college biology*

**With your current setup, running all benchmarks takes nearly a week of continuous GPU time.**
**With optimizations on the same RTX 4090, it takes under half a day.**

### Running Multiple Benchmarks Efficiently

**Strategy 1: Sequential with vLLM (simplest)**

Load the model once with vLLM, run all benchmarks sequentially:

```python
from vllm import LLM, SamplingParams

# Load once
llm = LLM(model="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
           quantization="bitsandbytes", gpu_memory_utilization=0.90)

# Run each benchmark
for benchmark in ["medqa", "pubmedqa", "medmcqa", "mmlu_medical"]:
    prompts = load_benchmark_prompts(benchmark)
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=512))
    save_results(benchmark, outputs)
```

Model loading takes ~2-5 minutes. By loading once and running all benchmarks, you save
significant overhead vs loading/unloading for each benchmark.

**Strategy 2: Prompt caching (advanced)**

Many benchmarks share the same system prompt. The prefill computation for the system
prompt can be cached and reused across questions:

- MedQA, MedMCQA, MMLU all use similar "You are a medical expert..." system prompts
- vLLM supports **prefix caching**: if multiple prompts share a prefix, the KV-cache for
  that prefix is computed once and reused
- Enable with `--enable-prefix-caching` in vLLM
- Typical savings: 10-20% for short shared prefixes, more for longer ones

**Strategy 3: Multi-GPU parallelization**

If you rent a multi-GPU instance (e.g., 2x A100 80GB):

```bash
# Tensor parallelism: split model across GPUs (lower latency)
vllm serve model_name --tensor-parallel-size 2

# OR pipeline parallelism: one GPU does first half of layers, other does second half
# Generally tensor parallelism is preferred for inference
```

With 2 GPUs, you can either:
- Run the model faster (tensor parallel) — ~2x speedup per question
- Run two models independently — ~2x throughput (different benchmarks simultaneously)

### Evaluation Harness Integration

Consider using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(by EleutherAI) for standardized benchmark evaluation:

```bash
pip install lm-eval

# Run MedQA with vLLM backend
lm_eval --model vllm \
    --model_args pretrained=unsloth/medgemma-27b-text-it-unsloth-bnb-4bit,quantization=bitsandbytes \
    --tasks medqa \
    --batch_size auto \
    --output_path results/
```

**Pros:**
- Standardized evaluation protocols (reproducible, comparable to published results)
- Built-in support for vLLM, HuggingFace, and many other backends
- Handles batching, prompt formatting, and metric computation automatically
- Supports most medical benchmarks out of the box

**Cons:**
- Less control over prompt formatting (your custom system prompt may need adaptation)
- May not support your custom benchmarks (EHRQA, CT-FHIR Delta) without writing a task config
- Harder to debug answer extraction issues

**Recommendation:** Use lm-eval-harness for standard benchmarks (MedQA, PubMedQA, MedMCQA,
MMLU) and keep your custom `eval_medqa.py` approach for custom benchmarks (EHRQA, CT-FHIR).

---

## 10. Concrete Recommendations

### Priority 1: Free, Immediate (Do Today)

**1a. Reduce max_new_tokens from 5,000 to 512**
```bash
python medqa_eval/eval_medqa.py --max-new-tokens 512 --max-samples 50
```
- Expected speedup: 2-3x
- Risk: None (verify by checking your existing results' output_tokens distribution)
- Effort: Change one CLI argument

**1b. Install Flash Attention 2**
```bash
pip install flash-attn --no-build-isolation
```
- Expected speedup: 10-30% on prompt processing
- Risk: None
- Effort: One pip install (may need to compile, takes a few minutes)

### Priority 2: Low Effort, Big Impact (This Week)

**2a. Switch to vLLM for evaluation**
```bash
pip install vllm
```
Modify `eval_medqa.py` to use vLLM's `LLM` class (see Section 5 code example).

- Expected speedup: 5-15x over current setup
- Risk: Low (vLLM is mature and well-tested)
- Effort: ~1 hour to modify the eval script

**2b. Consider AWQ quantization**
If an AWQ-quantized MedGemma 27B exists on HuggingFace, use it with vLLM for an
additional ~4x throughput boost over BnB quantization.

- Search HuggingFace for: `medgemma-27b AWQ` or `medgemma-27b GPTQ`
- If none exists, you can quantize it yourself (requires a machine with 64+ GB RAM
  to load the full-precision model)

### Priority 3: When You Start Fine-Tuning

**3a. Use Unsloth for all training**
Already part of your workflow via the quantized model. For training:
```bash
pip install unsloth
```
Use `FastLanguageModel` for 2-5x speedup and 50-70% memory reduction.

**3b. Rent an A100 80GB for SFT**
- Vast.ai: ~$0.67/hr
- RunPod: ~$1.49/hr (spot: $0.95/hr)
- Total cost for SFT: ~$1-5 per training run

**3c. Stick with r=32 for LoRA rank**
Your roadmap already specifies this. It's the sweet spot for medical domain adaptation.

### Priority 4: When You Start GRPO

**4a. Rent an A100 80GB (minimum) or H100**
GRPO on 27B absolutely requires 40+ GB VRAM. Budget $5-20 per training run.

**4b. Start with TRL GRPOTrainer + Unsloth**
Easiest to set up, good Unsloth integration.

**4c. If speed becomes a bottleneck, switch to OpenRLHF**
3x faster than TRL, but more complex setup. Only worth it if you're iterating
on reward functions and need fast experiment cycles.

### Priority 5: Scaling Up (When Running Full Benchmark Suite)

**5a. Use lm-evaluation-harness for standard benchmarks**
One-time setup, then run all standard medical benchmarks with a single command.

**5b. Create a benchmark runner script**
A single script that loads the model once and runs all benchmarks sequentially:
```bash
python run_all_benchmarks.py \
    --model unsloth/medgemma-27b-text-it-unsloth-bnb-4bit \
    --benchmarks medqa pubmedqa medmcqa mmlu_medical ehrqa ct_fhir \
    --output-dir results/full_suite/
```

**5c. Enable prefix caching in vLLM**
Free performance boost when running multiple benchmarks with shared system prompts.

---

## Quick Reference: Expected Timeline

| Phase | Task | Time (Current) | Time (Optimized) | GPU Needed |
|-------|------|----------------|-------------------|------------|
| 1A | MedQA baseline | 23 hours | 1-2 hours | RTX 4090 |
| 1B | EHRQA reconstruction | ~2 days | ~4-6 hours | RTX 4090 |
| 2 | CT-FHIR benchmark creation | N/A (scripting) | N/A | Any |
| 2C | CT-FHIR baseline eval | ~9 hours | ~30 min | RTX 4090 |
| 3 | SFT fine-tuning | Impossible (24GB) | ~30-60 min | A100 80GB |
| 4 | GRPO training | Impossible (24GB) | ~6-12 hours | A100 80GB |
| 5 | Full benchmark suite | ~152 hours | ~7-11 hours | RTX 4090 or A100 |

**Total cost estimate for the full pipeline (cloud GPU rental):**
- SFT: ~$1-5
- GRPO: ~$5-20
- Evaluation on cloud (if desired): ~$5-10
- **Total: ~$11-35** for the entire research project's compute

---

## Appendix: Key Formulas

**Tokens per second (generation):**
```
tokens/sec ≈ memory_bandwidth_GB/s / (model_size_GB / batch_size)
```

**KV-cache size per token:**
```
bytes = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
MedGemma: 2 × 16 × 128 × 46 × 2 = 376,832 bytes ≈ 0.36 MB
```

**QLoRA VRAM estimate:**
```
VRAM = model_4bit + lora_params×2 + optimizer×8 + activations(batch, seq_len)
```

**GRPO VRAM multiplier:**
```
VRAM_grpo ≈ VRAM_sft + KV_cache × generations_per_prompt + ref_model_logits
```
