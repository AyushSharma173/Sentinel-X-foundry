# Sentinel-X: MedGemma Fine-Tuning & Evaluation Roadmap

## The Big Picture

We're doing three things simultaneously:
1. **Win the Novel Task Prize** → CT-FHIR Delta Analysis as a new clinical reasoning task
2. **Produce world-class research content** → Publishable findings about fine-tuning medical LLMs
3. **Create lasting technical artifacts** → Benchmarks, fine-tuned models, evaluation frameworks

The fine-tuning work is the core engine that powers all three goals.

---

## Phase 1: Baseline Evaluation (START HERE)

### 1A. MedQA Baseline (~2 hours)
**Why**: Quick sanity check. Google reports 87.7%. If we get significantly different, 
something is wrong with our setup.

```bash
# Quick test (50 questions, ~30 min on RTX 4090)
python scripts/eval_medqa.py \
    --use-unsloth \
    --max-samples 50 \
    --output-dir results/medqa

# Full evaluation (1,273 questions, ~2-3 hours on RTX 4090)
python scripts/eval_medqa.py \
    --use-unsloth \
    --output-dir results/medqa
```

**What we learn:**
- Confirms model loading pipeline works
- Gives us a publishable "before" number
- Tests our answer extraction regex (important for all downstream evals)
- Baseline reasoning quality on medical questions

**Interesting variations to test:**
- With vs without system prompt ("You are a helpful medical assistant")
- Concise mode vs full reasoning (token efficiency vs accuracy tradeoff)
- Temperature 0.0 vs 0.1 (Google uses 0.0 for benchmarks)

### 1B. EHRQA Reconstruction (~1-2 days)
**Why**: This is the closest existing benchmark to CT-FHIR Delta Analysis. Google reports 
86.3% baseline → 93.6% after RL. Reproducing this gives us:
- A validated evaluation framework for FHIR reasoning
- Understanding of exactly which question types the model struggles with
- A direct comparison point for our fine-tuning results

```bash
# Option A: Use pre-generated Synthea data (fastest)
# Download from https://synthea.mitre.org/downloads
# Unzip FHIR R4 bundle into data/synthea_patients/fhir/

# Option B: Generate fresh Synthea patients (requires Java 11+)
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./gradlew build
cd ..
python scripts/ehrqa_reconstruct.py generate-patients \
    --count 100 \
    --output data/synthea_patients

# Generate QA pairs from FHIR bundles
python scripts/ehrqa_reconstruct.py generate-qa \
    --input data/synthea_patients/fhir \
    --output data/ehrqa

# Evaluate MedGemma 27B on the benchmark
python scripts/eval_ehrqa.py \
    --dataset data/ehrqa/ehrqa_test.jsonl \
    --model unsloth/medgemma-27b-text-it-unsloth-bnb-4bit \
    --output-dir results/ehrqa
```

**Key question types to focus on (from the technical report):**
- Multi-hop reasoning across interdependent records (biggest RL gains)
- Temporal reasoning (condition progression over time)
- Medication-condition relationships
- Lab trend analysis

### 1C. MedGemma 1.5 4B vs 27B Comparison (optional, 1 day)
**Why**: Google updated MedGemma to 1.5 (4B only so far). Interesting to compare:
- MedGemma 1 4B on EHRQA (68% reported)
- MedGemma 1.5 4B on EHRQA (90% reported — huge jump!)
- MedGemma 1 27B on EHRQA (86.3% reported, 93.6% after RL)
- The 1.5 4B OUTPERFORMS the 1.0 27B baseline. That's a remarkable finding.

This comparison is already interesting research content: "Does model scale matter
for FHIR reasoning, or is training data the dominant factor?"

---

## Phase 2: Build CT-FHIR Delta Analysis Benchmark

### 2A. Define the Task Formally
CT-FHIR Delta Analysis: Given CT imaging findings and a patient's FHIR clinical history,
classify each finding into one of four categories:

| Category | Definition | Example |
|----------|-----------|---------|
| ACUTE_NEW | Finding not present in prior history | New pulmonary embolism in patient with no PE history |
| CHRONIC_STABLE | Finding consistent with known history | Emphysema in patient with documented COPD |
| CHRONIC_WORSENING | Finding suggesting progression | Increased pleural effusion in known CHF patient |
| DISCORDANT | Finding conflicts with documented history | Cardiomegaly noted but no cardiac conditions in FHIR |

### 2B. Generate Benchmark Dataset
This requires:
1. **Synthea patients** with disease-relevant modules enabled (COPD, cardiovascular, etc.)
2. **Simulated radiology findings** paired with each patient's FHIR record
3. **Ground-truth labels** for each finding-patient pair

The key insight from the FHIR_GENERATION_IMPROVEMENTS doc is that there's currently
~0% overlap between Synthea conditions and radiology findings. We need to:
- Enable specific Synthea modules: copd, cardiovascular_disease, lung_cancer, etc.
- Generate findings that DO have meaningful overlap with the FHIR history
- Also generate DISCORDANT cases (findings that contradict the history)

Target: 200-500 labeled examples (GRPO works with as few as 100).

### 2C. Evaluate Baseline on CT-FHIR Delta Analysis
Run MedGemma 27B on the benchmark with zero-shot prompting to establish a baseline.
This is the "before" number for fine-tuning.

---

## Phase 3: SFT Fine-Tuning (QLoRA)

### Training Setup
```
Model: unsloth/medgemma-27b-text-it-unsloth-bnb-4bit
Hardware: A100 80GB (RunPod, ~$1.19/hr)
Framework: Unsloth + TRL SFTTrainer
```

### Hyperparameters (from research report)
```python
LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
SFTConfig(lr=2e-4, batch_size=2, grad_accum=8, epochs=1, cosine scheduler)
```

### Training Data
- CT-FHIR Delta Analysis labeled examples (Phase 2)
- 10-15% general medical QA (MedQA, PubMedQA) to prevent catastrophic forgetting
- Use Gemma 3 chat template with chain-of-thought reasoning traces
- train_on_responses_only

### Evaluation After SFT
- Re-run CT-FHIR Delta Analysis benchmark
- Re-run EHRQA to check for regression
- Re-run MedQA subset to check for catastrophic forgetting

---

## Phase 4: GRPO Reinforcement Learning

### Setup
```
Base: SFT checkpoint from Phase 3
Framework: TRL GRPOTrainer with LoRA adapters
Hardware: Same A100 80GB
```

### Reward Functions (verifiable)
1. **Correctness reward**: Does classification match ground truth? (+1/0)
2. **Format compliance**: Does output follow structured format? (+0.5/0)
3. **Reasoning quality**: Are specific FHIR resources referenced? (+0.25/0)

### Hyperparameters
```python
lr=5e-6 (10x lower than SFT)
generations_per_prompt=4-8
KL penalty against SFT checkpoint
```

### Evaluation After GRPO
- Same battery as post-SFT
- Compare: Base → SFT → SFT+GRPO
- Category-level breakdown (especially multi-hop reasoning)

---

## Phase 5: Research Content & Ablation Studies

### Thread 1: "Serial Late Fusion as an Architecture Pattern"
- Separating perception from reasoning
- VRAM-efficient deployment of large models
- Publishable as a systems paper

### Thread 2: "SFT vs GRPO on Domain-Specialized Small Models"
- Systematic comparison with matched training data
- When does RL help vs SFT alone?
- Which question types benefit most from RL?
- This directly extends Google's own findings

### Thread 3: "Information-Theoretic Analysis of Clinical Context Utilization"
- How much of the FHIR context does the model actually USE?
- Token-level attention analysis
- Can we measure the information gain from FHIR data?
- Novel methodology for quantifying context utilization

### Thread 4: "First Community Fine-Tune of MedGemma 27B"
- Document the entire process
- Publish LoRA adapters on HuggingFace
- Full model card with evaluation results
- Practical guide for others

---

## File Structure

```
sentinel-x-eval/
├── scripts/
│   ├── eval_medqa.py          # MedQA baseline evaluation
│   ├── ehrqa_reconstruct.py   # EHRQA benchmark reconstruction
│   ├── eval_ehrqa.py          # EHRQA evaluation harness (TODO)
│   ├── generate_delta_benchmark.py  # CT-FHIR Delta Analysis benchmark (TODO)
│   ├── eval_delta.py          # Delta Analysis evaluation (TODO)
│   ├── train_sft.py           # SFT fine-tuning script (TODO)
│   └── train_grpo.py          # GRPO training script (TODO)
├── configs/
│   ├── lora_config.yaml       # LoRA hyperparameters
│   ├── sft_config.yaml        # SFT training config
│   └── grpo_config.yaml       # GRPO training config
├── benchmarks/
│   ├── medqa/                 # MedQA evaluation data (via HuggingFace)
│   ├── ehrqa/                 # Reconstructed EHRQA benchmark
│   └── ct_fhir_delta/         # Our novel benchmark
├── results/
│   ├── medqa/                 # MedQA evaluation results
│   ├── ehrqa/                 # EHRQA evaluation results
│   └── delta/                 # Delta Analysis results
└── data/
    ├── synthea_patients/      # Generated Synthea FHIR bundles
    └── training/              # Fine-tuning datasets
```

---

## Quick Start (What To Do Right Now)

1. **Set up the environment:**
   ```bash
   pip install unsloth transformers datasets torch bitsandbytes accelerate trl
   ```

2. **Run MedQA sanity check (50 questions):**
   ```bash
   cd sentinel-x-eval
   python scripts/eval_medqa.py --use-unsloth --max-samples 50
   ```

3. **Download Synthea data:**
   - Go to https://synthea.mitre.org/downloads
   - Download "FHIR R4" (about 1GB for 1000 patients)
   - Extract to `data/synthea_patients/fhir/`

4. **Generate EHRQA benchmark:**
   ```bash
   python scripts/ehrqa_reconstruct.py generate-qa \
       --input data/synthea_patients/fhir \
       --output data/ehrqa
   ```

5. **Evaluate on EHRQA** (need to build eval_ehrqa.py next)

---

## Key Numbers to Track

| Metric | Google Reported | Our Baseline | Post-SFT | Post-GRPO |
|--------|----------------|-------------|----------|-----------|
| MedQA Accuracy | 87.7% | TBD | TBD | TBD |
| EHRQA Overall | 86.3% | TBD | TBD | TBD |
| EHRQA Multi-hop | ~75%? | TBD | TBD | TBD |
| CT-FHIR Delta (ours) | N/A | TBD | TBD | TBD |
| EHRQA (post-RL, Google) | 93.6% | — | — | TBD |

These numbers tell the story for both the competition writeup AND the research content.