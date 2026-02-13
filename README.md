# Sentinel-X Foundry

MedGemma fine-tuning & evaluation for CT-FHIR Delta Analysis.
See [roadmap.md](roadmap.md) for the full research plan.

---

## Setup

```bash
pip install unsloth transformers datasets torch bitsandbytes accelerate trl
```

**Hardware requirement:** NVIDIA GPU with >= 20 GB VRAM (RTX 4090 / A100).

---

## Running Scripts

### MedQA Baseline Evaluation

Evaluates MedGemma 27B (4-bit) on the MedQA benchmark (1,273 MCQs).
All output goes to a **single JSONL file** that appends live as the model answers each question.

```bash
# Sanity check — 50 questions (~5-10 min)
python medqa_eval/eval_medqa.py --max-samples 50

# Full evaluation — 1,273 questions (~2-3 hours)
python medqa_eval/eval_medqa.py
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `unsloth/medgemma-27b-text-it-unsloth-bnb-4bit` | HuggingFace model ID |
| `--max-samples` | all 1,273 | Limit number of questions |
| `--output-dir` | `medqa_eval/results` | Directory for the output JSONL |
| `--temperature` | `0.0` | Sampling temperature (0 = deterministic) |
| `--max-new-tokens` | `5000` | Max tokens per generation |
| `--batch-size` | `1` | Batch size (1 is safe for 24 GB VRAM) |

**Output format** (`medqa_eval/results/medqa_eval_<timestamp>.jsonl`):

```
Line 1      {"type": "run_header", "model_id": "...", ...}
Lines 2-N   {"type": "question", "question_id": 0, "is_correct": true, ...}
Last line    {"type": "summary", "accuracy": 0.877, ...}
```

You can tail the file in real-time while the eval runs:

```bash
tail -f medqa_eval/results/medqa_eval_*.jsonl
```

Ctrl+C saves a partial summary (tagged `"partial": true`) so no work is lost.

---

### Dataset Inspection (test utility)

Prints dataset metadata, sample questions, and answer distribution — no GPU needed.

```bash
python medqa_eval/tests/inspect_dataset.py
```

---

## Project Structure

```
Sentinel-X-foundry/
├── README.md                 # <-- you are here
├── roadmap.md                # Full research roadmap & key metrics
├── medqa_eval/
│   ├── eval_medqa.py         # MedQA evaluation script
│   ├── results/              # Output JSONL files (git-ignored)
│   └── tests/
│       └── inspect_dataset.py  # Dataset inspection utility
└── .gitignore
```
