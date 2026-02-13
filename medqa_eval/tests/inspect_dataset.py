"""Quick inspection of the MedQA dataset — prints metadata, sample questions, and answer distribution."""

from collections import Counter

from datasets import load_dataset

DATASET_ID = "openlifescienceai/medqa"
NUM_PREVIEW = 15


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {DATASET_ID} (test split) ...")
    ds = load_dataset(DATASET_ID, split="test")

    # ------------------------------------------------------------------
    # 2. Dataset-level info
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DATASET INFO")
    print(f"{'='*60}")
    print(f"Total questions : {len(ds)}")
    print(f"Column names    : {ds.column_names}")
    print(f"Features        : {ds.features}")

    # Show available splits
    ds_dict = load_dataset(DATASET_ID)
    print(f"Available splits: {list(ds_dict.keys())}")

    # ------------------------------------------------------------------
    # 3. Pretty-print first N questions
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FIRST {NUM_PREVIEW} QUESTIONS")
    print(f"{'='*60}")

    for i in range(min(NUM_PREVIEW, len(ds))):
        data = ds[i]["data"]
        question = data["Question"]
        options = data["Options"]
        correct = data["Correct Option"]

        print(f"\n--- Question {i + 1} ---")
        print(f"Q: {question}")
        for letter, text in sorted(options.items()):
            marker = " <-- correct" if letter == correct else ""
            print(f"  {letter}) {text}{marker}")
        print(f"Answer: {correct}")

    # ------------------------------------------------------------------
    # 4. Answer distribution across the full dataset
    # ------------------------------------------------------------------
    counts = Counter()
    for row in ds:
        counts[row["data"]["Correct Option"]] += 1

    print(f"\n{'='*60}")
    print("ANSWER DISTRIBUTION (full dataset)")
    print(f"{'='*60}")
    for letter in sorted(counts):
        pct = 100 * counts[letter] / len(ds)
        print(f"  {letter}: {counts[letter]:>5}  ({pct:.1f}%)")
    print(f"  Total: {len(ds)}")


if __name__ == "__main__":
    main()
