import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_records(path_text: str) -> dict:
    path = PROJECT_ROOT / path_text
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {record["id"]: record for record in records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="experiments/eval_chartqa_mini_base.jsonl",
    )
    parser.add_argument(
        "--lora",
        default="experiments/eval_chartqa_mini_lora.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/compare_chartqa_mini.json",
    )
    args = parser.parse_args()

    base_records = load_records(args.base)
    lora_records = load_records(args.lora)

    if set(base_records) != set(lora_records):
        raise ValueError("Base and LoRA evaluation files do not contain the same sample IDs.")

    improved, regressed, unchanged = [], [], []

    for sample_id in base_records:
        base = base_records[sample_id]
        lora = lora_records[sample_id]

        if not base["normalized_exact_match"] and lora["normalized_exact_match"]:
            improved.append(sample_id)
        elif base["normalized_exact_match"] and not lora["normalized_exact_match"]:
            regressed.append(sample_id)
        else:
            unchanged.append(sample_id)

    summary = {
        "samples": len(base_records),
        "base_exact_matches": sum(
            record["normalized_exact_match"] for record in base_records.values()
        ),
        "lora_exact_matches": sum(
            record["normalized_exact_match"] for record in lora_records.values()
        ),
        "improved_ids": improved,
        "regressed_ids": regressed,
        "unchanged_ids": unchanged,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=== Base vs LoRA comparison ===")
    print(f"Samples: {summary['samples']}")
    print(f"Base exact matches: {summary['base_exact_matches']}")
    print(f"LoRA exact matches: {summary['lora_exact_matches']}")
    print(f"Improved: {len(improved)}")
    print(f"Regressed: {len(regressed)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()