import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {"id", "image", "prompt", "response", "all_answers", "human_or_machine"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/chartqa_mini/train.jsonl")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    source_counts = Counter()
    invalid_rows = []
    first_record = None
    total = 0

    with data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            missing_fields = REQUIRED_FIELDS - set(record)
            image_path = PROJECT_ROOT / record.get("image", "")
            answers = record.get("all_answers", [])

            if (
                missing_fields
                or not image_path.is_file()
                or not isinstance(record.get("response"), str)
                or not isinstance(answers, list)
                or not answers
            ):
                invalid_rows.append(
                    {
                        "line": line_number,
                        "missing_fields": sorted(missing_fields),
                        "image_exists": image_path.is_file(),
                    }
                )

            source_counts[str(record.get("human_or_machine"))] += 1

            if first_record is None:
                first_record = record

    print(f"Dataset: {data_path}")
    print(f"Samples: {total}")
    print(f"Source distribution: {dict(source_counts)}")
    print(f"Invalid rows: {len(invalid_rows)}")

    if first_record:
        print("\n=== First converted record ===")
        print(json.dumps(first_record, ensure_ascii=False, indent=2))

    if invalid_rows:
        print("\n=== Invalid-row details ===")
        print(json.dumps(invalid_rows[:10], ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()