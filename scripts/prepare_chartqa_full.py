import argparse
import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "HuggingFaceM4/ChartQA"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def export_split(split: str, output_dir: Path) -> int:
    jsonl_path = output_dir / f"{split}.jsonl"
    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    completed_count = count_existing_rows(jsonl_path)
    mode = "a" if completed_count else "w"

    print(f"\nPreparing split: {split}")
    print(f"Already completed: {completed_count}")

    dataset = load_dataset(DATASET_NAME, split=split, streaming=True)

    with jsonl_path.open(mode, encoding="utf-8") as output_file:
        for index, sample in enumerate(dataset):
            if index < completed_count:
                continue

            image_path = image_dir / f"{index:06d}.jpg"
            sample["image"].convert("RGB").save(image_path, "JPEG", quality=95)

            answers = sample["label"]
            if not isinstance(answers, list):
                answers = [answers]

            record = {
                "id": f"chartqa-{split}-{index:06d}",
                "image": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "prompt": sample["query"],
                "response": str(answers[0]),
                "all_answers": [str(answer) for answer in answers],
                "human_or_machine": sample.get("human_or_machine", "unknown"),
            }

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

            if (index + 1) % 1000 == 0:
                print(f"Completed {index + 1} samples in {split}")

    total_count = count_existing_rows(jsonl_path)
    print(f"Finished {split}: {total_count} samples")
    return total_count


def main():
    parser = argparse.ArgumentParser(
        description="Download full ChartQA train/val data into the project format."
    )
    parser.add_argument(
        "--output-dir",
        default="data/chartqa_full",
        help="Directory for images and JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to export. Test is intentionally excluded by default.",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in args.splits:
        counts[split] = export_split(split, output_dir)

    metadata = {
        "dataset": DATASET_NAME,
        "splits": counts,
        "format": "image + prompt + response + all_answers",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Full ChartQA preparation complete ===")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()