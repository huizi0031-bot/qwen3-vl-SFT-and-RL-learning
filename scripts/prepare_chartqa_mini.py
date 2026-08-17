import argparse
import json
from itertools import islice
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "HuggingFaceM4/ChartQA"


def pick_eval_split(available_splits: list[str]) -> str:
    for name in ("validation", "val", "test"):
        if name in available_splits:
            return name
    raise ValueError(f"Cannot find evaluation split in: {available_splits}")


def export_split(split_name: str, size: int, output_dir: Path) -> int:
    image_dir = output_dir / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split_name}.jsonl"

    stream = load_dataset(DATASET_NAME, split=split_name, streaming=True)

    count = 0
    with jsonl_path.open("w", encoding="utf-8") as file:
        for index, sample in enumerate(islice(stream, size)):
            image_path = image_dir / f"{index:06d}.jpg"
            sample["image"].convert("RGB").save(image_path, quality=95)

            answers = sample["label"]
            record = {
                "id": f"chartqa-{split_name}-{index:06d}",
                "image": str(image_path.relative_to(PROJECT_ROOT)),
                "prompt": sample["query"],
                "response": answers[0],
                "all_answers": answers,
                "human_or_machine": sample["human_or_machine"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--output-dir", default="data/chartqa_mini")
    args = parser.parse_args()

    available_splits = get_dataset_split_names(DATASET_NAME)
    if "train" not in available_splits:
        raise ValueError(f"ChartQA train split not found: {available_splits}")

    eval_split = pick_eval_split(available_splits)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    print(f"Available splits: {available_splits}")
    print(f"Exporting {args.train_size} train samples...")
    train_count = export_split("train", args.train_size, output_dir)

    print(f"Exporting {args.eval_size} {eval_split} samples...")
    eval_count = export_split(eval_split, args.eval_size, output_dir)

    metadata = {
        "dataset": DATASET_NAME,
        "train_split": "train",
        "eval_split": eval_split,
        "train_samples": train_count,
        "eval_samples": eval_count,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n=== ChartQA mini dataset ready ===")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()