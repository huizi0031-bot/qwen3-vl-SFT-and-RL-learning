import argparse
from itertools import islice

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    dataset = load_dataset(
        "HuggingFaceM4/ChartQA",
        split=args.split,
        streaming=True,
    )

    for index, sample in enumerate(islice(dataset, args.num_samples), start=1):
        image = sample["image"]
        answers = sample["label"]

        print(f"\n=== ChartQA sample {index} ===")
        print(f"Fields: {list(sample.keys())}")
        print(f"Image size: {image.size}")
        print(f"Question: {sample['query']}")
        print(f"Answers: {answers}")
        print(f"Source flag (human_or_machine): {sample['human_or_machine']}")


if __name__ == "__main__":
    main()