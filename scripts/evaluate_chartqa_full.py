import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,!?:;，。！？：；")


def parse_number(value: str):
    value = normalize_text(value)
    value = value.replace(",", "").replace("$", "")

    if value.endswith("%"):
        value = value[:-1]

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return float(value)

    return None


def relaxed_match(prediction: str, answers: list[str]) -> tuple[bool, str]:
    """
    先做文本精确匹配；
    再做数值的相对误差 <= 5% 匹配。
    """
    normalized_prediction = normalize_text(prediction)

    for answer in answers:
        normalized_answer = normalize_text(answer)

        if normalized_prediction == normalized_answer:
            return True, "exact"

        prediction_number = parse_number(prediction)
        answer_number = parse_number(answer)

        if prediction_number is None or answer_number is None:
            continue

        if answer_number == 0:
            if prediction_number == 0:
                return True, "numeric_exact"
        elif abs(prediction_number - answer_number) / abs(answer_number) <= 0.05:
            return True, "numeric_relaxed"

    return False, "incorrect"


def build_model(adapter_path: str | None, device: str):
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    if adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            resolve_path(adapter_path),
        )
    else:
        model = base_model

    return model.eval()


def generate_answer(model, processor, sample, device, image_height, image_width,
                    max_new_tokens):
    image_path = resolve_path(sample["image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 保持与训练相同的图片尺寸和问题形式，保证评估公平。
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "resized_height": image_height,
                    "resized_width": image_width,
                },
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, _ = process_vision_info(messages, image_patch_size=16)

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )
    inputs = {
        name: value.to(device)
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    # 去掉输入 prompt，只保留模型新生成的 token。
    new_token_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return prediction.strip()


def summarize(rows: list[dict]) -> dict:
    by_source = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in rows:
        source = row.get("human_or_machine", "unknown")
        by_source[source]["total"] += 1
        by_source[source]["correct"] += row["relaxed_correct"]

    total = len(rows)
    correct = sum(row["relaxed_correct"] for row in rows)

    return {
        "evaluated_samples": total,
        "relaxed_correct": correct,
        "relaxed_accuracy": correct / total if total else 0.0,
        "exact_correct": sum(
            row["match_type"] == "exact"
            for row in rows
        ),
        "numeric_relaxed_correct": sum(
            row["match_type"] == "numeric_relaxed"
            for row in rows
        ),
        "by_human_or_machine": {
            source: {
                **values,
                "accuracy": (
                    values["correct"] / values["total"]
                    if values["total"] else 0.0
                ),
            }
            for source, values in by_source.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate Qwen3-VL answers on full ChartQA val."
    )
    parser.add_argument("--data", default="data/chartqa_full/val.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means evaluate every sample; positive value limits total samples.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = read_jsonl(resolve_path(args.data))
    if args.limit > 0:
        data = data[:args.limit]

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 支持中断后续跑：已有 id 不再重复生成。
    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    completed_ids = {row["id"] for row in completed_rows}

    print("=== ChartQA generation evaluation ===")
    print(f"Samples requested: {len(data)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Adapter: {args.adapter or 'none (base model)'}")

    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = build_model(args.adapter, args.device)

    with output_path.open("a", encoding="utf-8") as output_file:
        for index, sample in enumerate(data, start=1):
            if sample["id"] in completed_ids:
                continue

            prediction = generate_answer(
                model,
                processor,
                sample,
                args.device,
                args.image_height,
                args.image_width,
                args.max_new_tokens,
            )

            answers = sample.get("all_answers") or [sample["response"]]
            correct, match_type = relaxed_match(prediction, answers)

            record = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "all_answers": answers,
                "prediction": prediction,
                "relaxed_correct": correct,
                "match_type": match_type,
                "human_or_machine": sample.get("human_or_machine", "unknown"),
            }
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            if index % 20 == 0:
                print(f"Generated: {index}/{len(data)}")

    result_rows = read_jsonl(output_path)
    result_ids = {sample["id"] for sample in data}
    result_rows = [row for row in result_rows if row["id"] in result_ids]

    summary = summarize(result_rows)
    summary["adapter"] = args.adapter or "base_model"
    summary["max_new_tokens"] = args.max_new_tokens
    summary["image_size"] = [args.image_height, args.image_width]

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Evaluation summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions saved: {output_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()