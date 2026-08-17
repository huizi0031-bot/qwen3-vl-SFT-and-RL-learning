import argparse
import json
import re
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


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：]+", "", text).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo/eval.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default="experiments/eval_lora_food_batch_demo.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    with resolve_path(args.data).open(encoding="utf-8") as file:
        samples = [json.loads(line) for line in file if line.strip()]

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model = base_model

    if args.adapter:
      model = PeftModel.from_pretrained(
        base_model,
        resolve_path(args.adapter),
    )

    model.eval()

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches = 0
    with output_path.open("w", encoding="utf-8") as result_file:
        for sample in samples:
            image_path = resolve_path(sample["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path),
                            "resized_height": 288,
                            "resized_width": 160,
                        },
                        {"type": "text", "text": sample["prompt"]},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision_info(messages, image_patch_size=16)
            inputs = processor(
                text=[prompt],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                do_resize=False,
            ).to(device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            answer_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            prediction = processor.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            match = normalize(prediction) == normalize(sample["response"])
            matches += match

            result = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "reference": sample["response"],
                "prediction": prediction,
                "normalized_exact_match": match,
            }
            result_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"\nQuestion:   {sample['prompt']}")
            print(f"Reference:  {sample['response']}")
            print(f"Prediction: {prediction}")
            print(f"String match: {match}")

    print(f"\nSaved: {output_path}")
    print(f"Exact-match rate: {matches}/{len(samples)}")


if __name__ == "__main__":
    main()