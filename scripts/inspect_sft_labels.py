import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def build_inputs(processor, messages, add_generation_prompt):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SFT labels for one Qwen3-VL sample."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()

    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    full_messages = user_messages + [
        {
            "role": "assistant",
            "content": args.answer,
        }
    ]

    # Prompt ends just after "<|im_start|>assistant\n".
    # It is the boundary before the answer starts.
    prompt_inputs = build_inputs(
        processor,
        user_messages,
        add_generation_prompt=True,
    )

    # Full conversation includes the reference answer.
    full_inputs = build_inputs(
        processor,
        full_messages,
        add_generation_prompt=False,
    )

    prefix_length = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]

    if not torch.equal(
        input_ids[:, :prefix_length],
        prompt_inputs["input_ids"],
    ):
        raise RuntimeError("Prompt is not a prefix of the full conversation.")

    labels = input_ids.clone()

    # Ignore image tokens, user prompt, and assistant role prefix.
    labels[:, :prefix_length] = -100

    # Ignore padding if batching adds it later.
    labels[full_inputs["attention_mask"] == 0] = -100

    target_ids = labels[0][labels[0] != -100].tolist()

    print("\n=== Token counts ===")
    print(f"Full input length: {input_ids.shape[1]}")
    print(f"Ignored prompt length: {prefix_length}")
    print(f"Supervised answer length: {len(target_ids)}")

    print("\n=== Label rule ===")
    print("Prompt/image/user tokens: -100 (ignored by loss)")
    print("Assistant answer tokens: original token IDs (included in loss)")

    print("\n=== Assistant target tokens ===")
    print(processor.tokenizer.convert_ids_to_tokens(target_ids))

    print("\n=== Assistant target decoded ===")
    print(
        processor.tokenizer.decode(
            target_ids,
            skip_special_tokens=False,
        )
    )

    print("\n=== First 20 labels ===")
    print(labels[0, :20].tolist())

    print("\n=== Labels around answer boundary ===")
    start = max(0, prefix_length - 8)
    end = min(labels.shape[1], prefix_length + 32)
    print("input tokens:", processor.tokenizer.convert_ids_to_tokens(
        input_ids[0, start:end].tolist()
    ))
    print("labels:", labels[0, start:end].tolist())


if __name__ == "__main__":
    main()
