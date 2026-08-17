import argparse
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3-VL multimodal inputs without loading the model."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--question",
        default="请描述这张图片。",
    )
    args = parser.parse_args()

    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print("\n=== 1. Original messages ===")
    print(messages)

    print("\n=== 2. Chat-template prompt ===")
    print(prompt)

    print("\n=== 3. Processor outputs ===")
    for name, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{name}: {value}")

    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    print("image_grid_thw values:", inputs["image_grid_thw"].tolist())
    print("mm_token_type_ids values:", inputs["mm_token_type_ids"].unique().tolist())
    print("Valid attention positions:", inputs["attention_mask"].sum().item())

    print("\n=== 4. Token summary ===")
    print(f"Number of input IDs: {len(input_ids)}")
    print("First 30 tokens:")
    print(tokens[:30])

    special_tokens = [
        (index, token)
        for index, token in enumerate(tokens)
        if "vision" in token.lower() or "image" in token.lower()
    ]
    print("\nVision/image-related special tokens:")
    print(special_tokens[:20])

    print("\n=== 5. Decoded input preview ===")
    decoded = processor.tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )
    print(decoded[:1500])


if __name__ == "__main__":
    main()
