import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Offline Qwen3-VL image inference")
    parser.add_argument("--model-path", required=True, help="Local model snapshot path")
    parser.add_argument("--image", required=True, help="Absolute path to an image")
    parser.add_argument(
        "--question",
        default="请用一句话描述这张图片。",
        help="Question for the image",
    )
    args = parser.parse_args()

    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = "cuda"

    # Load only from the local model snapshot.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"\nQuestion: {args.question}")
    print(f"Answer: {output}")


if __name__ == "__main__":
    main()
