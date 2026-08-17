import argparse
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


def resolve_from_project(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/test/example.jpg")
    parser.add_argument("--question", default="请描述这张图片。")
    parser.add_argument("--adapter", default="outputs/lora-food-demo")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    image_path = resolve_from_project(args.image)
    adapter_path = resolve_from_project(args.adapter)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_path}\n"
            "Please run train_lora_minimal.py successfully first."
        )

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

    # 1. 加载原始、未微调的 4B 基座模型。
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    # 2. 将训练保存的 LoRA A、B 权重挂载到基座模型对应 q_proj、v_proj。
    model = PeftModel.from_pretrained(base_model, adapter_path).eval()

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
                {"type": "text", "text": args.question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(
        messages,
        image_patch_size=16,
    )
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
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

    # generate 的结果包含输入 prompt；只解码新生成的回答部分。
    new_token_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("=== LoRA inference ===")
    print(f"Image: {image_path}")
    print(f"Question: {args.question}")
    print(f"Adapter: {adapter_path}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()