import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def build_inputs(processor, messages, add_generation_prompt):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    image_inputs, video_inputs = process_vision_info(
    messages,
    image_patch_size=16,
)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate one SFT loss without parameter updates."
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

    device = "cuda"

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    user_messages = [
        {
            "role": "user",
            "content": [
               {
    "type": "image",
    "image": args.image,
    "min_pixels": 50176,
    "max_pixels": 50176,
},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    full_messages = user_messages + [
        {"role": "assistant", "content": args.answer}
    ]

    prompt_inputs = build_inputs(
        processor,
        user_messages,
        add_generation_prompt=True,
    )
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
    labels[:, :prefix_length] = -100
    labels[full_inputs["attention_mask"] == 0] = -100

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    # Loss calculation does not need generation KV cache.
    model.config.use_cache = False

    model_inputs = {
        name: value.to(device)
        for name, value in full_inputs.items()
    }
    labels = labels.to(device)

    torch.cuda.reset_peak_memory_stats()

    # No backward graph is created; no parameter will be changed.
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            labels=labels,
            use_cache=False,
        )

    print("\n=== SFT forward result ===")
    print(f"Supervised answer tokens: {(labels != -100).sum().item()}")
    print(f"Average SFT loss: {outputs.loss.item():.6f}")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(
        "Peak GPU memory during forward: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )
    print("Backward called: no")
    print("Parameters updated: no")


if __name__ == "__main__":
    main()
