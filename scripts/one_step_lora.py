import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
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
        padding=True,
        do_resize=False,
        return_tensors="pt",
    )


def grad_norm(parameter):
    if parameter.grad is None:
        return 0.0
    return parameter.grad.detach().float().norm().item()


def main():
    parser = argparse.ArgumentParser(
        description="One LoRA SFT training step for Qwen3-VL."
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
                    "resized_height": 288,
                    "resized_width": 160,
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

    print("Input sequence length:", input_ids.shape[1])
    print("Supervised answer tokens:", (labels != -100).sum().item())

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    # Lower activation memory; base parameters are still frozen by PEFT.
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )

    lora_a = next(
        parameter
        for name, parameter in model.named_parameters()
        if "lora_A" in name
    )
    lora_b = next(
        parameter
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )

    model_inputs = {
        name: value.to(device)
        for name, value in full_inputs.items()
    }
    labels = labels.to(device)

    print(f"LoRA A norm before: {lora_a.detach().float().norm().item():.6f}")
    print(f"LoRA B norm before: {lora_b.detach().float().norm().item():.6f}")

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    outputs = model(
        **model_inputs,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss

    print(f"Loss before update: {loss.item():.6f}")

    loss.backward()

    print(f"LoRA A gradient norm: {grad_norm(lora_a):.6f}")
    print(f"LoRA B gradient norm: {grad_norm(lora_b):.6f}")

    optimizer.step()

    print(f"LoRA A norm after: {lora_a.detach().float().norm().item():.6f}")
    print(f"LoRA B norm after: {lora_b.detach().float().norm().item():.6f}")
    print(
        "Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )
    print("Optimizer steps completed: 1")


if __name__ == "__main__":
    main()
