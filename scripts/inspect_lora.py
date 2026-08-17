import argparse

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def main():
    parser = argparse.ArgumentParser(
        description="Attach LoRA to Qwen3-VL and inspect trainable parameters."
    )
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")

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

    total, trainable = count_parameters(model)

    print("\n=== LoRA configuration ===")
    print("Target modules: q_proj, v_proj")
    print("Rank (r): 8")
    print("Alpha: 16")
    print("Dropout: 0.0")

    print("\n=== Parameter counts ===")
    print(f"Total parameters: {total:,}")
    print(f"Trainable LoRA parameters: {trainable:,}")
    print(f"Trainable percentage: {trainable / total * 100:.4f}%")

    print("\n=== PEFT summary ===")
    model.print_trainable_parameters()

    lora_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print("\n=== First trainable parameter names ===")
    for name in lora_parameter_names[:10]:
        print(name)

    lora_a = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_A" in name
    ]
    lora_b = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]

    print("\n=== Initial LoRA matrix norms ===")
    print(f"First LoRA A norm: {lora_a[0].float().norm().item():.6f}")
    print(f"First LoRA B norm: {lora_b[0].float().norm().item():.6f}")


if __name__ == "__main__":
    main()
