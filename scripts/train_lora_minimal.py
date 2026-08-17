import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_batch(sample: dict, processor, device: str) -> tuple[dict, torch.Tensor]:
    image_path = Path(sample["image"])
    if not image_path.is_absolute():
        image_path = (PROJECT_ROOT / image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 训练时固定缩放，避免原图产生上万个视觉 token 而显存溢出。
    user_messages = [
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
    full_messages = user_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["response"]}],
        }
    ]

    # prefix 只含“图片 + 问题 + assistant 开始标记”
    prefix_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_images, prefix_videos = process_vision_info(
        user_messages,
        image_patch_size=16,
    )
    prefix_inputs = processor(
        text=[prefix_text],
        images=prefix_images,
        videos=prefix_videos,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )

    # full 含标准答案：它用于模型前向计算。
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_images, full_videos = process_vision_info(
        full_messages,
        image_patch_size=16,
    )
    batch = processor(
        text=[full_text],
        images=full_images,
        videos=full_videos,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )
    batch = {name: value.to(device) for name, value in batch.items()}

    # -100 表示该位置不参与 loss；仅监督 assistant 的标准答案。
    prefix_length = prefix_inputs["input_ids"].shape[1]
    labels = batch["input_ids"].clone()
    labels[:, :prefix_length] = -100
    labels[batch["attention_mask"] == 0] = -100

    return batch, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo/train.jsonl")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/lora-food-demo")
    args = parser.parse_args()

    data_path = (PROJECT_ROOT / args.data).resolve()
    samples = load_jsonl(data_path)
    if not samples:
        raise ValueError(f"No samples found in: {data_path}")

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    # 关闭 KV cache、开启 checkpointing：训练时减少显存占用。
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # 此处特意设为 0，便于观察：第一步只有 LoRA B 会因梯度而改变。
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    torch.cuda.reset_peak_memory_stats(device)

    print(f"Samples: {len(samples)}")
    print(f"Training steps: {args.steps}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    losses = []
    for step in range(1, args.steps + 1):
        # 数据集只有一条样本时，会反复取这条样本：这是刻意的过拟合演示。
        sample = samples[(step - 1) % len(samples)]
        batch, labels = build_batch(sample, processor, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"step {step:>3}/{args.steps} | loss: {loss.item():.6f}")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存的是 LoRA adapter，而不是 4B 基座模型。
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    summary = {
        "data": str(data_path),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 2
        ),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n=== Training complete ===")
    print(f"First loss: {losses[0]:.6f}")
    print(f"Last loss:  {losses[-1]:.6f}")
    print(f"Adapter saved to: {output_dir}")
    print(f"Peak GPU memory: {summary['peak_gpu_memory_gb']} GB")


if __name__ == "__main__":
    main()