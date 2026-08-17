import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


class JsonlSFTDataset(Dataset):
    """只负责读取一条原始 SFT 样本，不做 token 化。"""

    def __init__(self, data_path: Path):
        with data_path.open(encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file if line.strip()]

        if not self.samples:
            raise ValueError(f"No samples found in: {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SFTCollator:
    """将一批原始样本转换为模型输入与 labels。"""

    def __init__(self, processor, device: str):
        self.processor = processor
        self.device = device

    def __call__(self, samples):
        prefix_texts, full_texts = [], []
        prefix_images, full_images = [], []

        for sample in samples:
            image_path = Path(sample["image"])
            if not image_path.is_absolute():
                image_path = (PROJECT_ROOT / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

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

            prefix_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            # 每条样本各有一张图；extend 后得到一个 batch 的图片列表。
            images, _ = process_vision_info(user_messages, image_patch_size=16)
            prefix_images.extend(images)

            images, _ = process_vision_info(full_messages, image_patch_size=16)
            full_images.extend(images)

        prefix_inputs = self.processor(
            text=prefix_texts,
            images=prefix_images,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        batch = self.processor(
            text=full_texts,
            images=full_images,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        batch = {name: value.to(self.device) for name, value in batch.items()}

        labels = batch["input_ids"].clone()

        # 每条样本的 prompt 长度不同，必须逐条掩码。
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1)
        for row, prefix_length in enumerate(prefix_lengths.tolist()):
            labels[row, :prefix_length] = -100

        # padding 也不参与 loss。
        labels[batch["attention_mask"] == 0] = -100
        return batch, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo/train.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/lora-food-batch-demo")
    args = parser.parse_args()

    data_path = (PROJECT_ROOT / args.data).resolve()
    dataset = JsonlSFTDataset(data_path)

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    collator = SFTCollator(processor, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    total_steps = len(dataloader) * args.epochs
    print(f"Samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Total optimizer steps: {total_steps}")

    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for batch, labels in dataloader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(
                f"epoch {epoch}/{args.epochs} | "
                f"step {global_step}/{total_steps} | "
                f"batch samples: {labels.shape[0]} | "
                f"loss: {loss.item():.6f}"
            )

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    summary = {
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer_steps": total_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 2
        ),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n=== Batched training complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()