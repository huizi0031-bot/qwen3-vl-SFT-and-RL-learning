import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

# 服务器没有图形桌面；Agg 后端只保存 PNG，不尝试弹出窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_metric(metrics_path: Path, record: dict) -> None:
    """将一条训练或验证指标以 JSONL 形式追加保存。"""
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_metrics(metrics_path: Path, output_path: Path) -> None:
    """
    从 metrics.jsonl 读取已保存的指标，生成 loss_curve.png。

    蓝线：若干次 LoRA 更新的平均 train loss。
    橙线：每个 epoch 结束后完整验证集上的 val loss。
    """
    if not metrics_path.exists():
        return

    with metrics_path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    train_rows = [row for row in rows if row["kind"] == "train_update"]
    epoch_rows = [row for row in rows if row["kind"] == "epoch"]

    if not train_rows:
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        [row["global_step"] for row in train_rows],
        [row["loss"] for row in train_rows],
        label="train loss",
        color="tab:blue",
    )

    if epoch_rows:
        axis.plot(
            [row["global_step"] for row in epoch_rows],
            [row["val_loss"] for row in epoch_rows],
            marker="o",
            label="validation loss",
            color="tab:orange",
        )

    axis.set_title("ChartQA LoRA SFT loss curve")
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Loss curve saved: {output_path}")


class JsonlSFTDataset(Dataset):
    """只读取原始 JSONL 样本；token 化由 collator 在组成 batch 时完成。"""

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
    """
    JSONL 样本 → Qwen3-VL batch。

    labels 中：
    - 图片、用户问题、assistant 起始标记：-100，不计算 loss；
    - 标准答案 token：真实 token id，计算 SFT loss。
    """

    def __init__(self, processor, image_height: int, image_width: int):
        self.processor = processor
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, samples):
        prefix_texts = []
        full_texts = []
        images_for_batch = []

        for sample in samples:
            image_path = resolve_path(sample["image"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path),
                            "resized_height": self.image_height,
                            "resized_width": self.image_width,
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

            # prefix 只到 assistant 的回答起点，用于确定 label 掩码长度。
            prefix_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            # full 包含标准答案，用于构造模型输入与 labels。
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            images, _ = process_vision_info(
                user_messages,
                image_patch_size=16,
            )
            images_for_batch.extend(images)

        prefix_inputs = self.processor(
            text=prefix_texts,
            images=images_for_batch,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        batch = self.processor(
            text=full_texts,
            images=images_for_batch,
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )

        labels = batch["input_ids"].clone()
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1)

        for row, prefix_length in enumerate(prefix_lengths.tolist()):
            labels[row, :prefix_length] = -100

        # batch 补齐产生的 padding token 也不能进入 loss。
        labels[batch["attention_mask"] == 0] = -100
        return batch, labels


def move_to_device(batch, labels, device):
    batch = {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }
    return batch, labels.to(device, non_blocking=True)


def make_train_loader(
    dataset,
    collator,
    batch_size,
    epoch,
    seed,
    start_batch,
):
    """
    每个 epoch 按 seed + epoch 生成固定随机顺序。

    checkpoint 恢复时重新产生同一随机顺序，再直接跳过已经完成的
    batch，因此剩余训练样本顺序不变。
    """
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)

    indices = torch.randperm(len(dataset), generator=generator).tolist()
    start_sample = start_batch * batch_size
    remaining_dataset = Subset(dataset, indices[start_sample:])

    return DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )


def make_val_loader(dataset, collator, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )


def evaluate_loss(model, dataloader, device):
    """
    验证阶段只 forward，不执行 backward 或 optimizer.step。
    因此 val loss 不会改变任何 LoRA 参数。
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        for batch, labels in dataloader:
            batch, labels = move_to_device(batch, labels, device)
            loss = model(**batch, labels=labels).loss
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model,
    optimizer,
    output_dir,
    global_step,
    next_epoch,
    next_batch,
    epoch_loss_sum,
    epoch_loss_count,
):
    """
    保存可恢复训练所需的三部分：
    1. LoRA adapter；
    2. AdamW 的优化器状态；
    3. 下一次该从哪个 epoch / batch 继续。
    """
    checkpoint_dir = output_dir / "checkpoints" / f"step-{global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # PEFT 只保存 LoRA adapter，不保存 4B 基座模型。
    model.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    state = {
        "global_step": global_step,
        "next_epoch": next_epoch,
        "next_batch": next_batch,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_loss_count": epoch_loss_count,
    }
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint saved: {checkpoint_dir}")


def load_model(args, device):
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    if args.resume_from:
        print(f"Resuming adapter from: {args.resume_from}")
        model = PeftModel.from_pretrained(
            base_model,
            resolve_path(args.resume_from),
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            base_model,
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
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Full ChartQA LoRA SFT with checkpointing and loss plots."
    )

    parser.add_argument("--train-data", default="data/chartqa_full/train.jsonl")
    parser.add_argument("--val-data", default="data/chartqa_full/val.jsonl")
    parser.add_argument("--output-dir", default="outputs/chartqa-full-lora")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--image-width", type=int, default=448)

    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="0 means no limit; 1 is useful for a real memory preflight.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to outputs/.../checkpoints/step-XXXXXX.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")

    set_seed(args.seed)

    device = args.device
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_dataset = JsonlSFTDataset(resolve_path(args.train_data))
    val_dataset = JsonlSFTDataset(resolve_path(args.val_data))

    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    collator = SFTCollator(
        processor,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    model = load_model(args, device)
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    start_epoch = 1
    start_batch = 0
    global_step = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0

    if args.resume_from:
        resume_dir = resolve_path(args.resume_from)
        state = json.loads(
            (resume_dir / "training_state.json").read_text(encoding="utf-8")
        )
        optimizer.load_state_dict(
            torch.load(resume_dir / "optimizer.pt", map_location=device)
        )

        start_epoch = state["next_epoch"]
        start_batch = state["next_batch"]
        global_step = state["global_step"]

        # 若 checkpoint 正好位于 epoch 结尾，下个 epoch 应重新计 loss。
        if start_batch > 0:
            resume_epoch_loss_sum = state["epoch_loss_sum"]
            resume_epoch_loss_count = state["epoch_loss_count"]

    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )

    print("=== Full ChartQA LoRA SFT configuration ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Image size: {args.image_height} x {args.image_width}")
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(
        "Effective batch size: "
        f"{args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"Optimizer steps / epoch: {optimizer_steps_per_epoch}")
    print(f"Checkpoint interval: {args.checkpoint_steps} updates")

    val_loader = make_val_loader(val_dataset, collator, args.batch_size)

    torch.cuda.reset_peak_memory_stats(device)
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0

        train_loader = make_train_loader(
            train_dataset,
            collator,
            args.batch_size,
            epoch,
            args.seed,
            epoch_start_batch,
        )

        epoch_loss_sum = (
            resume_epoch_loss_sum if epoch == start_epoch else 0.0
        )
        epoch_loss_count = (
            resume_epoch_loss_count if epoch == start_epoch else 0
        )

        update_loss_sum = 0.0
        update_loss_count = 0

        for local_batch_index, (batch, labels) in enumerate(train_loader):
            batch_index = epoch_start_batch + local_batch_index
            batch, labels = move_to_device(batch, labels, device)

            raw_loss = model(**batch, labels=labels).loss

            # 除以累积步数，保证累计梯度的尺度与有效大 batch 一致。
            (raw_loss / args.gradient_accumulation_steps).backward()

            epoch_loss_sum += raw_loss.item()
            epoch_loss_count += 1
            update_loss_sum += raw_loss.item()
            update_loss_count += 1

            is_last_batch = batch_index + 1 == batches_per_epoch
            should_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or is_last_batch
            )
            if not should_update:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            is_requested_last_step = (
                args.max_optimizer_steps > 0
                and global_step >= args.max_optimizer_steps
            )
            should_log = (
                global_step % args.logging_steps == 0
                or is_last_batch
                or is_requested_last_step
            )

            if should_log:
                mean_update_loss = update_loss_sum / update_loss_count

                append_metric(
                    metrics_path,
                    {
                        "kind": "train_update",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": mean_update_loss,
                        "grad_norm": grad_norm.item(),
                    },
                )

                print(
                    f"epoch {epoch}/{args.epochs} | "
                    f"update {global_step} | "
                    f"train loss: {mean_update_loss:.6f} | "
                    f"grad norm: {grad_norm.item():.4f}"
                )
                update_loss_sum = 0.0
                update_loss_count = 0

            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch == batches_per_epoch:
                next_epoch = epoch + 1
                next_batch = 0

            should_save = (
                global_step % args.checkpoint_steps == 0
                or is_last_batch
                or is_requested_last_step
            )
            if should_save:
                save_checkpoint(
                    model,
                    optimizer,
                    output_dir,
                    global_step,
                    next_epoch,
                    next_batch,
                    epoch_loss_sum,
                    epoch_loss_count,
                )

            if is_requested_last_step:
                stopped_early = True
                break

        if stopped_early:
            break

        train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        val_loss = evaluate_loss(model, val_loader, device)

        epoch_metrics = {
            "kind": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        append_metric(metrics_path, epoch_metrics)

        print("\n=== Epoch summary ===")
        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))

        # 下一个 epoch 不继承上一个 epoch 的累计 loss。
        resume_epoch_loss_sum = 0.0
        resume_epoch_loss_count = 0
        start_batch = 0

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    # 最终只保存 LoRA adapter 与 processor，不复制 4B 基座模型。
    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    loss_plot_path = output_dir / "loss_curve.png"
    plot_metrics(metrics_path, loss_plot_path)

    summary = {
        "completed": not stopped_early,
        "global_optimizer_steps": global_step,
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated(device) / 1024**3,
            2,
        ),
        "final_adapter": str(final_adapter_dir),
        "loss_plot": str(loss_plot_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Training run finished ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()