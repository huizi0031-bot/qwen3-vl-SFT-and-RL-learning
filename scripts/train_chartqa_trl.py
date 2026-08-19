import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Image, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


# 无论从哪个目录运行，都以项目根目录作为相对路径基准。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 使用服务器已经下载好的本地基座模型，不访问 Hugging Face 网络。
DEFAULT_MODEL_PATH = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def resolve_project_path(path_text: str) -> Path:
    """把相对项目根目录的路径转成绝对路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_trl_dataset(jsonl_path: Path):
    """
    把已有 ChartQA JSONL 转为 TRL 的 VLM 数据格式。

    原始字段：
        image, prompt, response

    交给 TRL 的字段：
        image: PIL 图片
        prompt: user 的图片 + 问题
        completion: assistant 的标准答案

    注意：这里只保存图片路径，并在 DataLoader 取样时解码图片，
    不会把 28,299 张图片全部预先读进内存。
    """
    dataset = load_dataset(
        "json",
        data_files=str(jsonl_path),
        split="train",
    )

    required = {"image", "prompt", "response"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{jsonl_path} is missing fields: {sorted(missing)}")

    def make_image_path_absolute(example):
        image_path = Path(example["image"])
        if not image_path.is_absolute():
            image_path = (PROJECT_ROOT / image_path).resolve()
        return {"image": str(image_path)}

    # JSONL 中的 image 原本是相对路径；先改为绝对路径，避免工作目录变化造成找图失败。
    dataset = dataset.map(
        make_image_path_absolute,
        desc=f"Resolving image paths: {jsonl_path.name}",
    )

    # datasets.Image 会在真正取到样本时，把路径解码为 PIL.Image。
    dataset = dataset.cast_column("image", Image(decode=True))

    def to_conversational_vlm_format(batch):
        """
        这是 TRL 需要的多模态 prompt-completion 格式。

        image 列单独存真实图片；
        prompt 里的 {'type': 'image'} 是图片占位符；
        completion 是标准答案。

        TRL 会调用 Qwen 的 chat template、processor，并自动生成 labels。
        """
        prompts = []
        completions = []

        for question, answer in zip(batch["prompt"], batch["response"]):
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            )
            completions.append(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    }
                ]
            )

        return {
            "image": batch["image"],
            "prompt": prompts,
            "completion": completions,
        }

    # with_transform 是“按需转换”：不额外写出一份巨大的处理后数据集。
    return dataset.with_transform(to_conversational_vlm_format)


def set_image_pixel_limit(processor, image_size: int) -> None:
    """
    限制视觉输入面积，控制显存。

    448 表示目标面积约为 448 × 448 像素。
    Qwen 的图像处理器仍会尽量保留长宽比，而不是强制拉成正方形。
    """
    if image_size <= 0:
        return

    pixels = image_size * image_size
    image_processor = processor.image_processor

    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = pixels

    print(f"Vision pixel budget: about {image_size} x {image_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 12-A: Full ChartQA LoRA SFT with TRL SFTTrainer."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", default="data/chartqa_full")
    parser.add_argument(
        "--output-dir",
        default="outputs/chartqa-full-trl-lora",
        help="New directory; do not overwrite the earlier manual-training experiment.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="-1 means full training; 1 is useful for a preflight check.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint in output-dir.",
    )
    args = parser.parse_args()

    model_path = resolve_project_path(args.model_path)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not train_jsonl.is_file() or not val_jsonl.is_file():
        raise FileNotFoundError(
            f"Expected train.jsonl and val.jsonl under: {data_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据：仍是之前准备好的完整 ChartQA，不重新下载。
    train_dataset = build_trl_dataset(train_jsonl)
    val_dataset = build_trl_dataset(val_jsonl)

    # 2. 加载本地模型与 processor。
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    set_image_pixel_limit(processor, args.image_size)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # 训练不需要生成时的 KV cache；关闭它可节省显存。
    model.config.use_cache = False

    # LoRA + gradient checkpointing 时，需要让输入 embedding 可接收梯度。
    model.enable_input_require_grads()

    # 3. 这与前面 Stage 4 的 LoRA 配置保持一致，方便公平比较。
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # 4. SFTConfig 替代此前手写的 optimizer、scheduler、epoch loop、
    #    gradient accumulation、log、checkpoint 与 validation loss。
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
       # 约为 3% × 3538 个 optimizer update。
        warmup_steps=106,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,

        # RTX 3090 上使用 BF16；gradient checkpointing 用计算换显存。
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,

        # VLM 不能随意截断 token，否则可能截掉图片 token。
        max_length=None,

        # 对应我们过去手写 labels[:prompt_length] = -100。
        completion_only_loss=True,

        # VLM 的 image/prompt/completion 需要保留给 TRL 的默认视觉 Collator。
        remove_unused_columns=False,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        # 每个 epoch 结束后，用 val.jsonl 计算 teacher-forcing eval_loss。
        eval_strategy="epoch",

        # 训练中断后可从 checkpoint 恢复；只保留最近两个。
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        # 先不引入 TensorBoard；稍后直接读取 trainer_state.json 画图。
        report_to="none",

        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    print("\n=== Stage 12-A: TRL full ChartQA configuration ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Micro batch size: {args.train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(
        "Estimated optimizer updates / epoch: "
        f"{math.ceil(math.ceil(len(train_dataset) / args.train_batch_size) / args.gradient_accumulation)}"
    )
    print(f"Output directory: {output_dir}")

    # 5. 不传自定义 collator。
    # TRL 识别到 image 列后，会自动选 DataCollatorForVisionLanguageModeling。
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # 最终目录只保存 LoRA adapter，而不是 4B 基座模型副本。
    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    processor.save_pretrained(final_adapter_dir)

    # 保存 TRL 的 log_history，画图脚本会读取它。
    trainer.save_state()

    summary = {
        "stage": "12-A",
        "trainer": "trl.SFTTrainer",
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "global_step": trainer.state.global_step,
        "final_adapter": str(final_adapter_dir),
        "trainer_state": str(output_dir / "trainer_state.json"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Training completed ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()