# 作用：以 Stage 12 的 SFT LoRA 为初始 policy，使用 ChartQA 偏好对进行 TRL 1.10 的 VLM DPO 训练。
# 输入：已转换的 DPO train / eval JSONL、Qwen3-VL base model、Stage 12 SFT adapter。
# 产物：训练 checkpoint 与新的 DPO policy adapter；原始 SFT adapter 不会被覆盖。

import argparse
from pathlib import Path

import torch
from datasets import Image, load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
from trl import DPOConfig, DPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def resolve_path(path_value):
    """把相对路径解析为项目根目录下的绝对路径。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dpo_dataset(jsonl_path, limit):
    """读取 DPO JSONL，并把 image 路径按 datasets.Image 在取样时解码为图片。"""
    dataset = load_dataset(
        "json",
        data_files=str(jsonl_path),
        split="train",
    )

    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # JSONL 中是图片绝对路径；这里转为按需解码的 PIL 图片。
    return dataset.cast_column("image", Image(decode=True))


def load_policy(base_model_path, adapter_path):
    """加载冻结底座与可训练的 Stage 12 SFT adapter。"""
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    policy = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=True,
    )

    # DPO 训练使用梯度检查点时不能保留 KV cache。
    policy.config.use_cache = False
    return policy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        default="rl/data/chartqa_dpo_train.jsonl",
    )
    parser.add_argument(
        "--eval-data",
        default="rl/data/chartqa_dpo_eval.jsonl",
    )
    parser.add_argument(
        "--adapter",
        default="outputs/chartqa-full-trl-lora/final_adapter",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/chartqa-dpo-lora",
    )

    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，DPO 训练需要可用 GPU。")

    train_data_path = resolve_path(args.train_data)
    eval_data_path = resolve_path(args.eval_data)
    adapter_path = resolve_path(args.adapter)
    output_dir = resolve_path(args.output_dir)

    if not train_data_path.is_file():
        raise FileNotFoundError(f"训练集不存在：{train_data_path}")
    if not eval_data_path.is_file():
        raise FileNotFoundError(f"监控集不存在：{eval_data_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"SFT adapter 不存在：{adapter_path}")

    set_seed(args.seed)

    train_dataset = load_dpo_dataset(train_data_path, args.train_limit)
    eval_dataset = load_dpo_dataset(eval_data_path, args.eval_limit)

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    processor.tokenizer.padding_side = "left"

    policy = load_policy(MODEL_DIR, adapter_path)

    print("=== Stage 17：DPO 训练配置 ===")
    print(f"训练偏好对：{len(train_dataset)}")
    print(f"监控偏好对：{len(eval_dataset)}")
    print(f"初始 SFT adapter：{adapter_path}")
    print(f"输出目录：{output_dir}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print("DPOTrainer 将自动复制冻结的 ref adapter。")

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        beta=args.beta,

        # VLM 不截断，避免错误移除图像 token。
        max_length=None,
        precompute_ref_log_probs=False,

        bf16=True,
        gradient_checkpointing=True,

        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        report_to=[],
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    print(f"当前 adapters：{list(policy.peft_config)}")
    policy.print_trainable_parameters()

    trainer.train()

    # 保存 active 的 policy adapter；原 SFT adapter 保持不变。
    policy.set_adapter("default")
    final_adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_path))
    processor.save_pretrained(str(final_adapter_path))

    print("=== DPO 训练完成 ===")
    print(f"DPO policy adapter：{final_adapter_path}")


if __name__ == "__main__":
    main()