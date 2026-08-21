# 作用：读取 ChartQA train，并为后续 SFT 多候选生成准备输入。
# 输入：ChartQA JSONL、Stage 12 SFT adapter 路径、采样参数。
# 产物：后续会写出候选答案 JSONL；当前版本只检查数据与参数，不运行模型。

import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 当前文件位于 rl/scripts/，向上两层就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 服务器已经下载好的 Qwen3-VL 基座，不访问网络。
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def load_sft_policy(adapter_path, device):
    """加载冻结的 base + SFT adapter，仅用于生成候选答案。"""
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # is_trainable=False：本阶段只生成候选，不更新 adapter。
    policy = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
    return policy.to(device).eval(), processor

def prepare_model_inputs(processor, sample, device, image_size):
    """把一条 ChartQA 图文样本转换为 Qwen3-VL 的模型输入张量。"""
    image_path = resolve_project_path(sample["image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到图片：{image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "resized_height": image_size,
                    "resized_width": image_size,
                },
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, _ = process_vision_info(messages, image_patch_size=16)

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )

    return {
        name: value.to(device)
        for name, value in inputs.items()
    }

def resolve_project_path(path_text):
    """把相对项目根目录的路径转换为绝对路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path):
    """逐行读取 JSONL，忽略空行。"""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="为 ChartQA 训练集准备 SFT 候选答案生成。"
    )
    parser.add_argument("--data", default="data/chartqa_full/train.jsonl")
    parser.add_argument(
        "--adapter",
        default="outputs/chartqa-full-trl-lora/final_adapter",
    )
    parser.add_argument(
        "--output",
        default="rl/data/sft_candidates_train.jsonl",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    data_path = resolve_project_path(args.data)
    adapter_path = resolve_project_path(args.adapter)
    output_path = resolve_project_path(args.output)

    if not data_path.is_file():
        raise FileNotFoundError(f"找不到训练数据：{data_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"找不到 SFT adapter：{adapter_path}")

    data = read_jsonl(data_path)
    if args.limit > 0:
        data = data[:args.limit]

    required_fields = {"id", "image", "prompt", "response"}
    missing = required_fields - set(data[0])
    if missing:
        raise ValueError(f"训练数据缺少字段：{sorted(missing)}")

    print("=== Stage 14: 候选生成预检查 ===")
    print(f"样本数：{len(data)}")
    print(f"SFT adapter：{adapter_path}")
    print(f"候选数 / 题：{args.num_candidates}")
    print(f"temperature：{args.temperature}")
    print(f"未来输出：{output_path}")

    for sample in data[:3]:
        print(
        f"\nid={sample['id']}"
        f"\nprompt={sample['prompt']}"
        f"\nchosen={sample['response']}"
        )
    print("\n正在加载冻结的 SFT policy...")
    policy, processor = load_sft_policy(adapter_path, args.device)

    total_params = sum(parameter.numel() for parameter in policy.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )

    print(f"policy 类型：{type(policy).__name__}")
    print(f"总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")    

    inputs = prepare_model_inputs(
        processor,
        data[0],
        args.device,
        args.image_size,
    )

    print("\n第一条样本的模型输入：")
    for name, value in inputs.items():
        print(
            f"{name}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )
        # 先用贪心生成验证推理链路；下一步才开启随机采样。
    with torch.inference_mode():
        generated_ids = policy.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    # generated_ids 前半部分是原 prompt，后半部分才是新生成的答案。
    prompt_length = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[:, prompt_length:]

    prediction = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print("\n=== 贪心生成结果 ===")
    print(f"reference / chosen：{data[0]['response']}")
    print(f"SFT prediction：{prediction}")

if __name__ == "__main__":
    main()