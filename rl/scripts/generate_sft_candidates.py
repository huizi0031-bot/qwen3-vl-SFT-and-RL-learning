# 作用：用冻结的 Stage 12 SFT adapter 为 ChartQA train 生成多个候选答案。
# 输入：ChartQA train.jsonl、base model、SFT adapter 与采样参数。
# 产物：逐候选写入 JSONL；后续只筛选其中不匹配标准答案的记录，构造 DPO preference pairs。

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# 当前文件位于 rl/scripts/，向上两层就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 服务器本地已经下载好的基座模型，不访问网络。
MODEL_DIR = Path(
    "/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)


def resolve_project_path(path_text):
    """把相对项目根目录的路径转换为绝对路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path):
    """逐行读取 JSONL，忽略空行。"""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_text(value):
    """沿用 Stage 11/12 的文本规范化规则。"""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,!?:;，。！？：；")


def parse_number(value):
    """把纯数值答案转换为 float，供 relaxed accuracy 使用。"""
    value = normalize_text(value)
    value = value.replace(",", "").replace("$", "")

    if value.endswith("%"):
        value = value[:-1]

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return float(value)

    return None


def relaxed_match(prediction, answers):
    """
    判断候选答案是否匹配 ChartQA 标准答案。
    先文本精确匹配；数值答案允许相对误差不超过 5%。
    """
    normalized_prediction = normalize_text(prediction)

    for answer in answers:
        normalized_answer = normalize_text(answer)

        if normalized_prediction == normalized_answer:
            return True, "exact"

        prediction_number = parse_number(prediction)
        answer_number = parse_number(answer)

        if prediction_number is None or answer_number is None:
            continue

        if answer_number == 0:
            if prediction_number == 0:
                return True, "numeric_exact"
        elif abs(prediction_number - answer_number) / abs(answer_number) <= 0.05:
            return True, "numeric_relaxed"

    return False, "incorrect"


def load_sft_policy(adapter_path, device):
    """加载冻结的 base + Stage 12 SFT adapter，只用于生成候选答案。"""
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # 本阶段仅生成候选，不允许更新 adapter 参数。
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
                {
                    "type": "text",
                    "text": sample["prompt"],
                },
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


def generate_one_candidate(policy, processor, inputs, args, generation_seed):
    """对同一条输入采样一个候选答案，并只解码新生成部分。"""
    # 每个候选使用独立种子；中断后续跑时，同一题同一候选编号仍可复现。
    torch.manual_seed(generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation_seed)

    with torch.inference_mode():
        generated_ids = policy.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    # generate 返回“原 prompt + 新 token”，这里只保留新生成答案。
    prompt_length = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[:, prompt_length:]

    return processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="使用冻结的 SFT policy 为 ChartQA train 生成候选答案。"
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
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="处理前多少道题；0 表示完整 train。",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="每道题顺序采样多少个候选答案。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="采样随机性；越高越多样，越低越保守。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus sampling 的概率阈值。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit 必须大于等于 0。")
    if args.num_candidates < 1:
        raise ValueError("--num-candidates 必须至少为 1。")
    if args.temperature <= 0:
        raise ValueError("--temperature 必须大于 0。")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p 必须在 (0, 1] 内。")

    data_path = resolve_project_path(args.data)
    adapter_path = resolve_project_path(args.adapter)
    output_path = resolve_project_path(args.output)

    if not data_path.is_file():
        raise FileNotFoundError(f"找不到训练数据：{data_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"找不到 SFT adapter：{adapter_path}")

    data = read_jsonl(data_path)
    if not data:
        raise ValueError("训练数据为空。")

    required_fields = {"id", "image", "prompt", "response"}
    missing = required_fields - set(data[0])
    if missing:
        raise ValueError(f"训练数据缺少字段：{sorted(missing)}")

    if args.limit > 0:
        data = data[:args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 已生成的 “题目 id + 候选序号” 不重复写入，支持中断后继续。
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    completed_keys = {
        (row["source_id"], row["sample_index"])
        for row in existing_rows
    }

    print("=== Stage 14：SFT 候选答案生成 ===")
    print(f"题目数：{len(data)}")
    print(f"每题候选数：{args.num_candidates}")
    print(f"SFT adapter：{adapter_path}")
    print(f"temperature：{args.temperature}")
    print(f"top_p：{args.top_p}")
    print(f"输出文件：{output_path}")
    print(f"已有候选记录：{len(existing_rows)}")

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

    generated_count = 0
    skipped_count = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for sample_position, sample in enumerate(data):
            inputs = prepare_model_inputs(
                processor,
                sample,
                args.device,
                args.image_size,
            )
            answers = sample.get("all_answers") or [sample["response"]]

            for sample_index in range(args.num_candidates):
                key = (sample["id"], sample_index)
                if key in completed_keys:
                    skipped_count += 1
                    continue

                # 每题、每个候选都有稳定且不同的随机种子。
                generation_seed = (
                    args.seed
                    + sample_position * args.num_candidates
                    + sample_index
                )
                candidate = generate_one_candidate(
                    policy,
                    processor,
                    inputs,
                    args,
                    generation_seed,
                )
                is_correct, match_type = relaxed_match(candidate, answers)

                record = {
                    "source_id": sample["id"],
                    "image": sample["image"],
                    "prompt": sample["prompt"],
                    "chosen": sample["response"],
                    "all_answers": answers,
                    "chosen_source": "chartqa_response",
                    "candidate": candidate,
                    "relaxed_correct": is_correct,
                    "match_type": match_type,
                    "sample_index": sample_index,
                    "generation_seed": generation_seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "human_or_machine": sample.get("human_or_machine"),
                }
                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                output_file.flush()
                generated_count += 1

            if (sample_position + 1) % 10 == 0 or sample_position + 1 == len(data):
                print(
                    f"已处理题目：{sample_position + 1}/{len(data)}，"
                    f"本次新生成候选：{generated_count}"
                )

    # 只统计本次目标题目对应的候选，避免输出文件中其他运行的记录干扰。
    result_rows = read_jsonl(output_path)
    target_ids = {sample["id"] for sample in data}
    result_rows = [
        row
        for row in result_rows
        if row["source_id"] in target_ids
    ]

    wrong_rows = [
        row
        for row in result_rows
        if not row["relaxed_correct"] and row["candidate"]
    ]

    print("\n=== 候选生成摘要 ===")
    print(f"目标题目数：{len(data)}")
    print(f"候选记录数：{len(result_rows)}")
    print(f"本次新生成：{generated_count}")
    print(f"本次跳过已存在候选：{skipped_count}")
    print(f"非空且不匹配标准答案的候选：{len(wrong_rows)}")
    print(f"候选记录已保存到：{output_path}")
    print("\n下一步只从“非空且 relaxed_correct=false”的记录中筛选 rejected。")


if __name__ == "__main__":
    main()