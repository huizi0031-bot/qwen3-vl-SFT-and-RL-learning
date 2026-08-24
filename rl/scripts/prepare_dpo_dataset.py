# 作用：按 TRL 1.10 的多模态 DPO 数据约定，转换偏好对并切分训练集、监控集。
# 输入：chartqa_dpo_pairs_train.jsonl，包含图像路径、问题、chosen、rejected。
# 产物：带 image、prompt、chosen、rejected 字段的 DPO JSONL；图片将在训练脚本中解码。

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value):
    """把相对路径转换为项目根目录下的绝对路径。"""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_jsonl(path):
    """读取 JSONL 的全部非空行。"""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records, overwrite):
    """安全地逐行写出 JSONL，默认避免覆盖已有文件。"""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在：{path}\n"
            "若确认要重建，请额外传入 --overwrite。"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_pair(row):
    """把一条构造好的偏好对转换为 TRL 1.10 的对话式偏好记录。"""
    required_fields = {"source_id", "image", "prompt", "chosen", "rejected"}
    missing_fields = required_fields - set(row)

    if missing_fields:
        raise ValueError(f"偏好对缺少字段：{sorted(missing_fields)}")

    image_path = resolve_path(row["image"])
    prompt = str(row["prompt"]).strip()
    chosen = str(row["chosen"]).strip()
    rejected = str(row["rejected"]).strip()

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    if not prompt or not chosen or not rejected:
        raise ValueError(f"偏好对含有空文本：{row['source_id']}")

    return {
        "source_id": str(row["source_id"]),
        # 写入绝对路径；下一阶段由 datasets.Image 解码为真正图片。
        "image": str(image_path),
        # image 与文本分开：TRL 的 VLM collator 会把 image 配到 prompt 上。
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": chosen,
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": rejected,
            }
        ],
        "human_or_machine": row.get("human_or_machine"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="rl/data/chartqa_dpo_pairs_train.jsonl",
        help="Stage 15 构造出的语义偏好对。",
    )
    parser.add_argument(
        "--train-output",
        default="rl/data/chartqa_dpo_train.jsonl",
    )
    parser.add_argument(
        "--eval-output",
        default="rl/data/chartqa_dpo_eval.jsonl",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="用于监控 DPO 指标的切分比例。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="固定切分的随机种子。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 条；0 表示全部。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖既有输出文件。",
    )
    args = parser.parse_args()

    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio 必须在 0 与 1 之间")

    input_path = resolve_path(args.input)
    train_output_path = resolve_path(args.train_output)
    eval_output_path = resolve_path(args.eval_output)

    raw_pairs = read_jsonl(input_path)

    if args.limit > 0:
        raw_pairs = raw_pairs[:args.limit]

    records = [convert_pair(row) for row in raw_pairs]

    source_ids = [record["source_id"] for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("输入中存在重复 source_id，不能让同一题跨集合出现。")

    if len(records) < 2:
        raise ValueError("至少需要 2 条偏好对才能切分训练集与监控集。")

    random.Random(args.seed).shuffle(records)

    eval_count = max(1, round(len(records) * args.eval_ratio))
    eval_records = records[:eval_count]
    train_records = records[eval_count:]

    write_jsonl(train_output_path, train_records, args.overwrite)
    write_jsonl(eval_output_path, eval_records, args.overwrite)

    print("=== Stage 16：TRL 1.10 DPO 数据准备摘要 ===")
    print(f"输入偏好对：{len(raw_pairs)}")
    print(f"训练集：{len(train_records)}")
    print(f"监控集：{len(eval_records)}")
    print(f"随机种子：{args.seed}")
    print(f"训练集输出：{train_output_path}")
    print(f"监控集输出：{eval_output_path}")


if __name__ == "__main__":
    main()