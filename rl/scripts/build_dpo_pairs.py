# 作用：从 SFT 候选记录中筛选不匹配标准答案的回答，构造 DPO 偏好对。
# 输入：generate_sft_candidates.py 生成的候选 JSONL。
# 产物：每道题最多一条 chosen / rejected 的 DPO JSONL；不加载模型、不使用 GPU。

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_text):
    """把相对项目根目录的路径转换为绝对路径。"""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path):
    """逐行读取 JSONL，忽略空行。"""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_text(value):
    """用于检查空回答和重复回答的简化文本规范化。"""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,!?:;，。！？：；")


def main():
    parser = argparse.ArgumentParser(
        description="从 SFT 候选记录构造 ChartQA DPO 偏好对。"
    )
    parser.add_argument(
        "--input",
        default="rl/data/sft_candidates_train.jsonl",
    )
    parser.add_argument(
        "--output",
        default="rl/data/chartqa_dpo_pairs_train.jsonl",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="最多写入多少条偏好对；0 表示不限制。",
    )
    args = parser.parse_args()

    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到候选记录：{input_path}")
    if args.max_pairs < 0:
        raise ValueError("--max-pairs 必须大于等于 0。")

    candidate_rows = read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_pairs = read_jsonl(output_path) if output_path.exists() else []
    completed_source_ids = {
        row["source_id"]
        for row in existing_pairs
    }

    selected_source_ids = set(completed_source_ids)
    pair_count = 0
    skipped_correct = 0
    skipped_empty = 0
    skipped_repeat_question = 0

    with output_path.open("a", encoding="utf-8") as output_file:
        for row in candidate_rows:
            source_id = row["source_id"]
            candidate = row["candidate"]

            # 正确回答不是 rejected；空回答也不作为第一版 DPO 数据。
            if row["relaxed_correct"]:
                skipped_correct += 1
                continue

            if not normalize_text(candidate):
                skipped_empty += 1
                continue

            # 每道题只取第一个合格错误候选。
            if source_id in selected_source_ids:
                skipped_repeat_question += 1
                continue

            pair = {
                "source_id": source_id,
                "image": row["image"],
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": candidate,
                "all_answers": row["all_answers"],
                "chosen_source": row["chosen_source"],
                "rejected_source": "sft_sampled_nonmatch",
                "candidate_sample_index": row["sample_index"],
                "generation_seed": row["generation_seed"],
                "temperature": row["temperature"],
                "top_p": row["top_p"],
                "human_or_machine": row["human_or_machine"],
            }
            output_file.write(
                json.dumps(pair, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            selected_source_ids.add(source_id)
            pair_count += 1

            if args.max_pairs > 0 and pair_count >= args.max_pairs:
                break

    print("=== DPO 偏好对构造摘要 ===")
    print(f"候选记录数：{len(candidate_rows)}")
    print(f"已有偏好对：{len(existing_pairs)}")
    print(f"本次新增偏好对：{pair_count}")
    print(f"跳过正确候选：{skipped_correct}")
    print(f"跳过空候选：{skipped_empty}")
    print(f"跳过同题额外错误候选：{skipped_repeat_question}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()