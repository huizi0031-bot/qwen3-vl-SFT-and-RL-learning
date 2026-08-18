import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = PROJECT_ROOT / "experiments/chartqa_full_base_val.jsonl"
LORA_PATH = PROJECT_ROOT / "experiments/chartqa_full_lora_val.jsonl"

OUTPUT_JSON = PROJECT_ROOT / "experiments/chartqa_full_comparison.json"
OUTPUT_PNG = PROJECT_ROOT / "experiments/chartqa_full_comparison.png"


def read_result_map(path: Path) -> dict[str, dict]:
    """读取结果 JSONL，并拒绝重复 id，防止评估结果被重复计数。"""
    rows = {}

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            sample_id = row["id"]

            if sample_id in rows:
                raise ValueError(
                    f"Duplicate id in {path.name}: {sample_id} "
                    f"(line {line_number})"
                )

            rows[sample_id] = row

    return rows


def safe_accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def main():
    base_rows = read_result_map(BASE_PATH)
    lora_rows = read_result_map(LORA_PATH)

    base_ids = set(base_rows)
    lora_ids = set(lora_rows)

    if base_ids != lora_ids:
        missing_in_lora = sorted(base_ids - lora_ids)[:5]
        missing_in_base = sorted(lora_ids - base_ids)[:5]

        raise ValueError(
            "Base and LoRA did not evaluate exactly the same samples.\n"
            f"Missing in LoRA, first 5: {missing_in_lora}\n"
            f"Missing in base, first 5: {missing_in_base}"
        )

    improved = []
    regressed = []
    both_correct = []
    both_wrong = []

    by_source = defaultdict(
        lambda: {
            "total": 0,
            "base_correct": 0,
            "lora_correct": 0,
        }
    )

    for sample_id in sorted(base_ids):
        base = base_rows[sample_id]
        lora = lora_rows[sample_id]

        base_correct = bool(base["relaxed_correct"])
        lora_correct = bool(lora["relaxed_correct"])

        # 数据集当前保存的是 0 / 1 编码；不擅自解释为 human 或 machine。
        source = str(base.get("human_or_machine", "unknown"))
        lora_source = str(lora.get("human_or_machine", "unknown"))

        if source != lora_source:
            raise ValueError(
                f"Source mismatch for {sample_id}: "
                f"base={source}, lora={lora_source}"
            )

        by_source[source]["total"] += 1
        by_source[source]["base_correct"] += int(base_correct)
        by_source[source]["lora_correct"] += int(lora_correct)

        record = {
            "id": sample_id,
            "question": base["prompt"],
            "answers": base["all_answers"],
            "base_prediction": base["prediction"],
            "lora_prediction": lora["prediction"],
            "base_match_type": base["match_type"],
            "lora_match_type": lora["match_type"],
            "subset": source,
        }

        if not base_correct and lora_correct:
            improved.append(record)
        elif base_correct and not lora_correct:
            regressed.append(record)
        elif base_correct and lora_correct:
            both_correct.append(record)
        else:
            both_wrong.append(record)

    total = len(base_ids)

    # 基座答对：两者都对 + 基座对但 LoRA 退化。
    base_correct_count = len(both_correct) + len(regressed)

    # LoRA 答对：两者都对 + 基座错但 LoRA 改正。
    lora_correct_count = len(both_correct) + len(improved)

    base_accuracy = safe_accuracy(base_correct_count, total)
    lora_accuracy = safe_accuracy(lora_correct_count, total)

    source_summary = {}
    for source, values in sorted(by_source.items()):
        source_summary[source] = {
            **values,
            "base_accuracy": safe_accuracy(
                values["base_correct"],
                values["total"],
            ),
            "lora_accuracy": safe_accuracy(
                values["lora_correct"],
                values["total"],
            ),
        }

    summary = {
        "total_samples": total,
        "base_correct": base_correct_count,
        "lora_correct": lora_correct_count,
        "base_relaxed_accuracy": base_accuracy,
        "lora_relaxed_accuracy": lora_accuracy,
        "accuracy_change_percentage_points": round(
            (lora_accuracy - base_accuracy) * 100,
            2,
        ),
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "both_correct_count": len(both_correct),
        "both_wrong_count": len(both_wrong),
        "by_subset": source_summary,
        "improved_examples": improved[:20],
        "regressed_examples": regressed[:20],
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 绘制 overall、subset 0、subset 1 的基座/LoRA 对比。
    sources = list(source_summary.keys())
    labels = ["overall"] + [f"subset {source}" for source in sources]

    base_values = [base_accuracy]
    lora_values = [lora_accuracy]

    for source in sources:
        base_values.append(source_summary[source]["base_accuracy"])
        lora_values.append(source_summary[source]["lora_accuracy"])

    positions = list(range(len(labels)))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    base_bars = axis.bar(
        [position - width / 2 for position in positions],
        base_values,
        width,
        label="base model",
        color="tab:blue",
    )
    lora_bars = axis.bar(
        [position + width / 2 for position in positions],
        lora_values,
        width,
        label="base + LoRA",
        color="tab:orange",
    )

    axis.set_title("ChartQA validation relaxed accuracy")
    axis.set_xlabel("Validation subset")
    axis.set_ylabel("Relaxed accuracy")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    axis.bar_label(
        base_bars,
        labels=[f"{value:.1%}" for value in base_values],
        padding=3,
        fontsize=9,
    )
    axis.bar_label(
        lora_bars,
        labels=[f"{value:.1%}" for value in lora_values],
        padding=3,
        fontsize=9,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=160)
    plt.close(figure)

    print("=== Base vs LoRA comparison ===")
    print(f"Samples: {total}")
    print(f"Base relaxed accuracy: {base_accuracy:.2%}")
    print(f"LoRA relaxed accuracy: {lora_accuracy:.2%}")
    print(
        "Accuracy change: "
        f"{summary['accuracy_change_percentage_points']:+.2f} percentage points"
    )
    print(f"Improved: {len(improved)}")
    print(f"Regressed: {len(regressed)}")
    print(f"Both correct: {len(both_correct)}")
    print(f"Both wrong: {len(both_wrong)}")
    print(f"Saved summary: {OUTPUT_JSON}")
    print(f"Saved chart: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()