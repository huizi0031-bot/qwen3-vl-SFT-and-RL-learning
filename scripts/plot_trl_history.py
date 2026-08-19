import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

# 服务器没有桌面环境，所以使用无界面后端生成 PNG。
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss and learning-rate curves from TRL trainer_state.json."
    )
    parser.add_argument(
        "--state",
        default="outputs/chartqa-full-trl-lora/trainer_state.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/chartqa-full-trl-lora/loss_curve.png",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Try opening the PNG in the current VS Code window.",
    )
    args = parser.parse_args()

    state_path = resolve_path(args.state)
    output_path = resolve_path(args.output)

    if not state_path.is_file():
        raise FileNotFoundError(f"trainer_state.json not found: {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state["log_history"]

    train_points = [
        (item["step"], item["loss"])
        for item in history
        if "loss" in item and "eval_loss" not in item
    ]
    eval_points = [
        (item["step"], item["eval_loss"])
        for item in history
        if "eval_loss" in item
    ]
    lr_points = [
        (item["step"], item["learning_rate"])
        for item in history
        if "learning_rate" in item
    ]

    if not train_points:
        raise RuntimeError("No training-loss records found in trainer_state.json.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (loss_axis, lr_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
    )

    train_steps, train_losses = zip(*train_points)
    loss_axis.plot(
        train_steps,
        train_losses,
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="train loss",
    )

    if eval_points:
        eval_steps, eval_losses = zip(*eval_points)
        loss_axis.plot(
            eval_steps,
            eval_losses,
            marker="o",
            markersize=7,
            linewidth=1.8,
            label="validation loss",
        )

    loss_axis.set_title("Stage 12-A: ChartQA TRL LoRA SFT")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    if lr_points:
        lr_steps, learning_rates = zip(*lr_points)
        lr_axis.plot(
            lr_steps,
            learning_rates,
            color="tab:green",
            linewidth=1.5,
        )

    lr_axis.set_xlabel("Optimizer update / global step")
    lr_axis.set_ylabel("Learning rate")
    lr_axis.grid(alpha=0.3)

    figure.savefig(output_path, dpi=180)
    print(f"Plot saved: {output_path}")

    # Remote SSH 下 plt.show() 不会在你电脑弹窗。
    # 如果 VS Code 的 code 命令可用，这里会尝试在当前窗口打开 PNG；
    # 不可用时，在 VS Code 文件栏点击该 PNG 即可。
    if args.open:
        code_command = shutil.which("code")
        if code_command:
            subprocess.Popen(
                [code_command, "--reuse-window", str(output_path)]
            )
            print("Requested VS Code to open the plot.")
        else:
            print("VS Code CLI not found; open the PNG from VS Code Explorer.")


if __name__ == "__main__":
    main()