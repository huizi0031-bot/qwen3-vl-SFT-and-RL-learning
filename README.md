# Qwen3-VL SFT and RL Learning

一个以 **Qwen3-VL-4B** 为基础的视觉语言模型（VLM）学习仓库。

本项目的目标不是只跑通一次训练，而是系统理解一条完整的模型后训练路径：

```text
Base VLM
   ↓
Supervised Fine-Tuning (SFT)
   ↓
SFT checkpoint / adapter
   ↓
Preference & reward learning
   ↓
Alignment / RL (DPO, GRPO ...)
```

学习分为两个**相互独立的项目阶段**：先完成并沉淀 SFT，再开始 RL/对齐学习。当前仓库首先聚焦 Project A；设计上为 Project B 预留数据和模型接口，但不会把 RL 概念混入 SFT 的学习过程。

## Learning goals

完成本仓库后，能够：

- 理解 Qwen3-VL 的图像、文本和生成输出如何协同工作；
- 从原始图文问答数据构建可训练的多模态 SFT 数据；
- 理解 `input_ids`、`labels`、loss、反向传播和优化器在 SFT 中的作用；
- 在单张 GPU 上完成 LoRA SFT、保存 adapter 并进行评估；
- 在资源可用时，将单卡训练扩展为多卡 DDP 训练；
- 以可复现实验记录和清晰的 Git 历史沉淀每一个阶段；
- 在 SFT 结束后，以保存的 SFT checkpoint 为起点进入偏好学习和 RL。

## Project map

```text
Project A — Qwen3-VL SFT Learning
│
├── Stage 0  Environment and repository setup
├── Stage 1  Qwen3-VL inference and architecture
├── Stage 2  Multimodal data pipeline
├── Stage 3  SFT mechanism and loss masking
├── Stage 4  Single-GPU LoRA SFT
├── Stage 5  Evaluation, analysis and reproducibility
└── Stage 6  Engineering extensions (QLoRA / multi-GPU)

                  SFT model / LoRA adapter
                              │
                              ▼
Project B — Alignment and RL Learning (later)
├── Preference-data construction
├── DPO
├── Reward / verifiable reward design
└── GRPO / other RL methods
```

## Project A: SFT learning path

### Stage 0 — Environment and project setup

**Goal:** build a reproducible research environment and establish the project workflow.

- Learn the basic roles of Linux, Conda, CUDA, PyTorch and GPU monitoring.
- Record GPU, driver, CUDA, PyTorch and package versions.
- Set up the repository layout, `.gitignore`, dependency file and experiment-record format.

**Deliverables**

```text
notes/environment.md
requirements.txt (or environment.yml)
```

Suggested commit: `chore: initialize reproducible training environment`

---

### Stage 1 — Meet Qwen3-VL

**Goal:** understand the model before training it.

First run an image-question inference demo, then connect its behavior to the model pipeline:

```text
image → vision encoder → visual tokens ┐
                                       ├→ language model → output tokens
text  → tokenizer / processor ─────────┘
```

Focus on the roles of the vision encoder, language-model backbone, multimodal projector, processor and tokenizer. Training is intentionally out of scope at this stage.

**Deliverables**

```text
scripts/inference.py
notes/qwen3vl_architecture.md
```

Suggested commit: `feat: add qwen3-vl inference demo`

---

### Stage 2 — Understand the multimodal data pipeline

**Goal:** trace one ChartQA-style sample from raw data to model tensors.

```text
image + question + answer
          ↓
conversation messages / chat template
          ↓
processor
          ↓
input_ids + attention_mask + pixel_values + image_grid_thw
```

Do not train yet. Print shapes, decoded tokens and important intermediate outputs. The point is to understand how an image is represented alongside text—not to treat `processor(...)` as a black box.

**Deliverables**

```text
scripts/inspect_sample.py
notes/multimodal_input_pipeline.md
```

Suggested commit: `feat: inspect qwen3-vl multimodal processor pipeline`

---

### Stage 3 — Understand the SFT objective

**Goal:** understand exactly what the model learns from an answer.

For a sample formatted as `User: <image + question> / Assistant: <answer>`, inspect:

```text
input_ids:  image tokens + prompt tokens + answer tokens
labels:     -100         + -100          + answer token ids
```

Only answer tokens contribute to the usual causal-language-model loss; tokens marked `-100` are ignored. Manually run a small forward pass, inspect the loss, and connect it to backward propagation, gradients and optimizer updates.

**Deliverables**

```text
scripts/inspect_sft_labels.py
scripts/one_step_train.py
notes/sft_loss_and_label_masking.md
```

Suggested commit: `feat: document sft labels and one-step training`

---

### Stage 4 — Run single-GPU LoRA SFT

**Goal:** complete the full SFT loop on one GPU.

The first run uses **bf16 LoRA**, not QLoRA. This keeps the learning focus on data, loss, trainable parameters, saving and evaluation; quantization is introduced later as an engineering optimization.

```text
prepare dataset
   ↓
load Qwen3-VL-4B
   ↓
attach LoRA adapters
   ↓
train / validate
   ↓
save adapter + configuration + metrics
   ↓
run inference comparison
```

Use a small, controlled subset first. Record the trainable-parameter count, VRAM use, batch size, learning rate, loss curve and example predictions.

**Deliverables**

```text
scripts/train_lora.py
scripts/evaluate.py
configs/sft_lora.yaml
experiments/<run-name>/metrics.md
notes/lora_sft.md
```

Suggested commit: `feat: train qwen3-vl with single-gpu lora sft`

---

### Stage 5 — Evaluate, analyze and reproduce

**Goal:** decide whether the model improved and explain why.

- Compare base-model and SFT-model outputs on a fixed evaluation set.
- Separate qualitative cases from quantitative metrics.
- Analyze incorrect answers: OCR, chart reading, reasoning, data noise, formatting or overfitting.
- Re-run a saved configuration to verify reproducibility.

**Deliverables**

```text
experiments/<run-name>/report.md
experiments/<run-name>/predictions.jsonl
notes/evaluation_and_error_analysis.md
```

Suggested commit: `docs: add sft evaluation and error analysis`

---

### Stage 6 — Engineering extensions (optional)

**Goal:** improve efficiency only after the baseline is understood.

This stage is not a prerequisite for SFT completion.

- **QLoRA:** learn NF4/4-bit quantization and compare memory, speed and quality with LoRA.
- **Multi-GPU:** learn `accelerate` / DDP when multiple GPUs are actually available; single-GPU capability remains the baseline.
- Explore gradient accumulation, sequence/image resolution, checkpointing and batch-size scaling.

**Deliverables**

```text
configs/sft_qlora.yaml
configs/sft_ddp.yaml
notes/qlora_and_distributed_training.md
```

Suggested commit: `feat: add qlora or distributed sft experiments`

## Bridge to Project B: leave interfaces, keep learning separate

Project A is finished when the SFT model is trained, saved and evaluated. Project B begins only then.

The SFT pipeline should avoid hard-coding a data schema that cannot grow. Keep a canonical prompt representation that can later extend from:

```json
{
  "image": "path/to/image.png",
  "prompt": "Read the value for 2024.",
  "response": "51"
}
```

to preference learning:

```json
{
  "image": "path/to/image.png",
  "prompt": "Read the value for 2024.",
  "chosen": "51",
  "rejected": "61"
}
```

This is an interface decision, not a requirement to create preference data now. At the SFT/RL boundary, preserve:

- the final LoRA adapter or merged SFT checkpoint;
- the exact base-model revision and processor configuration;
- prompt/chat-template conventions;
- dataset version, splits and evaluation set;
- training configuration, seed and metrics.

These artifacts let Project B start from a known SFT policy instead of rebuilding work.

## Repository conventions

```text
.
├── configs/        # Versioned training configurations
├── data/           # Data instructions and lightweight manifests (not raw large data)
├── scripts/         # Small, focused runnable scripts
├── notes/           # Concepts, observations and troubleshooting notes
├── experiments/     # Per-run configs, metrics, predictions and reports
├── outputs/         # Ignored model checkpoints and generated artifacts
├── requirements.txt
└── README.md
```

Do not commit model weights, raw datasets, access tokens or large generated files. Commit code, configurations, data manifests, notes and concise experiment evidence.

## GitHub learning loop

Every completed stage follows the same loop:

```text
learn one concept → run one focused experiment → record evidence → update notes → commit
```

Each commit should be small, runnable where applicable, and explain one meaningful change. This repository is both a codebase and a learning portfolio: documentation is a first-class deliverable, not an afterthought.

## Status

- [ ] Stage 0: Environment and project setup
- [ ] Stage 1: Qwen3-VL inference and architecture
- [ ] Stage 2: Multimodal data pipeline
- [ ] Stage 3: SFT objective and label masking
- [ ] Stage 4: Single-GPU LoRA SFT
- [ ] Stage 5: Evaluation and reproducibility
- [ ] Stage 6: QLoRA / multi-GPU extensions
- [ ] Project B: Alignment and RL learning

## Guiding principle

> First understand and complete SFT on one GPU. Then optimize it. Only after that, use the resulting SFT model as the starting point for RL and alignment learning.
