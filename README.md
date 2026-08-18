# Qwen3-VL SFT and RL Learning

这是一个以 **Qwen3-VL-4B-Instruct** 为基础的视觉语言模型（VLM）学习仓库。目标不是只训练出一个 adapter，而是亲手理解并复现完整链路：

```text
环境与模型加载
  ↓
图文推理
  ↓
多模态数据处理
  ↓
labels / loss / backward
  ↓
LoRA 参数更新
  ↓
最小训练闭环
  ↓
完整 ChartQA SFT 与生成式评估
  ↓
保存 SFT adapter（未来 RL 的起点）
```

仓库当前聚焦 **Project A：SFT**。Project B（偏好学习、DPO、GRPO/RL）会复用这里的 SFT adapter、数据格式和评估约定，但不会和当前 SFT 学习混在一起。

---

## 当前状态

### 已完成的学习阶段

- [x] Stage 0：环境、Conda、CUDA、PyTorch、Git 项目骨架
- [x] Stage 1：本地 Qwen3-VL-4B 图文推理
- [x] Stage 2：多模态 `processor` 输入输出检查
- [x] Stage 3：SFT labels、token-level loss、teacher forcing
- [x] Stage 4：LoRA A/B 矩阵与一次真实参数更新
- [x] Stage 5：单样本最小 LoRA SFT、保存与 adapter 推理
- [x] Stage 6：JSONL、Dataset、DataLoader、batched SFT
- [x] Stage 7：小型验证集上的生成与比较
- [x] Stage 8：ChartQA Mini 数据准备与检查
- [x] Stage 9：ChartQA Mini LoRA 训练与评估
- [x] Stage 10：完整 ChartQA 单卡 LoRA SFT、checkpoint、loss 可视化
- [x] Stage 11：完整 ChartQA validation 生成式评估与配对比较
- [ ] Stage 12：ChartQA test 最终盲测
- [ ] Project B：偏好数据、DPO、GRPO/RL

### 当前完整 SFT 实验结果

| 项目 | 值 | 中文说明 |
|---|---:|---|
| 基座模型 | Qwen3-VL-4B-Instruct | 本地 Hugging Face 快照加载，不在线下载 |
| 训练集 | ChartQA train，28,299 条 | 唯一参与 `backward()` 和 `optimizer.step()` 的数据 |
| 验证集 | ChartQA val，1,920 条 | 不参与参数更新；用于 loss 与生成式质量评估 |
| 图像尺寸 | 448 × 448 | 为图表文字、坐标轴和图例保留细节，并已在 3090 上通过显存预检 |
| micro batch | 1 | 一次放进显存的样本数 |
| 梯度累积 | 8 | 积累 8 个样本梯度后才更新一次 LoRA |
| 有效 batch | 8 | `1 × 8` |
| epoch | 1 | 固定配置下的完整基线实验 |
| LoRA | `r=8`，`alpha=16`，`q_proj`/`v_proj` | 基座冻结，只训练 attention 内的低秩增量 |
| 基座 val relaxed accuracy | 2.19%（42/1920） | 未经本任务 SFT 的基座表现 |
| LoRA val relaxed accuracy | 78.44%（1506/1920） | 完整 ChartQA SFT 后的表现 |

> 当前的 78.44% 是 validation 结果。若以后根据 validation 去选择 epoch、学习率、rank 或图像尺寸，validation 就成为开发集；最后应在尚未使用过的 test 上报告泛化结果。

---

## 1. SFT 到底在做什么

```text
图片 + 问题 + 标准答案
        ↓
统一 JSONL 样本
        ↓
messages / chat template
        ↓
processor
        ↓
input_ids + pixel_values + attention_mask
        ↓
labels：只保留答案 token
        ↓
模型预测下一个 token
        ↓
answer token 的 cross-entropy loss
        ↓
backward + AdamW
        ↓
只更新 LoRA A/B，基座保持冻结
```

训练使用 **teacher forcing**：标准答案在模型输入中出现，但模型只因“是否正确预测答案 token”而产生 loss。

```python
# prompt、图片 token、assistant 起始标记都能被模型看到，
# 但这些位置不参加 loss。
labels[row, :prefix_length] = -100

# batch 对齐产生的 padding 也不能学习。
labels[attention_mask == 0] = -100
```

推理则没有标准答案：模型根据图片和问题自回归生成。

```python
# 推理时 messages 中没有 response。
generated_ids = model.generate(**inputs, do_sample=False)
```

---

## 2. 按创建顺序理解文件与学习路径

每个脚本只验证一个新概念。这样出现问题时能定位到“模型、数据、loss、LoRA、训练循环”中的具体一层，而不是在一个大框架里盲猜。

### Stage 0：环境与仓库骨架

| 文件 / 目录 | 作用 | 原因 | 熟练后可替代为 |
|---|---|---|---|
| `README.md` | 路线、结论、复现入口 | 让实验从终端历史变为可读档案 | `docs/`、MkDocs、实验报告站 |
| `.gitignore` | 排除模型、数据、checkpoint、密钥 | 避免 GB 级文件进入 GitHub | Git LFS、对象存储 |
| `configs/` | 预留训练配置文件 | 超参数不应只留在命令历史中 | YAML + Hydra/OmegaConf |
| `notes/` | 预留概念与问题记录 | 代码和理解分开沉淀 | Jupyter Book |
| `experiments/` | 摘要、预测和图表 | 让结论可以审计 | W&B、MLflow、TensorBoard |
| `outputs/` | adapter、checkpoint、训练图 | 大文件留在服务器 | Hub / 对象存储 |

当前服务器事实：Conda 环境为 `qwen3vl`；主线以共享 3090 的单卡 `cuda:0` 作为基线。模型来自本地路径：

```text
/data/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/
snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/
```

### Stage 1：先推理，确认模型能工作

| 文件 | 作用 | 输入 → 输出 |
|---|---|---|
| `scripts/inference.py` | 最小图文推理 | 图片路径 + 问题 → 模型回答 |
| `data/test/example.jpg` | 最初的示例图片 | 图片文件，不是文本源码 |

```python
# processor 负责把人类可读的图片和文本，转换为模型张量。
inputs = processor(..., return_tensors="pt")

# generate 只用于推理；SFT 训练不会直接调用它。
answer_ids = model.generate(**inputs, max_new_tokens=128)
```

先跑推理的原因：训练前先排除模型路径、图片路径、GPU 与 processor 的问题。之后可以替换为 `transformers.pipeline`、vLLM、SGLang 或服务 API；学习阶段保留原生 `generate()` 最透明。

### Stage 2：多模态 processor 不是黑盒

Stage 2 的输入检查脚本打印一条图片消息经过 chat template 与 processor 后的中间结果：

```text
messages
  ↓ apply_chat_template()
<|vision_start|>、<|image_pad|> 等视觉占位模板
  ↓ process_vision_info() + processor()
input_ids / pixel_values / attention_mask / image_grid_thw
```

| 字段 | 中文解释 |
|---|---|
| `input_ids` | 文本 token；图片位置会展开成视觉占位 token |
| `pixel_values` | 图片经过 patch 预处理后的视觉输入 |
| `attention_mask` | 1 是有效 token，0 是 batch padding |
| `mm_token_type_ids` | 标记文本与多模态位置 |
| `image_grid_thw` | 视觉网格的时间/高/宽信息 |

最初直接处理原始大图曾触发 OOM，因此后续代码显式控制图片大小。图片分辨率决定视觉 token 数量，也直接影响显存。

熟练后可使用 Hugging Face `datasets`、TRL VLM collator 等封装；但不论框架如何变化，最终仍要得到以上模型张量。

### Stage 3：SFT labels 与 loss

| 文件 | 作用 | 学到什么 |
|---|---|---|
| `scripts/inspect_sft_labels.py` | 打印 token、labels 与掩码位置 | 模型不学习复述用户问题，只学习回答 |
| `scripts/forward_sft_loss.py` | 只做一次 forward 并输出 loss/logits/显存 | labels 如何真正参与 loss |

```text
图片 token + 用户问题 + assistant 起始标记  → labels = -100
标准答案 token                              → labels = 正确 token id
```

loss 是每个未掩码答案 token 的交叉熵，再取平均。`forward_sft_loss.py` **不调用** `backward()`；它只是证明数据与 loss 正确连接，是进入训练前的安全检查。

熟练后可替换为 `DataCollatorForLanguageModeling`、自定义 `compute_loss_func` 或 TRL 的 collator；多模态场景仍必须确认答案以外 token 被正确掩码。

### Stage 4：LoRA 的真实更新

| 文件 | 作用 | 观察重点 |
|---|---|---|
| `scripts/inspect_lora.py` | 打印 LoRA 参数名、配置、A/B 范数 | LoRA 加在何处、可训练参数有多少 |
| `scripts/one_step_lora.py` | 执行一次 backward 和 optimizer step | 梯度怎样改变 LoRA 参数 |

LoRA 没有拆掉原始线性层，而是在冻结权重 `W` 上增加低秩增量：

```text
W_new = W + (alpha / r) × B × A

W：冻结的基座权重
A：[r, input_dim]
B：[output_dim, r]
```

当前配置：

```python
LoraConfig(
    r=8,                    # 低秩容量；越大，adapter 参数越多。
    lora_alpha=16,          # 缩放；当前 alpha / r = 2。
    lora_dropout=0.0,       # 当前不加入 LoRA dropout。
    target_modules=["q_proj", "v_proj"],  # attention 的 Q/V 投影。
    bias="none",           # 不额外训练 bias。
)
```

`target_modules` 会匹配所有对应 attention 层；日志只展示了前几层参数名。PEFT 常将 `B` 初始化为 0，所以开始时 `B×A=0`，模型与基座完全一致；第一步中 A 梯度接近 0、B 有梯度是正常现象。

熟练后可试验 `k_proj`、`o_proj`、MLP、DoRA、AdaLoRA 或 QLoRA。先理解 A/B 的梯度和更新，才能判断这些替换实际改变了什么。

### Stage 5：单样本最小 SFT 闭环

| 文件 | 作用 |
|---|---|
| `scripts/train_lora_minimal.py` | 用一条样本重复训练，保存最小 adapter |
| `scripts/inference_with_lora.py` | 载入 base 与 adapter，观察回答差异 |
| `data/demo/train.jsonl` | 三条食物图片问答 demo |

这一步不是追求泛化，而是验证完整链路：图片读取 → labels → loss → LoRA 更新 → 保存 → 重新加载 → 推理。

```json
{
  "id": "food-demo-001",
  "image": "data/test/example.jpg",
  "prompt": "请描述这张图片。",
  "response": "木桌上摆着一碗辣椒炒肉，旁边还有一碟配菜。"
}
```

一条样本训练 30 次，不等于有 30 条数据；它只是让同一答案多次产生梯度，用于观察最小闭环和记忆现象。这个脚本以后仍有价值：它是任何大训练故障时的最小排错版本。

### Stage 6：Dataset、DataLoader 与 batch

| 文件 | 作用 | 增加的能力 |
|---|---|---|
| `scripts/train_lora_batched.py` | JSONL 的 batched SFT | `Dataset`、`DataLoader`、padding、逐行 labels 掩码 |
| `data/demo/eval.jsonl` | 小型未见措辞验证集 | 第一次区分训练与评估 |
| `scripts/evaluate_lora.py` | base / adapter 基础生成评估 | 固定输入下进行比较 |

核心是 `SFTCollator`。每条样本 prompt 长度不同，因此 labels 必须逐行处理：

```python
# 不能把同一个 prefix 长度套给整个 batch。
for row, prefix_length in enumerate(prefix_lengths.tolist()):
    labels[row, :prefix_length] = -100
```

熟练后可换成 `datasets.Dataset`、DataPipes、Hugging Face collator 或 TRL `SFTTrainer` 的数据约定。当前手写版的价值是每一个 token 的来源都清楚。

### Stage 7–9：真实 ChartQA Mini

| 文件 | 作用 |
|---|---|
| `scripts/inspect_chartqa_stream.py` | 流式查看原始 ChartQA 字段 |
| `scripts/prepare_chartqa_mini.py` | 导出 Mini 图片、train.jsonl、val.jsonl |
| `scripts/inspect_prepared_dataset.py` | 检查 schema、图片路径和数量 |
| `data/chartqa_mini/train.jsonl` / `val.jsonl` | 100 条 train / 20 条 val 的真实任务输入 |

统一样本格式：

```json
{
  "id": "chartqa-train-000000",
  "image": "data/chartqa_mini/images/train/000000.jpg",
  "prompt": "图表问题",
  "response": "SFT 使用的一个标准答案",
  "all_answers": ["所有可接受答案"],
  "human_or_machine": "原始来源标记"
}
```

`response` 用作 SFT 的 teacher-forcing target；`all_answers` 留给评估，因为一个问题可能有多个可接受答案。它也为未来的 `chosen/rejected` 偏好数据留下接口。Mini 原始数据可由脚本再生成，因此默认不提交 Git。

### Stage 10：完整 ChartQA 单卡 LoRA SFT

| 文件 | 作用 |
|---|---|
| `scripts/prepare_chartqa_full.py` | 导出完整 ChartQA `train` / `val` 到本地 JSONL 与图片 |
| `scripts/train_chartqa_full.py` | 当前正式单卡原生 PyTorch trainer |
| `outputs/chartqa-full-lora/` | 本地 checkpoint、final adapter、metrics、loss 图，不提交 Git |

完整数据（服务器本地）：

```text
data/chartqa_full/
├── train.jsonl       # 28,299 条；唯一参与 LoRA 更新的数据
├── val.jsonl         # 1,920 条；只用于验证
└── images/
    ├── train/
    └── val/
```

训练器不是 `transformers.Trainer`，也不是 TRL `SFTTrainer`，而是手写 PyTorch 循环：

```python
# 每个 micro batch：forward + backward，但尚不一定更新参数。
raw_loss = model(**batch, labels=labels).loss
(raw_loss / gradient_accumulation_steps).backward()

# 累积 8 次后：梯度裁剪、更新 LoRA、清空梯度。
torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

正式训练器额外处理：

```text
gradient accumulation  # batch=1 也能获得有效 batch=8。
gradient clipping      # 防止异常梯度破坏更新。
checkpoint             # adapter + AdamW state + 下一 batch 位置。
metrics.jsonl          # 保存训练与验证指标。
loss_curve.png         # 从 metrics 绘制 loss 曲线。
```

训练命令：

```bash
# 只使用 GPU 0；完整训练不要带 --max-optimizer-steps。
CUDA_VISIBLE_DEVICES=0 python scripts/train_chartqa_full.py \
  --output-dir outputs/chartqa-full-lora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 20 \
  --checkpoint-steps 500
```

恢复命令：

```bash
# checkpoint 中同时保存 LoRA、optimizer state 与恢复位置。
CUDA_VISIBLE_DEVICES=0 python scripts/train_chartqa_full.py \
  --output-dir outputs/chartqa-full-lora \
  --epochs 1 \
  --resume-from outputs/chartqa-full-lora/checkpoints/step-000500
```

### Stage 11：完整 validation 的生成式评估

| 文件 | 作用 |
|---|---|
| `scripts/evaluate_chartqa_full.py` | 对 base 或 adapter 在 val 上自由生成答案并评分 |
| `scripts/compare_chartqa_full_results.py` | 同一 id 配对比较 improved / regressed |
| `experiments/chartqa_full_base_val.jsonl` | 基座逐条预测 |
| `experiments/chartqa_full_lora_val.jsonl` | LoRA 逐条预测 |
| `experiments/*summary.json` | 两个模型的准确率摘要 |
| `experiments/chartqa_full_comparison.json` | 配对比较与错误样例 |
| `experiments/chartqa_full_comparison.png` | base / LoRA 柱状对比图 |

评估时没有 `response`、没有 labels、没有 backward：

```python
# 推理只有图片和问题；没有标准答案。
messages = [{"role": "user", "content": [image, question]}]

# do_sample=False：贪心生成，保证 base 与 LoRA 比较可复现。
prediction_ids = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=32,
)
```

评分规则为 relaxed accuracy：文本答案规范化后精确匹配；数值答案允许相对误差不超过 5%；匹配 `all_answers` 中任意一个即可。

```text
base 错 + LoRA 对  → improved
base 对 + LoRA 错  → regressed
两者都对           → both_correct
两者都错           → both_wrong
```

比较脚本曾暴露一个真实工程错误：overall 统计中把 `improved` 和 `regressed` 归属反了。修正后的关键代码是：

```python
# 基座答对 = 两者都对 + 基座对但 LoRA 退化。
base_correct_count = len(both_correct) + len(regressed)

# LoRA 答对 = 两者都对 + 基座错但 LoRA 改正。
lora_correct_count = len(both_correct) + len(improved)
```

这次经历说明：图表不是生成后就天然可信，必须用原始 summary、逐条预测和配对逻辑交叉验证。

---

## 3. 当前 trainer 用了什么库，没用什么库

```text
PyTorch
  ├── Dataset / DataLoader
  ├── 手写 collator
  ├── loss.backward()
  ├── AdamW optimizer.step()
  └── checkpoint / 指标写入

Transformers
  ├── Qwen3VLForConditionalGeneration
  └── AutoProcessor

PEFT
  └── get_peft_model()：向 q_proj / v_proj 注入 LoRA

qwen_vl_utils
  └── process_vision_info()：读取多模态消息中的图片

Matplotlib
  └── 从 metrics.jsonl 生成 loss_curve.png 与对比图
```

当前**没有使用**：

```text
TRL SFTTrainer       # 尚未使用
transformers.Trainer # 尚未使用
Accelerate / DDP     # 尚未使用
DeepSpeed            # 尚未使用
bitsandbytes / QLoRA # 尚未使用
```

这不是功能缺失，而是学习设计：先看懂透明的原生流程，再使用框架自动化它。

| 当前实现 | 熟练后可替代为 | 替代后得到什么 |
|---|---|---|
| 手写 PyTorch loop | `transformers.Trainer` | 标准化日志、checkpoint、训练参数管理 |
| 手写 collator | TRL `SFTTrainer` + VLM 数据格式 | 更少样板代码，并自然接入 DPO/GRPO |
| 单卡 `cuda:0` | `accelerate launch` / `torchrun` / DDP | 多卡数据并行 |
| bf16 LoRA | QLoRA + bitsandbytes | 更低显存，但引入量化复杂度 |
| JSONL + Matplotlib | W&B / MLflow / TensorBoard | 更强实验追踪和可视化 |

> 不急于用 TRL 覆盖当前脚本。手写 trainer 已经把 `labels → loss → backward → optimizer → LoRA update` 的因果关系展示出来；之后再用 TRL，才能明确它替你封装了什么。

---

## 4. SFT 如何连接未来 RL

当前 SFT 样本：

```json
{
  "image": "path/to/chart.jpg",
  "prompt": "What is the value in 2024?",
  "response": "51"
}
```

未来偏好学习样本可以扩展为：

```json
{
  "image": "path/to/chart.jpg",
  "prompt": "What is the value in 2024?",
  "chosen": "51",
  "rejected": "61"
}
```

两部分的目标不同：

```text
SFT：学习“面对图片和问题，应怎样产生任务答案”。
DPO / GRPO：在多个候选答案中，学习“哪个回答更符合偏好或 reward”。
```

进入 RL 前应保留：

- `outputs/chartqa-full-lora/final_adapter/`；
- 基座模型路径、processor 和 chat template；
- 图片尺寸、随机种子、训练配置；
- train / val / test 划分；
- loss 图、预测结果和 relaxed accuracy 规则。

---

## 5. Git 与数据管理

### 提交到 GitHub

```text
scripts/                         # 可运行、可复现的流程代码
experiments/*.json / *.png       # 轻量摘要、预测、对比图
data/demo/                       # 小型教学 demo
data/test/example.jpg            # 小型示例图片
README.md / notes/               # 概念解释、实验记录
```

### 留在服务器、不提交 GitHub

```text
data/chartqa_full/               # 完整原始数据与图片
data/chartqa_mini/               # 可由脚本再生成的数据
outputs/                         # adapter、checkpoint、optimizer state、训练图
Hugging Face cache               # 模型与数据缓存
HF token / 其他密钥              # 绝不提交
```

推荐的阶段闭环：

```text
理解一个概念
  ↓
写一个小脚本验证
  ↓
保存结果或图表
  ↓
更新 README / notes
  ↓
Git commit
```

---

## 6. 暂停后的下一步

当前可以安全暂停。恢复时不必同时做所有事，可以选择一条独立路线：

1. **Stage 12：test 最终盲测**
   导出 ChartQA test，对固定的 base 与 LoRA 配置做最终评估；看过 test 后不再据它调参。

2. **SFT 工程对照**
   用同一数据和超参数写 `transformers.Trainer` 或 TRL `SFTTrainer` 版本，对照它们与当前手写 trainer 的职责边界。

3. **SFT 扩展实验**
   在 val 上一次只改一个变量，例如 epoch、learning rate、LoRA rank、分辨率、QLoRA 或多卡，并保留每次配置和结果。

完成 SFT 的沉淀后，才进入 Project B：偏好数据构造 → DPO → 可验证 reward → GRPO / RL。

---

> **核心原则：先用一张 GPU、一个透明的原生训练器，真正理解从图文数据到 LoRA 参数更新的全过程；再用 TRL、QLoRA 和多卡把已经理解的流程工程化、规模化，最后把 SFT adapter 作为 RL 的起点。**
