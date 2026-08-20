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
TRL SFTTrainer 框架对照
  ↓
冻结 SFT adapter（未来 RL 的起点）
```

Project A：SFT 已完成到 Stage 12。Project B（偏好学习、DPO、GRPO/RL）会复用这里的 SFT adapter、数据格式和评估约定；后续按单一阶段推进，每完成、验证和记录一个阶段后再进入下一个。

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
- [x] Stage 12：TRL `SFTTrainer` 完整训练、评估与手写 trainer 对照
- [ ] Stage 13：加载并验证 SFT policy，建立 RL 基线
- [ ] Stage 14：构造小规模偏好数据
- [ ] Stage 15：单 batch DPO 原理与最小闭环
- [ ] Stage 16：完整 DPO 实验与评估
- [ ] Stage 17：ChartQA 可验证 reward 设计与测试
- [ ] Stage 18：小规模 GRPO 学习闭环
- [ ] Stage 19：GRPO 扩展实验与 SFT / DPO / GRPO 对比
- [ ] Stage 20（可选）：DPO → GRPO、Reward Model 与 PPO

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
| 手写 LoRA val relaxed accuracy | 78.44%（1506/1920） | Stage 10/11 的完整 ChartQA SFT 表现 |
| TRL LoRA val relaxed accuracy | 78.44%（1506/1920） | Stage 12，与手写 trainer 完全一致 |
| TRL train loss / val loss | 0.2334 / 0.2556 | `SFTTrainer` 在 3,538 steps 后的记录 |
| RL 起始 adapter | `outputs/chartqa-full-trl-lora/final_adapter/` | 只读保存；后续每条 RL 路线都从它独立分支 |

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

### Stage 12：TRL `SFTTrainer` 框架对照

这一阶段没有追求更高分，而是用同一份完整 ChartQA 数据、同一 LoRA 配置和同一评估规则，确认 TRL 的框架训练能复现手写 trainer 的结果。这样进入 DPO / GRPO 前，已经知道框架替代了哪些样板代码，而没有跳过 SFT 的底层因果链路。

| 文件 / 目录 | 作用 |
|---|---|
| `scripts/train_chartqa_trl.py` | 使用 `trl.SFTTrainer` 的完整单卡 VLM SFT |
| `scripts/plot_trl_history.py` | 从 `trainer_state.json` 生成 loss 曲线 |
| `experiments/chartqa_full_trl_val.jsonl` | TRL adapter 的逐条 validation 生成结果 |
| `experiments/chartqa_full_trl_val.summary.json` | 生成式评分摘要 |
| `outputs/chartqa-full-trl-lora/final_adapter/` | 选定的、供后续 RL 复用的 SFT adapter |

数据在进入框架前从 `image / prompt / response` 映射为图像、prompt 和 completion；仍然只对 completion 计算 loss。关键配置保持不变：28,299 条 train、1,920 条 val、1 epoch、`batch_size=1`、梯度累积 8、`learning_rate=1e-4`、cosine scheduler、bf16、gradient checkpointing，以及 `r=8 / alpha=16 / q_proj,v_proj` 的 LoRA。

| 指标 | 手写 trainer | TRL `SFTTrainer` |
|---|---:|---:|
| validation relaxed accuracy | 78.44%（1506/1920） | 78.44%（1506/1920） |
| train loss | — | 0.2334 |
| validation loss | — | 0.2556 |
| 训练步数 | — | 3,538 |

结论：两条路线的生成式准确率完全一致。Stage 12 证明的是训练框架迁移的等价性，不是模型质量提升；因此后续 RL 的统一起点固定为：

```text
base model  = Qwen3-VL-4B-Instruct
SFT adapter = outputs/chartqa-full-trl-lora/final_adapter/
SFT baseline = validation relaxed accuracy 78.44%
```

该 adapter 视为只读基线，DPO、GRPO 和任何后续实验都保存到新的输出目录，绝不覆盖它。

---

## 3. 当前 trainer 用了什么库，没用什么库

```text
PyTorch
  ├── Dataset / DataLoader
  ├── 手写 collator
  ├── loss.backward()
  ├── AdamW optimizer.step()
  └── 手写 trainer 的 checkpoint / 指标写入

Transformers
  ├── Qwen3VLForConditionalGeneration
  └── AutoProcessor

PEFT
  └── get_peft_model()：向 q_proj / v_proj 注入 LoRA

TRL
  └── SFTTrainer：Stage 12 的数据整理、loss、optimizer、scheduler、checkpoint 与日志封装

qwen_vl_utils
  └── process_vision_info()：读取多模态消息中的图片

Matplotlib
  └── 从 metrics.jsonl 生成 loss_curve.png 与对比图
```

当前**还没有使用**：

```text
transformers.Trainer # 尚未使用
Accelerate / DDP     # 尚未使用
DeepSpeed            # 尚未使用
bitsandbytes / QLoRA # 尚未使用
DPOTrainer / GRPOTrainer / PPOTrainer
```

这不是功能缺失，而是学习顺序：先看懂透明的原生流程，再验证框架自动化是否等价，最后才将同一个 SFT policy 接入偏好优化与 RL。

| 已掌握实现 | 对应职责 | 后续学习连接点 |
|---|---|---|
| 手写 PyTorch loop | 手工展示 `labels → loss → backward → optimizer → LoRA update` | 理解 DPO / GRPO 的 loss 或 reward 从何而来 |
| TRL `SFTTrainer` | 标准化训练、评估、checkpoint、日志 | 以同一 TRL / PEFT 栈自然进入 `DPOTrainer` / `GRPOTrainer` |
| 单卡 `cuda:0` | 可控的显存、时间与随机性 | RL 第一轮仍坚持单卡、小数据、小生成组 |
| bf16 LoRA | 低成本训练 adapter | 每种 RL 算法保存独立 adapter，不覆盖 SFT 基线 |
| JSONL + Matplotlib | 可复核的数据和指标记录 | 继续保存偏好对、reward 统计和逐条预测 |

> Stage 12 没有覆盖或删除手写 trainer。两套实现并存：手写版本用于理解，TRL 版本作为后续 RL 学习的工程起点。

---

## 4. 从 SFT adapter 开始的 RL 学习计划

Project B 的目标是学习完整的偏好优化与可验证奖励流程，不是立即追求一个更高的数字。整个项目固定复用 Stage 12 的 SFT policy：

```text
base model  = Qwen3-VL-4B-Instruct
SFT adapter = outputs/chartqa-full-trl-lora/final_adapter/
baseline    = validation relaxed accuracy 78.44%
```

第一轮 DPO 和第一轮 GRPO 都从该 SFT adapter 独立分支。这样可以清楚区分“DPO 带来的变化”和“GRPO 带来的变化”；只有两条路线都掌握后，才把 DPO adapter 继续接到 GRPO。

### 数据与评估边界

当前 SFT 样本：

```json
{
  "image": "path/to/chart.jpg",
  "prompt": "What is the value in 2024?",
  "response": "51"
}
```

Stage 14 形成的偏好样本：

```json
{
  "image": "path/to/chart.jpg",
  "prompt": "What is the value in 2024?",
  "chosen": "51",
  "rejected": "61",
  "source": "sft-generation-or-rule",
  "preference_reason": "chosen matches the chart answer"
}
```

- RL 训练只使用 train 或单独划出的 RL-train 数据；validation 和 test 不进入偏好对构造、reward 拟合或参数更新。
- 每个阶段都在相同的 1,920 条 validation 上报告 relaxed accuracy；test 保持为最终一次性泛化评估，不参与路线选择。
- `final_adapter/` 永远只读；新产物命名为 `outputs/chartqa-dpo-lora/`、`outputs/chartqa-grpo-lora/` 等独立目录。

### 学习顺序与阶段验收

| 阶段 | 本阶段只学习什么 | 产物与通过条件 |
|---|---|---|
| Stage 13：SFT policy 基线 | 加载 base + Stage 12 adapter；区分 policy、冻结 reference policy、reward、rollout 四个角色 | 用同一评估脚本复现约 78.44%；记录模型路径、generation 配置和显存 |
| Stage 14：偏好数据 | 让 SFT policy 为同一图文问题生成多个候选；按答案正确性优先、格式规范次之，标出 `chosen / rejected` | 人工检查 20～50 对；检查重复、训练/验证泄漏和“仅因答案更长而获胜”的偏差；此阶段不训练 |
| Stage 15：最小 DPO | 在一个小 batch 上逐项观察 chosen/rejected 的 log-prob、reference log-prob 和 DPO loss | 能解释 loss 的方向；用 20～50 对做最小过拟合实验；保存笔记和日志 |
| Stage 16：正式 DPO | 用 TRL `DPOTrainer` 训练一个小规模、可复现实验；policy 从 SFT adapter 初始化，reference 固定为训练前的 SFT policy | 保存 `chartqa-dpo-lora`；同时报告 preference 指标与 ChartQA validation relaxed accuracy，确认没有遗忘 SFT 任务 |
| Stage 17：可验证 reward | 为 ChartQA 写 reward：答案 relaxed match 为主，数值容差和输出格式为辅；枚举正确、错误、空答案、绕格式等样例 | reward 函数单独测试通过；检查不会给冗长解释、投机格式或空回答异常高分；此阶段不训练 |
| Stage 18：最小 GRPO | 从原始 SFT adapter 出发，对一个 prompt 采样小组候选，理解组内相对奖励与 advantage | 小数据、小生成组完成一次闭环；记录 reward 分布、零方差组比例、样例与 validation 指标 |
| Stage 19：GRPO 对照 | 扩大到受控规模，但保持模型、数据划分与评估规则不变 | 在同一 validation 上比较 Base / SFT / DPO / GRPO 的准确率、格式错误和训练成本；保存独立 adapter |
| Stage 20（可选） | DPO → GRPO 的串联、Reward Model、PPO | 只在前述分支都有清晰结论后再开始；PPO 最后学习，因为它额外引入 value / rollout / 显存复杂度 |

### 为什么按这个顺序

```text
SFT policy 基线
  ↓
偏好对（离线、可人工审查）
  ↓
DPO（先理解相对偏好优化）
  ↓
可验证 reward（先验证奖励本身）
  ↓
GRPO（再学习带采样的策略优化）
  ↓
统一比较；最后才进入 PPO / Reward Model
```

DPO 是离线偏好优化：它比较 `chosen` 与 `rejected` 相对冻结 reference policy 的概率，不需要先训练显式 reward model。GRPO 是基于采样候选和 reward 的策略优化。因此先把偏好数据和 reward 分别做干净，再让模型优化它们。

建议的新增目录只用于未来 Project B，不改动现有 SFT 复现脚本：

```text
rl/
  data/          # 小型偏好对、reward 测试样例与数据清单
  scripts/       # 每个阶段的最小可运行脚本
  experiments/   # 指标、样例、对比结果
  notes/         # 本阶段原理、疑问和结论
  configs/       # 可复现实验配置
```

每一阶段都遵守同一个闭环：先写清要验证的问题，再跑最小实验，检查失败样例和指标，最后更新 README / notes；没有通过验收就不进入下一阶段。

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

下一步只做 **Stage 13：SFT policy 基线**，不要同时启动 DPO 和 GRPO：

1. 加载 `Qwen3-VL-4B-Instruct + outputs/chartqa-full-trl-lora/final_adapter/`。
2. 在固定 validation 配置上复现约 78.44% relaxed accuracy。
3. 将这个可运行对象命名为 `sft_policy`，并保存一次不可训练的 reference snapshot。
4. 写一页笔记：policy、reference policy、候选生成、reward 分别是什么；列出 Stage 14 需要的偏好数据字段。

Stage 13 验收通过后，才进入 Stage 14 的 20～50 对偏好数据。任何一阶段如果 validation 明显退化、样本检查不通过或概念还解释不清，就留在当前阶段排查，不跳到下一种算法。

---

> **核心原则：先用一张 GPU 和透明的手写 trainer 理解 SFT；再用 Stage 12 验证 TRL 框架等价；之后始终从冻结的 SFT adapter 分支，以“数据 → 单 batch → 小实验 → 统一评估”的节奏学习 DPO 和 GRPO。**
