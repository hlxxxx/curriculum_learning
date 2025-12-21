# 探究课程学习对大模型解题能力的影响

本项目实现了一个完整的「课程学习（Curriculum Learning）」驱动的大语言模型（GLM-Z1）参数高效微调框架，用于数学类推理任务。其核心思想是将数学题目按照难度（Level 1 至 Level 5）分阶段训练，实现模型从易到难逐步适应复杂推理任务。该框架支持 LoRA 微调、混合精度训练、数据重放机制和结构化评估。

---

## 🔧 项目结构说明

```
.
├── config.py              # 训练配置类
├── curriculum.py          # 构建课程（分阶段）数据
├── data.py                # 数据加载、分组与编码
├── evaluate.py            # 模型评估脚本
├── models.py              # 模型加载 + LoRA 注入
├── trainer.py             # 主训练循环（含重放机制）
├── train.py               # 训练主入口
├── split_dataset.py       # 数据集切分（带分层保证）
└── dataset/
    └── split_dataset/
        ├── train.parquet
        └── test.parquet
```

---

## 依赖环境安装

使用以下命令安装依赖项：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `transformers`
- `peft`
- `accelerate`
- `datasets`
- `pyarrow`
- `sympy`
- `tqdm`
- `modelscope`

---

## 数据格式与处理

使用 HuggingFace 上的 [competition_math](https://huggingface.co/datasets/qwedsacf/competition_math) 数据集或自定义 `.parquet` 文件，字段包括：

```json
{
  "problem": "题目描述",
  "solution": "标准解答",
  "level": "Level 1" 至 "Level 5",
  "type": "Algebra" / "Counting & Probability" / ...
}
```

可使用 `split_dataset.py` 进行层级分层划分，生成 `train.parquet` 和 `test.parquet` 文件。

---

## 启动训练（课程学习）

示例命令如下（默认从 Level 1 到 Level 5）：

```bash
python train.py   --model_name ./pretrained/zai-org/GLM-Z1-9B-0414   --output_dir ./outputs_glm_z1_math   --max_length 1024   --batch_size 4   --learning_rate 2e-4   --num_epochs_per_level 1   --max_train_samples_per_level 1000   --use_level
```

每个 level 会单独训练若干 epoch，并将样本加入 replay buffer 保持历史记忆，避免“灾难性遗忘”。

---

## 模型评估

模型评估支持两个数据集：

- `competition_math`
- `combicbench`（组合数学题）

评估命令如下：

```bash
python evaluate.py   --model_name ./pretrained/zai-org/GLM-Z1-9B-0414   --eval_model_path ./outputs_glm_z1_math/checkpoint_last   --eval_dataset_name competition_math   --eval_dataset_file ./dataset/split_dataset/test.parquet   --eval_output_file eval_results.jsonl   --batch_size 4   --max_new_tokens 512
```

评估方式包括：

- 严格字符串匹配
- 数学表达式等价（使用 SymPy 判断）
- 数值近似判断（浮点容差）

---

## LoRA 参数注入说明

项目采用 LoRA 进行微调，仅对部分参数注入权重，目标模块包括：

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]
```

LoRA 配置示例：

```python
LoraConfig(
  r=64,
  lora_alpha=128,
  lora_dropout=0.05,
  target_modules=...,
  task_type="CAUSAL_LM"
)
```

---

## 模型保存与重启

- 每个 level 的训练结果保存在 `outputs_glm_z1_math/level_LevelX/`
- 所有训练阶段的最终 checkpoint 保存在 `outputs_glm_z1_math/checkpoint_last/`
- 可中断恢复训练（断点续训），自动记录当前 level 与 step 信息

---

## Replay Buffer 机制

每次训练新 level 时，会引入一定比例（默认 30%）的历史样本（最大 5000 条）进行混合训练，防止模型“遗忘”前期学习内容。

---

## 开发建议

- 修改 `config.py` 中的 `level_order` 可控制训练顺序（例如反向训练）
- 支持 resume_from_checkpoint 参数进行训练断点恢复
- 可接入 CombiBench 数据集做泛化能力测试

