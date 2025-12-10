# config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    # 模型与输出
    model_name: str = "zai-org/GLM-Z1-9B-0414"  # 换成你实际用的 GLM-Z1 名
    output_dir: str = "./outputs_glm_z1_math"

    # 数据
    max_length: int = 2048
    max_train_samples_per_level: int | None = None  # 限制每个 level 样本数（调试用）

    # 训练
    per_device_batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs_per_level: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # curriculum：从易到难
    level_order: List[str] = field(
        default_factory=lambda: [
            "Level 1",
            "Level 2",
            "Level 3",
            "Level 4",
            "Level 5",
        ]
    )

    # 其他
    logging_steps: int = 50
    save_per_level: bool = True
    seed: int = 42
