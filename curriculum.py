# curriculum.py
from typing import Dict, List, Any
from config import TrainingConfig
from data import load_competition_math, group_by_level


def build_curriculum(cfg: TrainingConfig) -> Dict[str, List[dict]]:
    """
    加载 competition_math，将其按 level 分组并返回。
    你也可以在这里做 sample 截断（max_train_samples_per_level）。
    """
    examples = load_competition_math()
    level_groups = group_by_level(examples)

    if cfg.max_train_samples_per_level is not None:
        for level, exs in level_groups.items():
            level_groups[level] = exs[: cfg.max_train_samples_per_level]

    return level_groups
