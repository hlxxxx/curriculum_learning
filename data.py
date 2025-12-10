# data.py
from typing import Dict, List, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset
# 注意：我们使用 torch.utils.data.Dataset 作为基类，
# 如果使用 load_dataset，则不需要导入 datasets.Dataset。
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_competition_math() -> List[Dict[str, Any]]:
    """
    加载 qwedsacf/competition_math 并转换为统一格式：
    {
        "input_text": prompt,
        "label_text": solution,
        "level": "Level 1" ... "Level 5",
        "type": "Algebra" / ...
    }
    """
    # ★ 关键修改：使用 load_dataset，并选择 'train' split
    raw_ds = load_dataset("qwedsacf/competition_math")["train"]
    
    # 从 Dataset 对象中提取列表形式的字段
    # 使用 .to_list() 或直接访问字段名可以获取 Python list
    problem = raw_ds["problem"]
    solution = raw_ds["solution"]
    level = raw_ds["level"]
    qtype = raw_ds["type"]

    def _build_example(ex):
        
        prompt = (
            "You are a helpful math tutor. "
            "Solve the following problem step by step and give the final answer.\n\n"
            f"Problem: {ex[0]}\n\nAnswer:"
        )

        return {
            "input_text": prompt,
            "label_text": ex[1],  # ex[1] 是 solution
            "level": ex[2],  # ex[2] 是 level
            "type": ex[3],  # ex[3] 是 type
        }

    # 将所有字段打包并生成 examples 列表
    examples = [ _build_example(ex) for ex in zip(problem, solution, level, qtype) ]
    
    return examples


def group_by_level(
    examples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    按 level 分组，返回 { level_str: [examples] }
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        level = ex["level"]
        groups[level].append(ex)
    return groups


class MathQADataset(Dataset):
    """
    简单 Dataset，将 input_text + label_text 编码为单序列。
    labels 中 prompt 部分设为 -100，只对答案部分计算 loss。
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["input_text"]
        answer = ex["label_text"]

        # 估算 prompt token 数，用于构造 label mask
        # 注意：truncation 可能导致边界略有偏差，但足够作为模板使用
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full_text = prompt + "\n" + answer

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = -100  # 不对 prompt 部分计算 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }