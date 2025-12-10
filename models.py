# models.py
import os
import torch
from typing import Tuple, List

from config import TrainingConfig

from modelscope import snapshot_download          # ★ 从 ModelScope 下载权重
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def get_default_lora_targets(model) -> List[str]:
    """
    针对 GLM-Z1/GLM-4 类模型的一组常见 LoRA 注入层名。
    建议你 print(model) 看一下模块名后再微调。
    """
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def resolve_model_path(model_name: str) -> str:
    """
    统一处理两种情况：
    1）传进来是本地目录：/path/to/ZhipuAI/GLM-Z1-9B-0414
       → 直接用这个路径。
    2）传进来是 ModelScope 名字：ZhipuAI/GLM-Z1-9B-0414
       → 用 snapshot_download 从 ModelScope 下到 ./pretrained 再返回本地路径。
    """
    # 情况1：已经是本地目录
    if os.path.isdir(model_name):
        return model_name

    # 情况2：当成 ModelScope 的 repo id
    local_dir = snapshot_download(
        model_name,               # 例如 "ZhipuAI/GLM-Z1-9B-0414"
        cache_dir="./pretrained", # 下载到当前目录下的 ./pretrained/...
        revision="master",
    )
    return local_dir


def load_glm_z1_with_lora(
    cfg: TrainingConfig,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    从 ModelScope（或本地路径）加载 GLM-Z1，然后插入 LoRA，返回 tokenizer 和 model。
    """

    model_path = resolve_model_path(cfg.model_name)

    # 用 transformers 的 AutoTokenizer / AutoModel 加载本地目录
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,    # 只用本地文件，不去 HuggingFace
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True, 
        local_files_only=True,
    )


    target_modules = get_default_lora_targets(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model
