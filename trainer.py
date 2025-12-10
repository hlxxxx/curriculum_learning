# trainer.py
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import TrainingConfig
from models import load_glm_z1_with_lora
from data import MathQADataset, collate_fn
from curriculum import build_curriculum

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_level(
    level: str,
    examples: List[dict],
    cfg: TrainingConfig,
    tokenizer,
    model,
    # device: torch.device,
    optimizer,
    accelerator: Accelerator,
    global_step_start: int = 0,
):
    """
    在某一个难度 level 上训练 num_epochs_per_level 轮。
    返回更新后的 global_step。
    """
    # 创建数据集
    dataset = MathQADataset(examples, tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    
    num_update_steps_per_epoch = max(
        len(dataloader) // cfg.gradient_accumulation_steps, 1
    )
    total_training_steps = cfg.num_epochs_per_level * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_training_steps),
        num_training_steps=total_training_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    model.train()
    global_step = global_step_start

    for epoch in range(cfg.num_epochs_per_level):
            # 禁用非主进程的进度条
            pbar = tqdm(dataloader, desc=f"[{level}] epoch {epoch+1}/{cfg.num_epochs_per_level}", disable=not accelerator.is_main_process)
            optimizer.zero_grad()
    
            for step, batch in enumerate(pbar):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    # 1. 仅使用原始的 loss，accelerator 会在需要时自动缩放
                    loss = outputs.loss 
                    
                    # 2. 使用 accelerator.backward()
                    accelerator.backward(loss) 
                
                if accelerator.sync_gradients: 
                    # 3. 使用 accelerator 的裁剪，并只在同步时执行
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm) 
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # 4. 进度条更新（在主进程，且使用未缩放的 loss）
                if accelerator.is_main_process and global_step % cfg.logging_steps == 0:
                    # 注意：这里 loss.item() 已经是在当前设备上的平均损失，直接打印即可。
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"}) 
    
    return global_step


def train_with_curriculum(cfg: TrainingConfig):
    set_seed(cfg.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    print(f"[{accelerator.process_index}/{accelerator.num_processes}] Device: {accelerator.device}, CUDA visible: {torch.cuda.device_count()}")
    
    tokenizer, model = load_glm_z1_with_lora(cfg)

    # 构建 curriculum（按 level 分组）
    level_groups = build_curriculum(cfg)

    # 优化器（只会对 LoRA 可训练参数生效）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # 将模型和优化器准备好进行分布式训练
    model, optimizer = accelerator.prepare(model, optimizer) 
    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    for level in cfg.level_order:
        if level not in level_groups:
            print(f"[WARN] Level {level} not in dataset, skip.")
            continue

        examples = level_groups[level]
        print(f"\n===== Training on {level} | {len(examples)} samples =====")

        global_step = train_one_level(
            level=level,
            examples=examples,
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            global_step_start=global_step,
            accelerator=accelerator,
        )

        if cfg.save_per_level:
            # ★ 1. 等待所有进程完成当前 Level 的训练
            accelerator.wait_for_everyone() 
            
            # ★ 2. 仅在主进程执行 I/O (保存 Checkpoint)
            if accelerator.is_main_process: 
                out_dir = os.path.join(cfg.output_dir, f"checkpoint_{level.replace(' ', '_')}")
                
                # ★ 3. 使用 unwrap_model 获取原始的 PeftModel (LoRA) 实例进行保存