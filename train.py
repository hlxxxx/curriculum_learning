# train.py
import argparse
from config import TrainingConfig
from trainer import train_with_curriculum

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Curriculum learning fine-tuning of GLM-Z1 on competition_math."
    )

    parser.add_argument("--model_name", type=str, default="zai-org/GLM-Z1-9B-0414")
    parser.add_argument("--output_dir", type=str, default="./outputs_glm_z1_math")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs_per_level", type=int, default=1)
    parser.add_argument("--max_train_samples_per_level", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--level_order",
        type=str,
        default="Level 1,Level 2,Level 3,Level 4,Level 5",
        help="用逗号分隔的 level 顺序；可改为 Level 5,...,Level 1 做 hard→easy 对比",
    )

    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_epochs_per_level=args.num_epochs_per_level,
        max_train_samples_per_level=args.max_train_samples_per_level,
        logging_steps=args.logging_steps,
        seed=args.seed,
        level_order=[s.strip() for s in args.level_order.split(",")],
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()

    train_with_curriculum(cfg)
