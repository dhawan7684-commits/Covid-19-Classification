"""
train.py
========
Main training entry point for the ExCV pipeline.

Runs full two-stage training:
    Stage 1 — Frozen backbone, head only    (5 epochs)
    Stage 2 — Full fine-tuning              (20 epochs)

Usage:
    # Train EfficientNet (default)
    python src/training/train.py

    # Train ViT
    python src/training/train.py --model vit

    # Custom epochs
    python src/training/train.py --model efficientnet --s1_epochs 5 --s2_epochs 30

Outputs:
    outputs/checkpoints/stage1_best.pth   ← best Stage 1 model
    outputs/checkpoints/best_model.pth    ← best overall model
    outputs/logs/tensorboard/             ← TensorBoard logs

Author: ExCV Project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataloader     import build_dataloaders
from models.model_factory import get_model, get_device
from training.losses      import get_loss_fn
from training.scheduler   import (
    get_stage1_optimizer, get_stage1_scheduler,
    get_stage2_optimizer, get_stage2_scheduler,
)
from training.trainer     import Trainer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DATASET_ROOT   = PROJECT_ROOT / "COVID_19_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "outputs" / "logs" / "tensorboard"


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model"      : "efficientnet",
    "batch_size" : 32,
    "num_workers": 4,
    "s1_epochs"  : 5,
    "s2_epochs"  : 20,
    "s1_lr"      : 1e-3,
    "s2_backbone_lr": 1e-4,
    "s2_head_lr" : 1e-3,
    "patience"   : 5,
    "loss"       : "label_smoothing",
    "seed"       : 42,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExCV Training Script")
    parser.add_argument("--model",       type=str,   default=DEFAULT_CONFIG["model"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_workers", type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--s1_epochs",   type=int,   default=DEFAULT_CONFIG["s1_epochs"])
    parser.add_argument("--s2_epochs",   type=int,   default=DEFAULT_CONFIG["s2_epochs"])
    parser.add_argument("--s1_lr",       type=float, default=DEFAULT_CONFIG["s1_lr"])
    parser.add_argument("--s2_backbone_lr", type=float, default=DEFAULT_CONFIG["s2_backbone_lr"])
    parser.add_argument("--s2_head_lr",  type=float, default=DEFAULT_CONFIG["s2_head_lr"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--loss",        type=str,   default=DEFAULT_CONFIG["loss"])
    parser.add_argument("--seed",        type=int,   default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = get_device()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 70)
    print("ExCV TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  S1 epochs  : {args.s1_epochs}")
    print(f"  S2 epochs  : {args.s2_epochs}")
    print(f"  Loss       : {args.loss}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. DataLoaders
    # ----------------------------------------------------------------
    loaders = build_dataloaders(
        dataset_root=DATASET_ROOT,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
    )

    # ----------------------------------------------------------------
    # 2. Model (Stage 1: frozen backbone)
    # ----------------------------------------------------------------
    model = get_model(
        name=args.model,
        pretrained=True,
        freeze_backbone=True,
    )

    # ----------------------------------------------------------------
    # 3. Loss
    # ----------------------------------------------------------------
    loss_fn = get_loss_fn(args.loss)

    # ----------------------------------------------------------------
    # 4. Trainer
    # ----------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loaders=loaders,
        loss_fn=loss_fn,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        device=device,
    )

    # ----------------------------------------------------------------
    # 5. Stage 1 — Head only
    # ----------------------------------------------------------------
    opt1 = get_stage1_optimizer(model, lr=args.s1_lr)
    sch1 = get_stage1_scheduler(opt1, epochs=args.s1_epochs)

    trainer.run_stage1(
        optimizer=opt1,
        scheduler=sch1,
        epochs=args.s1_epochs,
        patience=args.patience,
    )

    # ----------------------------------------------------------------
    # 6. Stage 2 — Full fine-tuning
    # ----------------------------------------------------------------
    # Unfreeze backbone
    model.unfreeze_backbone()

    opt2 = get_stage2_optimizer(
        model,
        backbone_lr=args.s2_backbone_lr,
        head_lr=args.s2_head_lr,
    )
    sch2 = get_stage2_scheduler(opt2, epochs=args.s2_epochs)

    trainer.run_stage2(
        optimizer=opt2,
        scheduler=sch2,
        epochs=args.s2_epochs,
        patience=args.patience,
    )

    # ----------------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best val_acc : {trainer.best_val_acc:.4f}")
    print(f"  Checkpoint   : {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"  TensorBoard  : tensorboard --logdir {LOG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()