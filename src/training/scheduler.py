"""
scheduler.py
============
Learning rate scheduler configuration for two-stage training.

Stage 1 — Head only (frozen backbone):
    Optimizer : Adam, lr=1e-3
    Scheduler : StepLR (decay every 2 epochs)
    Epochs    : 5

Stage 2 — Full fine-tuning (unfrozen backbone):
    Optimizer : Adam with layer-wise LR (backbone lr=1e-4, head lr=1e-3)
    Scheduler : CosineAnnealingLR
    Epochs    : 20

Author: ExCV Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Optimizer factories
# ---------------------------------------------------------------------------

def get_stage1_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """
    Stage 1 optimizer — only trains classifier head parameters.

    Parameters
    ----------
    model : nn.Module
    lr    : float — learning rate. Default: 1e-3.

    Returns
    -------
    torch.optim.Adam
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    print(f"[Optimizer] Stage 1 — Adam(lr={lr})  |  {len(trainable)} param groups")
    return optimizer


def get_stage2_optimizer(
    model: nn.Module,
    backbone_lr: float = 1e-4,
    head_lr:     float = 1e-3,
) -> torch.optim.Optimizer:
    """
    Stage 2 optimizer — layer-wise learning rates.

    Backbone layers use a smaller lr to preserve pretrained features.
    Classifier head uses a larger lr for faster adaptation.

    Parameters
    ----------
    model       : nn.Module
    backbone_lr : float — lr for backbone layers. Default: 1e-4.
    head_lr     : float — lr for classifier head. Default: 1e-3.

    Returns
    -------
    torch.optim.Adam
    """
    # Separate backbone and head parameters
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "heads" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    print(f"[Optimizer] Stage 2 — Adam  |  "
          f"backbone_lr={backbone_lr}  head_lr={head_lr}  |  "
          f"backbone_params={len(backbone_params)}  head_params={len(head_params)}")
    return optimizer


# ---------------------------------------------------------------------------
# Scheduler factories
# ---------------------------------------------------------------------------

SchedulerName = Literal["cosine", "step", "plateau"]


def get_stage1_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
) -> StepLR:
    """
    Stage 1 scheduler — StepLR, decays by 0.5 every 2 epochs.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    epochs    : int — total Stage 1 epochs. Default: 5.

    Returns
    -------
    StepLR
    """
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    print(f"[Scheduler] Stage 1 — StepLR(step_size=2, gamma=0.5)  epochs={epochs}")
    return scheduler


def get_stage2_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 20,
    eta_min: float = 1e-6,
) -> CosineAnnealingLR:
    """
    Stage 2 scheduler — CosineAnnealingLR.

    Smoothly decays lr from initial value to eta_min over training.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    epochs    : int   — total Stage 2 epochs. Default: 20.
    eta_min   : float — minimum lr. Default: 1e-6.

    Returns
    -------
    CosineAnnealingLR
    """
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    print(f"[Scheduler] Stage 2 — CosineAnnealingLR(T_max={epochs}, eta_min={eta_min})")
    return scheduler


def get_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    patience: int = 3,
    factor:   float = 0.5,
) -> ReduceLROnPlateau:
    """
    ReduceLROnPlateau — reduce lr when val_loss stops improving.
    Optional alternative to cosine for Stage 2.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    patience  : int   — epochs without improvement before reducing. Default: 3.
    factor    : float — factor to reduce lr. Default: 0.5.

    Returns
    -------
    ReduceLROnPlateau
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=patience,
        factor=factor,
        verbose=True,
    )
    print(f"[Scheduler] Plateau — ReduceLROnPlateau(patience={patience}, factor={factor})")
    return scheduler


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from models.efficientnet import EfficientNetClassifier

    model = EfficientNetClassifier(freeze_backbone=True)

    print("\n--- Stage 1 ---")
    opt1  = get_stage1_optimizer(model, lr=1e-3)
    sch1  = get_stage1_scheduler(opt1, epochs=5)

    print("\n--- Stage 2 ---")
    model.unfreeze_backbone()
    opt2  = get_stage2_optimizer(model, backbone_lr=1e-4, head_lr=1e-3)
    sch2  = get_stage2_scheduler(opt2, epochs=20)

    # Simulate LR decay over 5 epochs
    print("\nStage 2 LR schedule (simulated):")
    for epoch in range(1, 6):
        sch2.step()
        lrs = [pg["lr"] for pg in opt2.param_groups]
        print(f"  Epoch {epoch}: backbone_lr={lrs[0]:.2e}  head_lr={lrs[1]:.2e}")

    print("\n✅ Schedulers validated.")