"""
model_factory.py
================
Unified model factory for the ExCV pipeline.

Usage:
    from models.model_factory import get_model

    model = get_model("efficientnet", pretrained=True, freeze_backbone=True)
    model = get_model("vit",          pretrained=True, freeze_backbone=True)

Author: ExCV Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal

from efficientnet import EfficientNetClassifier
from vit import ViTClassifier


# ---------------------------------------------------------------------------
# Supported model names
# ---------------------------------------------------------------------------

ModelName = Literal["efficientnet", "vit"]

MODEL_REGISTRY = {
    "efficientnet": EfficientNetClassifier,
    "vit":          ViTClassifier,
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_model(
    name: ModelName,
    num_classes: int  = 3,
    pretrained:  bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model name. One of: 'efficientnet', 'vit'.
    num_classes : int
        Number of output classes. Default: 3.
    pretrained : bool
        Use ImageNet pretrained weights. Default: True.
    freeze_backbone : bool
        Freeze backbone for Stage 1 training. Default: True.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    model = model_cls(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    print(f"\n[ModelFactory] Built model: {name}")
    total, trainable = model.count_parameters()
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {total - trainable:,}")

    return model


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    path: Path | str,
) -> None:
    """
    Save a full training checkpoint.

    Saved keys:
        epoch, model_state_dict, optimizer_state_dict,
        val_loss, val_acc

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    epoch : int
    val_loss : float
    val_acc : float
    path : Path | str
        File path to save checkpoint (.pth).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss"            : val_loss,
        "val_acc"             : val_acc,
    }, path)

    print(f"[Checkpoint] Saved → {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: nn.Module,
    path: Path | str,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).

    Parameters
    ----------
    model : nn.Module
        Model to load weights into.
    path : Path | str
        Path to .pth checkpoint file.
    optimizer : torch.optim.Optimizer | None
        If provided, optimizer state is also restored.
    device : torch.device | None
        Target device. Defaults to CPU.

    Returns
    -------
    dict
        Full checkpoint dictionary.
    """
    path   = Path(path)
    device = device or torch.device("cpu")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch    = checkpoint.get("epoch", -1)
    val_acc  = checkpoint.get("val_acc", 0.0)
    val_loss = checkpoint.get("val_loss", 0.0)

    print(f"[Checkpoint] Loaded ← {path}")
    print(f"  epoch={epoch}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")

    return checkpoint


# ---------------------------------------------------------------------------
# Device utility
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Return best available device: CUDA > MPS > CPU.

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = get_device()

    print("\n" + "=" * 60)
    print("Testing EfficientNet factory")
    print("=" * 60)
    eff_model = get_model("efficientnet", pretrained=True, freeze_backbone=True)
    eff_model = eff_model.to(device)

    dummy = torch.randn(2, 3, 224, 224).to(device)
    out   = eff_model(dummy)
    assert out.shape == (2, 3)
    print(f"Output shape: {out.shape}  ✅")

    print("\n" + "=" * 60)
    print("Testing ViT factory")
    print("=" * 60)
    vit_model = get_model("vit", pretrained=True, freeze_backbone=True)
    vit_model = vit_model.to(device)

    out = vit_model(dummy)
    assert out.shape == (2, 3)
    print(f"Output shape: {out.shape}  ✅")

    print("\n" + "=" * 60)
    print("Testing checkpoint save/load")
    print("=" * 60)
    optimizer = torch.optim.Adam(eff_model.parameters(), lr=1e-3)
    save_checkpoint(
        model=eff_model,
        optimizer=optimizer,
        epoch=1,
        val_loss=0.42,
        val_acc=0.87,
        path="outputs/checkpoints/test_checkpoint.pth",
    )
    load_checkpoint(
        model=eff_model,
        path="outputs/checkpoints/test_checkpoint.pth",
        optimizer=optimizer,
        device=device,
    )

    print("\n✅ Model factory fully validated.")