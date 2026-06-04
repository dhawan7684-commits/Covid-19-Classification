"""
efficientnet.py
===============
EfficientNet-B0 classification model for COVID-19 chest X-ray classification.

Architecture:
    - Backbone  : EfficientNet-B0 (pretrained on ImageNet)
    - Head      : Dropout → Linear(1280 → 3)
    - Classes   : COVID | Normal | Viral Pneumonia

Transfer Learning Strategy:
    Stage 1 — Frozen backbone, train head only       (5 epochs)
    Stage 2 — Unfreeze all layers, fine-tune entire network  (20 epochs)

Author: ExCV Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from typing import Tuple


NUM_CLASSES = 3
DROPOUT_RATE = 0.3


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 3-class chest X-ray classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default: 3.
    dropout : float
        Dropout rate before the final classifier. Default: 0.3.
    pretrained : bool
        Load ImageNet pretrained weights. Default: True.
    freeze_backbone : bool
        Freeze backbone weights for Stage 1 training. Default: True.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_RATE,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Backbone — EfficientNet-B0
        # ----------------------------------------------------------------
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # ----------------------------------------------------------------
        # Replace classifier head
        # in_features = 1280 for EfficientNet-B0
        # ----------------------------------------------------------------
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        # ----------------------------------------------------------------
        # Optionally freeze backbone for Stage 1
        # ----------------------------------------------------------------
        if freeze_backbone:
            self.freeze_backbone()

        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 3, 224, 224].

        Returns
        -------
        torch.Tensor
            Logits of shape [B, num_classes].
        """
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers except the classifier head.
        Used in Stage 1 (head-only training).
        """
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[EfficientNet] Backbone FROZEN — training classifier head only.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all layers for full fine-tuning.
        Used in Stage 2 (full network training).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[EfficientNet] Backbone UNFROZEN — fine-tuning entire network.")

    def unfreeze_top_blocks(self, num_blocks: int = 3) -> None:
        """
        Unfreeze only the top N MBConv blocks of the backbone.
        Useful for gradual unfreezing between Stage 1 and Stage 2.

        Parameters
        ----------
        num_blocks : int
            Number of top feature blocks to unfreeze. Default: 3.
        """
        features = list(self.backbone.features.children())
        for block in features[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"[EfficientNet] Top {num_blocks} blocks UNFROZEN.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns
        -------
        Tuple[int, int]
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the backbone feature extractor (without classifier).
        Used by Grad-CAM to hook into the final conv layer.

        Returns
        -------
        nn.Module
        """
        return self.backbone.features

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"EfficientNetClassifier(\n"
            f"  backbone    = EfficientNet-B0\n"
            f"  num_classes = {self.num_classes}\n"
            f"  total_params     = {total:,}\n"
            f"  trainable_params = {trainable:,}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EfficientNet-B0 — Stage 1 (frozen backbone)")
    print("=" * 60)
    model = EfficientNetClassifier(freeze_backbone=True)
    print(model)

    dummy = torch.randn(4, 3, 224, 224)
    logits = model(dummy)
    print(f"\nInput  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    assert logits.shape == (4, 3), f"Unexpected output shape: {logits.shape}"

    print("\n" + "=" * 60)
    print("EfficientNet-B0 — Stage 2 (unfreeze all)")
    print("=" * 60)
    model.unfreeze_backbone()
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    print("\n✅ EfficientNet-B0 validated.")