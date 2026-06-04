"""
vit.py
======
Vision Transformer (ViT-B/16) for COVID-19 chest X-ray classification.

Used as the secondary model for:
    - Comparison against EfficientNet-B0
    - Attention Rollout explainability (Task 6)

Architecture:
    - Backbone  : ViT-B/16 (pretrained on ImageNet-21k → ImageNet-1k)
    - Head      : Dropout → Linear(768 → 3)
    - Classes   : COVID | Normal | Viral Pneumonia

Note:
    ViT requires more data and compute than EfficientNet-B0.
    On a dataset of ~3,994 images, EfficientNet-B0 is the primary model.
    ViT is included for explainability comparison (Attention Rollout vs Grad-CAM).

Author: ExCV Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from typing import Tuple


NUM_CLASSES  = 3
DROPOUT_RATE = 0.1
HIDDEN_DIM   = 768   # ViT-B/16 CLS token dimension


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) fine-tuned for 3-class chest X-ray classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default: 3.
    dropout : float
        Dropout rate before the final classifier. Default: 0.1.
    pretrained : bool
        Load ImageNet pretrained weights. Default: True.
    freeze_backbone : bool
        Freeze backbone for Stage 1 training. Default: True.
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
        # Backbone — ViT-B/16
        # ----------------------------------------------------------------
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # ----------------------------------------------------------------
        # Replace classification head
        # in_features = 768 for ViT-B/16
        # ----------------------------------------------------------------
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

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
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "heads" not in name:
                param.requires_grad = False
        print("[ViT] Backbone FROZEN — training classifier head only.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[ViT] Backbone UNFROZEN — fine-tuning entire network.")

    def unfreeze_top_blocks(self, num_blocks: int = 4) -> None:
        """
        Unfreeze the top N transformer encoder blocks.

        Parameters
        ----------
        num_blocks : int
            Number of encoder blocks to unfreeze from the top. Default: 4.
        """
        encoder_blocks = list(self.backbone.encoder.layers.children())
        for block in encoder_blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"[ViT] Top {num_blocks} encoder blocks UNFROZEN.")

    # ------------------------------------------------------------------
    # Attention hooks (for Attention Rollout — Task 6)
    # ------------------------------------------------------------------

    def get_attention_weights(self) -> list:
        """
        Register forward hooks to capture attention weights from all
        transformer blocks. Used by AttentionRollout in Task 6.

        Returns
        -------
        list
            List of attention weight tensors after one forward pass.
        """
        attention_weights = []

        def hook_fn(module, input, output):
            # output[1] is attention weights when need_weights=True
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])

        hooks = []
        for block in self.backbone.encoder.layers:
            h = block.self_attention.register_forward_hook(hook_fn)
            hooks.append(h)

        return attention_weights, hooks

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> Tuple[int, int]:
        """
        Return (total_params, trainable_params).

        Returns
        -------
        Tuple[int, int]
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"ViTClassifier(\n"
            f"  backbone    = ViT-B/16\n"
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
    print("ViT-B/16 — Stage 1 (frozen backbone)")
    print("=" * 60)
    model = ViTClassifier(freeze_backbone=True)
    print(model)

    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(f"\nInput  : {dummy.shape}")
    print(f"Output : {logits.shape}")
    assert logits.shape == (2, 3), f"Unexpected output shape: {logits.shape}"

    print("\n" + "=" * 60)
    print("ViT-B/16 — Stage 2 (unfreeze top 4 blocks)")
    print("=" * 60)
    model.unfreeze_top_blocks(num_blocks=4)
    total, trainable = model.count_parameters()
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    print("\n✅ ViT-B/16 validated.")