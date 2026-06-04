"""
losses.py
=========
Loss functions for chest X-ray classification.

Provides:
    - LabelSmoothingCrossEntropy  : Standard CE with label smoothing
    - FocalLoss                   : Down-weights easy examples, focus on hard ones
    - get_loss_fn()               : Factory function

Recommendation:
    Start with LabelSmoothingCrossEntropy (smoothing=0.1).
    Switch to FocalLoss if class performance diverges after epoch 5.

Author: ExCV Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# ---------------------------------------------------------------------------
# Label Smoothing Cross Entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Prevents overconfident predictions by softening hard targets.
    Recommended smoothing=0.1 for medical imaging.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor in [0, 1). Default: 0.1.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, C]  — raw model outputs (before softmax)
        targets : Tensor [B]     — integer class labels

        Returns
        -------
        Tensor — scalar loss
        """
        num_classes = logits.size(1)
        log_probs   = F.log_softmax(logits, dim=1)

        # Smooth targets: (1 - ε) for true class, ε/(C-1) for others
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    alpha : float
        Weighting factor for rare class. Default: 0.25.
    gamma : float
        Focusing parameter. γ=0 → standard CE. Default: 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, C]
        targets : Tensor [B]

        Returns
        -------
        Tensor — scalar loss
        """
        ce_loss  = F.cross_entropy(logits, targets, reduction="none")
        pt       = torch.exp(-ce_loss)
        focal    = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LossName = Literal["cross_entropy", "label_smoothing", "focal"]

def get_loss_fn(name: LossName = "label_smoothing", **kwargs) -> nn.Module:
    """
    Get a loss function by name.

    Parameters
    ----------
    name : str
        One of: 'cross_entropy', 'label_smoothing', 'focal'.
    **kwargs
        Passed to the loss constructor.

    Returns
    -------
    nn.Module
    """
    registry = {
        "cross_entropy"   : lambda: nn.CrossEntropyLoss(**kwargs),
        "label_smoothing" : lambda: LabelSmoothingCrossEntropy(**kwargs),
        "focal"           : lambda: FocalLoss(**kwargs),
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(registry.keys())}")
    loss_fn = registry[name]()
    print(f"[Loss] Using: {loss_fn.__class__.__name__}")
    return loss_fn


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C = 8, 3
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    for name in ["cross_entropy", "label_smoothing", "focal"]:
        fn   = get_loss_fn(name)
        loss = fn(logits, targets)
        print(f"  {name:<25} loss={loss.item():.4f}  shape={loss.shape}")

    print("\n✅ All loss functions validated.")