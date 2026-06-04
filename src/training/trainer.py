"""
trainer.py
==========
Full two-stage training loop for chest X-ray classification.

Stage 1: Frozen backbone, train head only        (5 epochs,  lr=1e-3)
Stage 2: Full fine-tuning, all layers unfrozen   (20 epochs, lr=1e-4/1e-3)

Features:
    - Early stopping (patience=5 on val_loss)
    - Best model checkpointing
    - Per-epoch metric logging (loss, accuracy)
    - TensorBoard logging
    - Gradient clipping (Stage 2)

Author: ExCV Project
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model_factory import save_checkpoint, get_device


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val_loss does not improve for `patience` epochs.

    Parameters
    ----------
    patience : int   — epochs without improvement before stopping. Default: 5.
    delta    : float — minimum improvement to qualify. Default: 1e-4.
    """

    def __init__(self, patience: int = 5, delta: float = 1e-4) -> None:
        self.patience  = patience
        self.delta     = delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """
        Update state. Returns True if training should stop.

        Parameters
        ----------
        val_loss : float

        Returns
        -------
        bool — True if early stopping triggered
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Single epoch functions
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    torch.device,
    clip_grad: float | None = None,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader — training loader
    optimizer : torch.optim.Optimizer
    loss_fn   : nn.Module
    device    : torch.device
    clip_grad : float | None — gradient clipping max norm. None = disabled.

    Returns
    -------
    Tuple[float, float]
        (avg_loss, accuracy)
    """
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model:   nn.Module,
    loader:  DataLoader,
    loss_fn: nn.Module,
    device:  torch.device,
) -> Tuple[float, float]:
    """
    Run one validation epoch (no gradients).

    Returns
    -------
    Tuple[float, float]
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = loss_fn(logits, labels)

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Two-stage trainer for EfficientNet / ViT classification.

    Parameters
    ----------
    model          : nn.Module
    loaders        : dict — {'train': DataLoader, 'val': DataLoader}
    loss_fn        : nn.Module
    checkpoint_dir : Path — where to save best model
    log_dir        : Path — TensorBoard log directory
    device         : torch.device
    """

    def __init__(
        self,
        model:          nn.Module,
        loaders:        Dict[str, DataLoader],
        loss_fn:        nn.Module,
        checkpoint_dir: Path,
        log_dir:        Path,
        device:         torch.device,
    ) -> None:
        self.model          = model.to(device)
        self.loaders        = loaders
        self.loss_fn        = loss_fn
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device         = device
        self.writer         = SummaryWriter(log_dir=str(log_dir))
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        self.best_val_acc  = 0.0
        self.global_epoch  = 0

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def run_stage1(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        epochs:    int = 5,
        patience:  int = 5,
    ) -> None:
        """
        Stage 1: Train classifier head with frozen backbone.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        scheduler : LR scheduler
        epochs    : int
        patience  : int — early stopping patience
        """
        print("\n" + "=" * 70)
        print("STAGE 1 — Head Training (backbone frozen)")
        print("=" * 70)

        early_stop = EarlyStopping(patience=patience)

        for epoch in range(1, epochs + 1):
            self.global_epoch += 1
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                self.model, self.loaders["train"],
                optimizer, self.loss_fn, self.device,
                clip_grad=None,   # no clipping in Stage 1
            )
            val_loss, val_acc = validate_one_epoch(
                self.model, self.loaders["val"],
                self.loss_fn, self.device,
            )

            scheduler.step()
            self._log_epoch(epoch, "S1", train_loss, train_acc, val_loss, val_acc, t0)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_best("stage1_best.pth", optimizer, val_loss, val_acc)

            if early_stop.step(val_loss):
                print(f"  [EarlyStopping] Triggered at epoch {epoch}")
                break

        print(f"\nStage 1 complete. Best val_acc: {self.best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def run_stage2(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        epochs:    int = 20,
        patience:  int = 5,
    ) -> None:
        """
        Stage 2: Full fine-tuning with unfrozen backbone.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        scheduler : LR scheduler
        epochs    : int
        patience  : int — early stopping patience
        """
        print("\n" + "=" * 70)
        print("STAGE 2 — Full Fine-Tuning (backbone unfrozen)")
        print("=" * 70)

        early_stop = EarlyStopping(patience=patience)

        for epoch in range(1, epochs + 1):
            self.global_epoch += 1
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                self.model, self.loaders["train"],
                optimizer, self.loss_fn, self.device,
                clip_grad=1.0,    # gradient clipping in Stage 2
            )
            val_loss, val_acc = validate_one_epoch(
                self.model, self.loaders["val"],
                self.loss_fn, self.device,
            )

            scheduler.step()
            self._log_epoch(epoch, "S2", train_loss, train_acc, val_loss, val_acc, t0)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_best("best_model.pth", optimizer, val_loss, val_acc)

            if early_stop.step(val_loss):
                print(f"  [EarlyStopping] Triggered at epoch {epoch}")
                break

        print(f"\nStage 2 complete. Best val_acc: {self.best_val_acc:.4f}")
        self.writer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch:      int,
        stage:      str,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_acc:    float,
        t0:         float,
    ) -> None:
        elapsed = time.time() - t0
        print(
            f"  [{stage}] Epoch {epoch:>3}  |  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  |  "
            f"{elapsed:.1f}s"
        )

        # TensorBoard
        self.writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss},   self.global_epoch)
        self.writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},    self.global_epoch)

        # History
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    def _save_best(
        self,
        filename:  str,
        optimizer: torch.optim.Optimizer,
        val_loss:  float,
        val_acc:   float,
    ) -> None:
        save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            epoch=self.global_epoch,
            val_loss=val_loss,
            val_acc=val_acc,
            path=self.checkpoint_dir / filename,
        )