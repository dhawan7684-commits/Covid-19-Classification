"""
validate_preprocessing.py
==========================
End-to-end validation of the preprocessing pipeline.

Checks:
    1.  All splits load without error
    2.  Tensor shapes are correct  [3, 224, 224]
    3.  Normalized value range is reasonable  (roughly -3 to +3)
    4.  Class distribution is printed for each split
    5.  Generates a visual grid of augmented training samples

Output:
    outputs/figures/preprocessing_sample_grid.png

Run from project root:
    python src/data/validate_preprocessing.py

Author: ExCV Project
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Make src importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset    import ChestXRayDataset, IDX_TO_CLASS, CLASS_NAMES
from data.transforms import get_transforms, get_inverse_normalize
from data.dataloader import build_dataloaders


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMG_SIZE    = 224
GRID_ROWS   = 3      # one row per class
GRID_COLS   = 4      # samples per class in grid
FIGURE_PATH = Path(__file__).resolve().parents[2] / "outputs" / "figures" / "preprocessing_sample_grid.png"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_shapes(loaders: dict) -> bool:
    """Verify all batch tensors have shape [B, 3, 224, 224]."""
    print("\n[1] Shape Check")
    ok = True
    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        expected_shape = (images.shape[0], 3, IMG_SIZE, IMG_SIZE)
        shape_ok = images.shape == torch.Size(expected_shape)
        status = "✅" if shape_ok else "❌"
        print(f"    {status}  [{split:>5}]  images={images.shape}  labels={labels.shape}")
        if not shape_ok:
            ok = False
    return ok


def check_value_range(loaders: dict) -> bool:
    """Verify normalized tensors have reasonable range after normalization."""
    print("\n[2] Value Range Check  (expect roughly -3 to +3)")
    ok = True
    for split, loader in loaders.items():
        images, _ = next(iter(loader))
        vmin, vmax, vmean = images.min().item(), images.max().item(), images.mean().item()
        range_ok = -5.0 < vmin and vmax < 5.0
        status = "✅" if range_ok else "❌"
        print(f"    {status}  [{split:>5}]  min={vmin:.4f}  max={vmax:.4f}  mean={vmean:.4f}")
        if not range_ok:
            ok = False
    return ok


def check_class_distribution(loaders: dict) -> None:
    """Print per-class sample counts for each split."""
    print("\n[3] Class Distribution")
    for split, loader in loaders.items():
        dataset: ChestXRayDataset = loader.dataset
        counts = dataset.get_class_counts()
        total = len(dataset)
        print(f"\n    [{split}]  {total} total")
        for cls in CLASS_NAMES:
            count = counts.get(cls, 0)
            pct   = 100 * count / total if total > 0 else 0
            print(f"      {cls:<22} {count:>5}  ({pct:5.1f}%)")


def check_dtype(loaders: dict) -> bool:
    """Verify tensors are float32."""
    print("\n[4] Dtype Check  (expect torch.float32)")
    ok = True
    for split, loader in loaders.items():
        images, _ = next(iter(loader))
        dtype_ok = images.dtype == torch.float32
        status = "✅" if dtype_ok else "❌"
        print(f"    {status}  [{split:>5}]  dtype={images.dtype}")
        if not dtype_ok:
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Visual grid
# ---------------------------------------------------------------------------

def generate_sample_grid(train_dataset: ChestXRayDataset) -> None:
    """
    Generate a grid of augmented training samples per class.
    Saves to FIGURE_PATH.
    """
    inv_normalize = get_inverse_normalize()

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 2.5, GRID_ROWS * 2.5),
    )
    fig.suptitle("Augmented Training Samples (per class)", fontsize=13, y=1.01)

    from dataset import CLASS_TO_IDX

    for row_idx, class_name in enumerate(CLASS_NAMES):
        # Collect indices for this class
        class_label = CLASS_TO_IDX[class_name]
        class_indices = [
            i for i, (_, lbl) in enumerate(train_dataset.samples)
            if lbl == class_label
        ][:GRID_COLS]

        for col_idx, sample_idx in enumerate(class_indices):
            tensor, _ = train_dataset[sample_idx]
            # Inverse normalize for display
            display = inv_normalize(tensor).clamp(0, 1)
            # Take first channel (grayscale)
            np_img = display[0].numpy()

            ax = axes[row_idx][col_idx]
            ax.imshow(np_img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[5] Sample grid saved → {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "COVID_19_dataset"

    print("=" * 72)
    print("PREPROCESSING PIPELINE VALIDATION")
    print("=" * 72)

    # Build loaders (num_workers=0 for validation script)
    loaders = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
    )

    all_ok = True
    all_ok &= check_shapes(loaders)
    all_ok &= check_value_range(loaders)
    check_class_distribution(loaders)
    all_ok &= check_dtype(loaders)

    # Generate visual grid from training set
    if "train" in loaders:
        generate_sample_grid(loaders["train"].dataset)

    print("\n" + "=" * 72)
    if all_ok:
        print("✅  ALL CHECKS PASSED — preprocessing pipeline is ready.")
    else:
        print("❌  SOME CHECKS FAILED — review output above.")
    print("=" * 72)


if __name__ == "__main__":
    main()