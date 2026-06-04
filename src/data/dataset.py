"""
dataset.py
==========
Custom PyTorch Dataset for the COVID-19 chest X-ray classification task.

Folder structure expected:
    COVID_19_dataset/
    ├── train/
    │   ├── COVID/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    ├── val/
    │   └── ...
    └── test/
        └── ...

Author: ExCV Project
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Class label mapping  (alphabetical → deterministic)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = ["COVID", "Normal", "Viral Pneumonia"]
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray image classification.

    Loads images from a split directory, converts to RGB, and applies
    the provided transform pipeline.

    Parameters
    ----------
    root_dir : Path | str
        Path to the split directory (e.g., COVID_19_dataset/train).
    transform : Callable | None
        Torchvision transform pipeline.  If None, only ToTensor is applied.
    class_to_idx : Dict[str, int] | None
        Custom class-to-index mapping.  Defaults to CLASS_TO_IDX.

    Attributes
    ----------
    samples : List[Tuple[Path, int]]
        (image_path, label_index) pairs.
    classes : List[str]
        Sorted list of class folder names found in root_dir.
    class_to_idx : Dict[str, int]
        Mapping from class name to integer label.
    """

    def __init__(
        self,
        root_dir: Path | str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.ToTensor()
        self.class_to_idx = class_to_idx or CLASS_TO_IDX

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.samples: List[Tuple[Path, int]] = self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Walk each class folder and collect (path, label) pairs.

        Returns
        -------
        List[Tuple[Path, int]]
        """
        samples = []
        for class_name in self.classes:
            if class_name not in self.class_to_idx:
                print(f"[WARNING] Unknown class folder '{class_name}' — skipping.")
                continue
            label = self.class_to_idx[class_name]
            class_dir = self.root_dir / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append((img_path, label))
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load image and return (tensor, label).

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, int]
            (image_tensor [3, H, W], integer label)
        """
        img_path, label = self.samples[idx]

        # Open as RGB (grayscale X-rays → 3 identical channels for EfficientNet)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_class_counts(self) -> Dict[str, int]:
        """
        Return per-class sample counts.

        Returns
        -------
        Dict[str, int]
            { class_name: count }
        """
        counts: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
        for _, label in self.samples:
            class_name = IDX_TO_CLASS[label]
            counts[class_name] += 1
        return counts

    def get_labels(self) -> List[int]:
        """
        Return all labels as a flat list (used by WeightedRandomSampler).

        Returns
        -------
        List[int]
        """
        return [label for _, label in self.samples]

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        lines = [
            f"ChestXRayDataset(root={self.root_dir})",
            f"  Total samples : {len(self)}",
            f"  Classes       : {self.classes}",
        ]
        for cls, count in counts.items():
            lines.append(f"    {cls:<20} {count:>5} images")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from data.transforms import get_transforms

    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "COVID_19_dataset"

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[SKIP] {split_dir} not found.")
            continue

        ds = ChestXRayDataset(
            root_dir=split_dir,
            transform=get_transforms(split),
        )
        print(ds)
        img, label = ds[0]
        print(f"  Sample tensor shape : {img.shape}")
        print(f"  Sample label        : {label} ({IDX_TO_CLASS[label]})\n")

    print("✅ Dataset class validated.")