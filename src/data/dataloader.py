"""
dataloader.py
=============
DataLoader factory for the COVID-19 chest X-ray classification pipeline.

Handles:
    - Weighted random sampling to address class imbalance
    - Reproducible worker seeding
    - Configurable batch size, num_workers, pin_memory

Author: ExCV Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ChestXRayDataset, CLASS_NAMES
from transforms import get_transforms


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "batch_size"  : 32,
    "num_workers" : 4,
    "pin_memory"  : True,   # set False if running on CPU only
    "seed"        : 42,
}


# ---------------------------------------------------------------------------
# Weighted Sampler (class imbalance correction)
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples minority classes.

    Each sample is assigned a weight inversely proportional to its
    class frequency so that all classes are seen equally per epoch.

    Parameters
    ----------
    dataset : ChestXRayDataset

    Returns
    -------
    WeightedRandomSampler
    """
    labels = dataset.get_labels()
    class_counts = dataset.get_class_counts()

    # Weight per class = 1 / class_count
    class_weights = {
        cls: 1.0 / count
        for cls, count in class_counts.items()
        if count > 0
    }

    # Assign per-sample weights
    from dataset import IDX_TO_CLASS
    sample_weights = [
        class_weights[IDX_TO_CLASS[label]]
        for label in labels
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# Seed worker function (reproducibility)
# ---------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """
    Set per-worker random seeds for reproducible DataLoader output.

    Called automatically by DataLoader if worker_init_fn is set.
    """
    import numpy as np
    import random

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_root: Path | str,
    batch_size:   int  = DEFAULT_CONFIG["batch_size"],
    num_workers:  int  = DEFAULT_CONFIG["num_workers"],
    pin_memory:   bool = DEFAULT_CONFIG["pin_memory"],
    seed:         int  = DEFAULT_CONFIG["seed"],
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    dataset_root : Path | str
        Root directory containing train/, val/, test/ subdirectories.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Parallel data loading workers.  Set 0 for debugging.
    pin_memory : bool
        Pin CPU tensors to memory for faster GPU transfer.
    seed : int
        Random seed for reproducibility.
    use_weighted_sampler : bool
        If True, apply WeightedRandomSampler on training split to
        address class imbalance.  Val/test always use sequential sampling.

    Returns
    -------
    Dict[str, DataLoader]
        Keys: 'train', 'val', 'test'
    """
    dataset_root = Path(dataset_root)
    generator = torch.Generator().manual_seed(seed)

    dataloaders: Dict[str, DataLoader] = {}

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split

        if not split_dir.exists():
            print(f"[WARNING] Split directory not found: {split_dir}  — skipping.")
            continue

        dataset = ChestXRayDataset(
            root_dir=split_dir,
            transform=get_transforms(split),
        )

        # Sampler logic
        if split == "train" and use_weighted_sampler:
            sampler = build_weighted_sampler(dataset)
            shuffle = False          # shuffle is mutually exclusive with sampler
        else:
            sampler = None
            shuffle = False          # val/test must be deterministic

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=(split == "train"),   # drop incomplete last batch during training only
        )

        dataloaders[split] = loader
        print(f"[{split:>5}]  {len(dataset):>5} samples  |  "
              f"{len(loader):>4} batches  |  "
              f"sampler={'weighted' if split == 'train' and use_weighted_sampler else 'sequential'}")

    return dataloaders


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "COVID_19_dataset"

    print("Building DataLoaders...\n")
    loaders = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=32,
        num_workers=0,    # 0 for quick local test
        pin_memory=False,
    )

    print("\nBatch shape verification:")
    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        print(f"  [{split:>5}]  images={images.shape}  labels={labels.shape}  "
              f"dtype={images.dtype}")

    print("\n✅ DataLoaders validated.")