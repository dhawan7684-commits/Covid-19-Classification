"""
transforms.py
=============
Data augmentation and preprocessing transforms for chest X-ray classification.

Dataset-specific normalization values (computed from COVID_19_dataset):
    mean = [0.515781, 0.515781, 0.515781]
    std  = [0.251694, 0.251694, 0.251694]

Author: ExCV Project
"""

from torchvision import transforms

# ---------------------------------------------------------------------------
# Dataset-specific normalization constants
# ---------------------------------------------------------------------------
DATASET_MEAN = [0.515781, 0.515781, 0.515781]
DATASET_STD  = [0.251694, 0.251694, 0.251694]

IMG_SIZE = 224   # EfficientNet-B0 standard input size


# ---------------------------------------------------------------------------
# Transform Factories
# ---------------------------------------------------------------------------

def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with aggressive augmentation.

    Augmentation strategy:
        - Random resized crop   → simulates different zoom levels / patient positioning
        - Horizontal flip       → left-right chest symmetry
        - Rotation (±15°)       → slight acquisition angle variation
        - Color jitter          → scanner brightness / contrast variation
        - Random erasing        → simulates occlusion / artifacts
        - Normalize             → dataset-specific mean/std

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        # Spatial augmentations
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),          # over-size before crop
        transforms.RandomResizedCrop(
            size=IMG_SIZE,
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # Intensity augmentations
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.0,   # grayscale-converted → saturation irrelevant
            hue=0.0,
        ),

        # Convert to tensor [0, 1]
        transforms.ToTensor(),

        # Occlusion simulation (applied after ToTensor)
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value=0,
        ),

        # Dataset-specific normalization
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validation transforms — deterministic, no augmentation.

    Only operations:
        - Resize to IMG_SIZE × IMG_SIZE
        - ToTensor
        - Normalize

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Test transforms — identical to validation transforms.

    No augmentation; ensures reproducible evaluation metrics.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_inverse_normalize() -> transforms.Normalize:
    """
    Inverse normalization transform for visualization (e.g., Grad-CAM overlays).

    Converts normalized tensor back to [0, 1] range for display.

    Returns
    -------
    transforms.Normalize
    """
    inv_mean = [-m / s for m, s in zip(DATASET_MEAN, DATASET_STD)]
    inv_std  = [1.0 / s for s in DATASET_STD]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


# ---------------------------------------------------------------------------
# Transform Registry  (used by DataLoader factory)
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY = {
    "train": get_train_transforms,
    "val":   get_val_transforms,
    "test":  get_test_transforms,
}


def get_transforms(split: str) -> transforms.Compose:
    """
    Retrieve transforms by split name.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.

    Returns
    -------
    transforms.Compose

    Raises
    ------
    ValueError
        If split is not recognized.
    """
    if split not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[split]()


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from PIL import Image
    import torch

    dummy = Image.new("RGB", (299, 299), color=128)

    for split in ["train", "val", "test"]:
        t = get_transforms(split)
        tensor = t(dummy)
        assert tensor.shape == torch.Size([3, IMG_SIZE, IMG_SIZE]), \
            f"Shape mismatch for split '{split}': {tensor.shape}"
        print(f"[{split:>5}]  shape={tensor.shape}  "
              f"min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")

    print("\n✅ All transforms validated.")