# Data Preprocessing Pipeline

## Overview

This document describes the complete data preprocessing pipeline for the COVID-19 chest X-ray classification task.

---

## Normalization Constants

Computed empirically from the full training dataset (Task 1.9 — Pixel Statistics Analysis):

| Channel | Mean     | Std Dev  |
|---------|----------|----------|
| R       | 0.515781 | 0.251694 |
| G       | 0.515781 | 0.251694 |
| B       | 0.515781 | 0.251694 |

> All three channels are identical because the original grayscale X-ray images are converted to RGB by replicating the single channel.  Using dataset-specific statistics rather than ImageNet defaults (mean=0.485, std=0.229) is more appropriate for medical imaging.

---

## Input Resolution

All images in the dataset are uniformly **299 × 299 pixels** (confirmed by Task 1.8).  
They are resized to **224 × 224** for EfficientNet-B0 compatibility.

---

## Transform Pipelines

### Training (`get_train_transforms`)

| Step | Operation | Parameters | Rationale |
|------|-----------|------------|-----------|
| 1 | Resize | 256 × 256 | Over-size before random crop |
| 2 | RandomResizedCrop | 224, scale=(0.80–1.00) | Simulates patient positioning variation |
| 3 | RandomHorizontalFlip | p=0.5 | Left-right chest symmetry |
| 4 | RandomRotation | ±15° | Acquisition angle variation |
| 5 | ColorJitter | brightness=0.2, contrast=0.2 | Scanner calibration variation |
| 6 | ToTensor | — | Convert PIL → float32 tensor [0,1] |
| 7 | RandomErasing | p=0.25, scale=(0.02–0.10) | Occlusion / artifact simulation |
| 8 | Normalize | mean=DATASET_MEAN, std=DATASET_STD | Zero-center activations |

### Validation / Test (`get_val_transforms`, `get_test_transforms`)

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Resize | 224 × 224 |
| 2 | ToTensor | — |
| 3 | Normalize | mean=DATASET_MEAN, std=DATASET_STD |

No augmentation is applied at validation or test time to ensure deterministic, reproducible metrics.

---

## Class Label Mapping

| Index | Class Name       |
|-------|-----------------|
| 0     | COVID19          |
| 1     | NORMAL           |
| 2     | ViralPneumonia   |

Alphabetically sorted for determinism.

---

## Class Imbalance Handling

A `WeightedRandomSampler` is applied during training.  Each sample receives a weight inversely proportional to its class frequency:

```
weight(sample) = 1 / count(class_of_sample)
```

This ensures all three classes are seen approximately equally per epoch without discarding any samples.

---

## DataLoader Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 32 | Fits comfortably on 8GB GPU |
| num_workers | 4 | Adjust to CPU core count |
| pin_memory | True | Faster GPU transfer |
| drop_last | True (train only) | Avoids unstable final batch |
| seed | 42 | Full reproducibility |

---

## File Structure

```
src/
└── data/
    ├── transforms.py           ← Transform pipelines + inverse normalize
    ├── dataset.py              ← ChestXRayDataset class
    ├── dataloader.py           ← DataLoader factory + WeightedRandomSampler
    └── validate_preprocessing.py  ← End-to-end validation + visual grid
```

---

## Validation

Run the validation script from the project root:

```bash
python src/data/validate_preprocessing.py
```

Expected output:
```
[1] Shape Check
    ✅  [train]  images=torch.Size([16, 3, 224, 224])
    ✅  [  val]  images=torch.Size([16, 3, 224, 224])
    ✅  [ test]  images=torch.Size([16, 3, 224, 224])

[2] Value Range Check  (expect roughly -3 to +3)
    ✅  [train]  min=-2.XXXX  max=2.XXXX  mean≈0

[3] Class Distribution
    ...

[4] Dtype Check  (expect torch.float32)
    ✅  All splits — torch.float32

[5] Sample grid saved → outputs/figures/preprocessing_sample_grid.png

✅  ALL CHECKS PASSED — preprocessing pipeline is ready.
```

---

## Progress

| Task | Status |
|------|--------|
| 2.1 Transforms Pipeline | ✅ Complete |
| 2.2 Custom Dataset Class | ✅ Complete |
| 2.3 DataLoader Factory | ✅ Complete |
| 2.4 Preprocessing Validation | ✅ Complete |
| 2.5 Documentation | ✅ Complete |