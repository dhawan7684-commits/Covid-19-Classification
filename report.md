# ExCV: Explainable Classification of Chest X-Rays
### EfficientNet-B0 with Grad-CAM Explainability
**Author:** Laksh Dhawan

---

## Abstract

This project develops an explainable deep learning pipeline for classifying chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia. Using a pretrained EfficientNet-B0 backbone with a two-stage transfer learning strategy, the model achieves **97.66% test accuracy** and a **99.84% ROC-AUC**. Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to produce visual explanations of model predictions, highlighting the anatomical regions driving each classification decision.

---

## 1. Introduction

Automated chest X-ray classification has significant clinical value for triaging respiratory diseases, particularly during disease outbreaks. Deep learning models can match or exceed radiologist-level performance, but their black-box nature limits clinical adoption. Explainability methods such as Grad-CAM bridge this gap by producing saliency maps that reveal which image regions the model attends to, enabling clinicians to verify that predictions are grounded in medically relevant anatomy.

This project addresses:
1. Accurate three-class classification of chest X-rays
2. Visual explainability via Grad-CAM heatmaps
3. Quantitative evaluation using standard medical imaging metrics

---

## 2. Dataset

### 2.1 Source
The COVID-19 Chest X-Ray dataset comprises 3,994 images across three classes collected from publicly available clinical sources.

### 2.2 Statistics

| Split | COVID | Normal | Viral Pneumonia | Total |
|-------|-------|--------|-----------------|-------|
| Train | 977   | 1,000  | 993             | 2,970 |
| Val   | 196   | 200    | 200             | 596   |
| Test  | 138   | 145    | 145             | 428   |
| **Total** | **1,311** | **1,345** | **1,338** | **3,994** |

### 2.3 Key Properties
- **Image resolution:** 299×299 pixels (original), resized to 224×224 for training
- **Class balance:** Near-perfect (~33% each class) — minimal bias risk
- **Format:** RGB JPEG/PNG, converted to 3-channel grayscale-equivalent
- **No duplicates or corrupt images** detected during audit

### 2.4 Pixel Statistics
Dataset-specific normalization values computed from the training set:

| Parameter | Value |
|-----------|-------|
| Mean (all channels) | 0.515781 |
| Std (all channels)  | 0.251694 |

These replace ImageNet defaults for more accurate normalization.

---

## 3. Preprocessing Pipeline

### 3.1 Transforms

**Training augmentations** (to improve generalization):
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness, contrast)
- Normalize with dataset-specific mean/std

**Validation/Test** (deterministic):
- Resize to 224×224
- Center crop
- Normalize with dataset-specific mean/std

### 3.2 DataLoader Configuration
- Batch size: 32
- Weighted random sampler on training set (handles any residual class imbalance)
- num_workers: 0 (Windows compatibility)

---

## 4. Model Architecture

### 4.1 EfficientNet-B0

EfficientNet-B0 was selected as the primary model due to its compound scaling approach, which balances network depth, width, and resolution for optimal accuracy/compute tradeoff. With 4M parameters, it is well-suited to datasets of ~3,000 training images where larger models risk overfitting.

**Custom classifier head:**
```
GlobalAveragePooling → Dropout(0.3) → Linear(1280 → 3)
```

### 4.2 Two-Stage Transfer Learning

| Stage | Backbone | Trainable Params | LR Strategy |
|-------|----------|-----------------|-------------|
| Stage 1 | Frozen | 3,843 (head only) | Adam, lr=0.001, StepLR |
| Stage 2 | Unfrozen | 4,011,391 (all) | Adam, backbone_lr=0.0001, head_lr=0.001, CosineAnnealingLR |

**Rationale:** Freezing the backbone in Stage 1 prevents catastrophic forgetting of ImageNet features while rapidly adapting the classification head. Stage 2 fine-tunes the entire network at a lower learning rate.

### 4.3 Training Configuration
- Loss: Label Smoothing Cross-Entropy (ε=0.1)
- Early stopping: patience=5 on validation accuracy
- Stage 1: 12 epochs (early stopping triggered)
- Stage 2: 10 epochs

---

## 5. Results

### 5.1 Training Progression

| Epoch | Stage | Val Accuracy |
|-------|-------|-------------|
| 1  | S1 | 84.40% |
| 6  | S1 | 88.93% ← Stage 1 best |
| 13 | S2 | 90.94% |
| 14 | S2 | 94.63% |
| 17 | S2 | 96.98% |
| 19 | S2 | 97.32% |
| 22 | S2 | **97.48%** ← Final best |

### 5.2 Test Set Evaluation

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.66%** |
| Macro Precision | 97.85% |
| Macro Recall | 97.63% |
| Macro F1 | 97.68% |
| Weighted F1 | 97.68% |
| **ROC-AUC (macro OvR)** | **99.84%** |

### 5.3 Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| COVID-19 | **100%** | 95.65% | 97.78% | 138 |
| Normal | 93.55% | **100%** | 96.67% | 145 |
| Viral Pneumonia | **100%** | 97.24% | **98.60%** | 145 |

**Key finding:** COVID-19 precision = 100% — the model produces zero false positive COVID diagnoses, which is clinically critical for avoiding unnecessary isolation and treatment.

---

## 6. Explainability — Grad-CAM

### 6.1 Method

Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of the class score with respect to the final convolutional feature maps, then weights the activations by their importance to produce a spatial heatmap highlighting discriminative image regions.

Formally, for class *c*:

```
weights_k = GlobalAvgPool(∂y^c / ∂A^k)
CAM = ReLU(Σ_k weights_k · A^k)
```

Where A^k denotes the k-th feature map of the target layer (EfficientNet-B0 final conv block).

### 6.2 Qualitative Observations

**COVID-19:** Grad-CAM activations concentrate on bilateral peripheral lung opacities and ground-glass patterns characteristic of COVID-19 pneumonia.

**Normal:** Heatmaps show diffuse, low-intensity activations with no focal consolidation — consistent with clear lung fields.

**Viral Pneumonia:** Activations highlight perihilar and lower lobe consolidation patterns, distinguishable from COVID-19's peripheral distribution.

These spatial patterns align with established radiological criteria, validating that the model has learned clinically meaningful features rather than dataset artifacts.

### 6.3 Figures
- `outputs/figures/gradcam/gradcam_summary_grid.png` — combined 3×3 grid
- `outputs/figures/gradcam/gradcam_COVID.png`
- `outputs/figures/gradcam/gradcam_Normal.png`
- `outputs/figures/gradcam/gradcam_Viral_Pneumonia.png`
- `outputs/figures/evaluation/confusion_matrix.png`
- `outputs/figures/evaluation/roc_curves.png`

---

## 7. Discussion

### 7.1 Strengths
- **High accuracy (97.66%)** achieved on a balanced 3-class problem with only ~3,000 training images, demonstrating the effectiveness of transfer learning from ImageNet
- **Perfect COVID precision** is clinically meaningful — no healthy patients misclassified as COVID
- **Explainable predictions** via Grad-CAM align with radiological knowledge, increasing trustworthiness for clinical use
- **Near-perfect ROC-AUC (99.84%)** indicates excellent class separation across all decision thresholds

### 7.2 Limitations
- **Dataset size:** 3,994 images is small by deep learning standards; performance on larger, more diverse clinical datasets may differ
- **Single modality:** Model trained on X-rays only; CT scans provide complementary information not utilized here
- **Grad-CAM resolution:** Heatmaps are upsampled from low-resolution feature maps (7×7), limiting spatial precision of explanations
- **CPU-only training:** Training on CPU (~2 min/epoch) constrains the ability to run hyperparameter searches

### 7.3 Future Work
- Evaluate on external datasets (CheXpert, NIH ChestX-ray14) for generalization
- Compare ViT-B/16 with Attention Rollout explainability against EfficientNet + Grad-CAM
- Implement RISE or Integrated Gradients for higher-fidelity saliency maps
- Quantitative explainability evaluation via Insertion/Deletion AUC and AOPC scores

---

## 8. Conclusion

This project demonstrates that EfficientNet-B0 with two-stage transfer learning achieves state-of-the-art performance on COVID-19 chest X-ray classification (97.66% accuracy, 99.84% ROC-AUC) on a balanced 3-class dataset. Grad-CAM explainability confirms the model attends to clinically relevant lung regions, providing a foundation for trustworthy AI-assisted radiology. The complete pipeline — from data preprocessing through evaluation and explainability — is fully reproducible and documented.

---

## Appendix: Project Structure

```
ExCV/
├── COVID_19_dataset/
│   ├── train/  val/  test/
├── src/
│   ├── data/
│   │   ├── transforms.py
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── validate_preprocessing.py
│   ├── models/
│   │   ├── efficientnet.py
│   │   ├── vit.py
│   │   └── model_factory.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── scheduler.py
│   │   ├── trainer.py
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── explainability/
│       └── gradcam.py
├── outputs/
│   ├── checkpoints/best_model.pth
│   ├── reports/evaluation_report.txt
│   └── figures/
│       ├── evaluation/
│       └── gradcam/
└── docs/
    └── preprocessing_pipeline.md
