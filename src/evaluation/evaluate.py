"""
Task 5 — Evaluation Framework
Run: python src/evaluation/evaluate.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

from data.dataloader import build_dataloaders
from models.model_factory import get_model, get_device

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path("outputs/checkpoints/best_model.pth")
CLASS_NAMES = ["COVID", "Normal", "Viral Pneumonia"]
OUTPUT_DIR  = Path("outputs/reports")
FIGURES_DIR = Path("outputs/figures/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()
    )

# ── Confusion Matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Figure] Confusion matrix saved → {path}")

# ── ROC Curves ────────────────────────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs, path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls}  (AUC={auc:.4f})")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Figure] ROC curves saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print("=" * 70)
    print("EVALUATION — EfficientNet-B0 on Test Set")
    print("=" * 70)

    # Load model
    model = get_model("efficientnet", num_classes=3, pretrained=False)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch   = ckpt.get("epoch", 0)
    val_acc = ckpt.get("val_acc", 0.0)
    model.unfreeze_backbone()
    model = model.to(device)
    print(f"[Checkpoint] Loaded epoch={epoch}  val_acc={val_acc:.4f}")

    # DataLoader (test only)
    loaders = build_dataloaders(dataset_root="COVID_19_dataset")
    test_loader = loaders["test"]

    # Inference
    print("\nRunning inference on test set...")
    y_pred, y_true, y_probs = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred)
    prec_mac = precision_score(y_true, y_pred, average='macro')
    rec_mac  = recall_score(y_true, y_pred, average='macro')
    f1_mac   = f1_score(y_true, y_pred, average='macro')
    f1_wt    = f1_score(y_true, y_pred, average='weighted')

    from sklearn.preprocessing import label_binarize
    y_bin   = label_binarize(y_true, classes=[0,1,2])
    roc_auc = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')

    per_class_f1  = f1_score(y_true, y_pred, average=None)
    per_class_rec = recall_score(y_true, y_pred, average=None)
    per_class_pre = precision_score(y_true, y_pred, average=None)

    # ── Print Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro Precision   : {prec_mac:.4f}")
    print(f"  Macro Recall      : {rec_mac:.4f}")
    print(f"  Macro F1          : {f1_mac:.4f}")
    print(f"  Weighted F1       : {f1_wt:.4f}")
    print(f"  ROC-AUC (macro)   : {roc_auc:.4f}")
    print()
    print("  Per-Class Metrics:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*52}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<20} {per_class_pre[i]:>10.4f} {per_class_rec[i]:>10.4f} {per_class_f1[i]:>10.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Save text report ──────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("EfficientNet-B0 — Test Set Evaluation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint : {CHECKPOINT}\n")
        f.write(f"Epoch      : {epoch}\n")
        f.write(f"Val Acc    : {val_acc:.4f}\n\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"Macro Precision   : {prec_mac:.4f}\n")
        f.write(f"Macro Recall      : {rec_mac:.4f}\n")
        f.write(f"Macro F1          : {f1_mac:.4f}\n")
        f.write(f"Weighted F1       : {f1_wt:.4f}\n")
        f.write(f"ROC-AUC (macro)   : {roc_auc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"[Report] Saved → {report_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, y_probs, FIGURES_DIR / "roc_curves.png")

    print("\n" + "=" * 70)
    print("✅  EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()