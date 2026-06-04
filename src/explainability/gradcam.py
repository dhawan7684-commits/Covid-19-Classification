"""
Task 6 — Grad-CAM Explainability
Run: python src/explainability/gradcam.py
Produces heatmap overlays for sample images from each class.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image

from data.transforms import get_transforms
from models.model_factory import get_model, get_device

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = Path("outputs/checkpoints/best_model.pth")
DATASET_ROOT = Path("COVID_19_dataset/test")
CLASS_NAMES  = ["COVID", "Normal", "Viral Pneumonia"]
OUTPUT_DIR   = Path("outputs/figures/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_CLASS = 3

# ── Grad-CAM Implementation ───────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W]
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx, probs.squeeze().detach().cpu().numpy()


# ── Overlay Heatmap on Image ──────────────────────────────────────────────────
def overlay_heatmap(image_np, cam, alpha=0.45):
    """Blend Grad-CAM heatmap onto original image."""
    heatmap = cm.jet(cam)[:, :, :3]          # RGB heatmap
    image_f = image_np.astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_f
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ── Denormalize tensor → numpy image ─────────────────────────────────────────
def tensor_to_image(tensor):
    mean = np.array([0.515781, 0.515781, 0.515781])
    std  = np.array([0.251694, 0.251694, 0.251694])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ── Collect Sample Images ─────────────────────────────────────────────────────
def get_sample_paths(n_per_class=3):
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.exists():
            print(f"[WARNING] Folder not found: {cls_dir}")
            continue
        imgs = sorted(cls_dir.glob("*.jpg"))[:n_per_class]
        if not imgs:
            imgs = sorted(cls_dir.glob("*.png"))[:n_per_class]
        for p in imgs:
            samples.append((p, cls))
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device    = get_device()
    transform = get_transforms("test")

    print("=" * 70)
    print("GRAD-CAM EXPLAINABILITY")
    print("=" * 70)

    # Load model
    model = get_model("efficientnet", num_classes=3, pretrained=False)
    model.unfreeze_backbone()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[Checkpoint] Loaded epoch={ckpt.get('epoch',0)}  val_acc={ckpt.get('val_acc',0):.4f}")

    # Target layer — last conv block of EfficientNet-B0
    target_layer = model.backbone.features[-1]
    gradcam      = GradCAM(model, target_layer)

    # Collect samples
    samples = get_sample_paths(SAMPLES_PER_CLASS)
    print(f"\n[Samples] Found {len(samples)} images ({SAMPLES_PER_CLASS} per class)\n")

    # Per-class figure: original | heatmap | overlay
    for cls_name in CLASS_NAMES:
        cls_samples = [(p, c) for p, c in samples if c == cls_name]
        if not cls_samples:
            continue

        n     = len(cls_samples)
        fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Grad-CAM — {cls_name}", fontsize=15, fontweight='bold')

        for row, (img_path, _) in enumerate(cls_samples):
            # Load & transform
            pil_img = Image.open(img_path).convert("RGB")
            tensor  = transform(pil_img).to(device)

            # Grad-CAM
            cam, pred_idx, probs = gradcam.generate(tensor)
            pred_name = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            # Convert tensor → numpy image
            img_np  = tensor_to_image(tensor)
            overlay = overlay_heatmap(img_np, cam)

            # Plot
            axes[row][0].imshow(img_np)
            axes[row][0].set_title("Original", fontsize=10)
            axes[row][0].axis('off')

            axes[row][1].imshow(cam, cmap='jet')
            axes[row][1].set_title("Grad-CAM Heatmap", fontsize=10)
            axes[row][1].axis('off')

            axes[row][2].imshow(overlay)
            axes[row][2].set_title(
                f"Overlay\nPred: {pred_name} ({confidence:.1f}%)", fontsize=10
            )
            axes[row][2].axis('off')

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"gradcam_{cls_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {save_path}")

    # ── Combined summary grid ─────────────────────────────────────────────────
    print("\n[Grid] Generating combined summary grid...")
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Grad-CAM Summary — One Sample Per Class", fontsize=16, fontweight='bold')

    for row, cls_name in enumerate(CLASS_NAMES):
        cls_samples = [(p, c) for p, c in samples if c == cls_name]
        if not cls_samples:
            continue
        img_path, _ = cls_samples[0]
        pil_img  = Image.open(img_path).convert("RGB")
        tensor   = transform(pil_img).to(device)
        cam, pred_idx, probs = gradcam.generate(tensor)
        img_np   = tensor_to_image(tensor)
        overlay  = overlay_heatmap(img_np, cam)
        pred_name   = CLASS_NAMES[pred_idx]
        confidence  = probs[pred_idx] * 100

        axes[row][0].imshow(img_np)
        axes[row][0].set_ylabel(cls_name, fontsize=12, fontweight='bold', rotation=90, labelpad=10)
        axes[row][0].set_title("Original" if row == 0 else "", fontsize=11)
        axes[row][0].axis('off')

        axes[row][1].imshow(cam, cmap='jet')
        axes[row][1].set_title("Heatmap" if row == 0 else "", fontsize=11)
        axes[row][1].axis('off')

        axes[row][2].imshow(overlay)
        axes[row][2].set_title("Overlay" if row == 0 else "", fontsize=11)
        axes[row][2].set_xlabel(f"Pred: {pred_name} ({confidence:.1f}%)", fontsize=10)
        axes[row][2].axis('off')

    plt.tight_layout()
    grid_path = OUTPUT_DIR / "gradcam_summary_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {grid_path}")

    print("\n" + "=" * 70)
    print("✅  GRAD-CAM COMPLETE")
    print(f"    Figures → {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()