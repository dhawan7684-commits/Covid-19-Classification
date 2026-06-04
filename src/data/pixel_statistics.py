"""
Pixel Statistics Analyzer for Chest X-Ray Datasets

This production-ready script calculates exact global normalization parameters (Mean/Std Dev)
using a memory-efficient online standard deviation algorithm and evaluates class-wise 
brightness bias to catch potential dataset shortcuts.

It automatically identifies the project root from subdirectories.

Author: Senior Computer Vision & Research Software Engineer
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_images(dataset_root: Path) -> Tuple[List[Path], Dict[str, List[Path]]]:
    """
    Scans the dataset directory for supported image formats and groups them by class.
    
    Args:
        dataset_root: Path to the root directory of the dataset.
        
    Returns:
        A tuple containing:
            - List of all valid image paths.
            - Dictionary mapping class names to their respective list of image paths.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_images: List[Path] = []
    class_images: Dict[str, List[Path]] = {}

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory not found at: {dataset_root}")

    # Walk through the dataset structure (train/val/test -> classes)
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            all_images.append(path)
            # The class name is the immediate parent directory
            class_name = path.parent.name
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(path)

    if not all_images:
        raise ValueError(f"No valid images found in {dataset_root} with extensions {supported_extensions}")

    return all_images, class_images


def analyze_pixel_stats(
    all_images: List[Path], 
    class_images: Dict[str, List[Path]]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]], np.ndarray]:
    """
    Iterates through images to compute global running statistics and track per-class brightness.
    Memory-friendly implementation avoiding loading all images into RAM.
    
    Returns:
        A tuple containing:
            - global_mean: np.ndarray of shape (3,)
            - global_std: np.ndarray of shape (3,)
            - class_intensities: Dict mapping class names to lists of per-image mean intensities.
            - global_histogram: np.ndarray of shape (256, 3) for 0-255 distribution tracking.
    """
    # Accumulators for exact mean and std calculation on [0.0, 1.0] scaled pixels
    total_pixels: int = 0
    running_sum = np.zeros(3, dtype=np.float64)
    running_squared_sum = np.zeros(3, dtype=np.float64)
    
    # Global histogram accumulator (0-255 values)
    global_histogram = np.zeros((256, 3), dtype=np.float64)
    
    # Tracker for per-class image mean intensities (scaled to [0.0, 1.0])
    class_intensities: Dict[str, List[float]] = {cls: [] for cls in class_images.keys()}

    print(f"Processing {len(all_images)} images for statistical analysis...")
    
    for idx, img_path in enumerate(all_images, start=1):
        # Read image using OpenCV (BGR)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        num_pixels_img = h * w
        total_pixels += num_pixels_img
        
        # 1. Update Global Histogram (0-255)
        for ch in range(3):
            hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256])
            global_histogram[:, ch] += hist.flatten()
            
        # 2. Compute Scaled Statistics [0.0, 1.0]
        img_scaled = img_rgb.astype(np.float64) / 255.0
        
        # Running sums per channel
        img_sum = np.sum(img_scaled, axis=(0, 1))
        img_sq_sum = np.sum(np.square(img_scaled), axis=(0, 1))
        
        running_sum += img_sum
        running_squared_sum += img_sq_sum
        
        # 3. Track Class Brightness Bias
        # Average intensity across all 3 channels for this specific image
        img_mean_intensity = np.mean(img_scaled)
        class_name = img_path.parent.name
        class_intensities[class_name].append(float(img_mean_intensity))
        
        if idx % 500 == 0 or idx == len(all_images):
            print(f" Progress: [{idx}/{len(all_images)}] images analyzed.")

    # Compute final global statistics
    global_mean = running_sum / total_pixels
    # Variance formula: E[X^2] - (E[X])^2
    global_var = (running_squared_sum / total_pixels) - np.square(global_mean)
    # Ensure no negative values due to floating-point inaccuracies
    global_std = np.sqrt(np.clip(global_var, a_min=0.0, a_max=None))

    return global_mean, global_std, class_intensities, global_histogram


def create_visualizations(
    output_dir: Path, 
    global_histogram: np.ndarray, 
    class_intensities: Dict[str, List[float]]
) -> None:
    """
    Generates and saves publication-quality plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot Global Pixel Intensity Histogram
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#E64B35", "#4DBBD5", "#00A087"] # Crisp RGB Palette
    channels = ["Red Channel", "Green Channel", "Blue Channel"]
    bins = np.arange(256)
    
    for ch in range(3):
        # Normalize histogram for visualization clarity
        ch_hist = global_histogram[:, ch]
        total_hist_sum = np.sum(ch_hist)
        pdf = ch_hist / (total_hist_sum if total_hist_sum > 0 else 1.0)
        ax.plot(bins, pdf, label=channels[ch], color=colors[ch], lw=2.0, alpha=0.8)
        ax.fill_between(bins, pdf, alpha=0.1, color=colors[ch])
        
    ax.set_title("Global Pixel Intensity Distribution (0-255)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Pixel Value", fontsize=12)
    ax.set_ylabel("Relative Frequency", fontsize=12)
    ax.set_xlim([0, 255])
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "pixel_intensity_histogram.png", bbox_inches="tight")
    plt.close()

    # 2. Plot Class Brightness Bias Boxplot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    classes = list(class_intensities.keys())
    data = [class_intensities[cls] for cls in classes]
    
    box_plot = ax.boxplot(
        data, 
        labels=classes, 
        patch_artist=True, 
        widths=0.6,
        medianprops=dict(color="#222222", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, markeredgecolor="none", alpha=0.5)
    )
    
    # Custom aesthetic coloring for boxplots
    box_colors = ["#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"]
    for patch, color in zip(box_plot["boxes"], box_colors * (len(classes) // len(box_colors) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")

    ax.set_title("Systemic Class Brightness Bias Assessment", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Target Class Name", fontsize=12)
    ax.set_ylabel("Mean Image Intensity [0.0 - 1.0]", fontsize=12)
    ax.set_ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(output_dir / "class_brightness_bias.png", bbox_inches="tight")
    plt.close()


def save_report(
    output_path: Path, 
    global_mean: np.ndarray, 
    global_std: np.ndarray, 
    class_intensities: Dict[str, List[float]]
) -> None:
    """
    Saves a structured analytical text report detailing global and class-wise parameters.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("        CHEST X-RAY DATASET PIXEL STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("## GLOBAL NORMALIZATION VALUES (Scaled [0.0, 1.0])\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean (R, G, B): [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]\n")
        f.write(f"Std  (R, G, B): [{global_std[0]:.6f}, {global_std[1]:.6f}, {global_std[2]:.6f}]\n\n")
        
        f.write("## PER-CLASS BRIGHTNESS BIAS METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class Name':<25} | {'Avg Intensity Mean':<20} | {'Intensity Std Dev':<18}\n")
        f.write("-" * 70 + "\n")
        
        for class_name, intensities in class_intensities.items():
            if intensities:
                avg_mean = np.mean(intensities)
                std_mean = np.std(intensities)
                f.write(f"{class_name:<25} | {avg_mean:<20.6f} | {std_mean:<18.6f}\n")
            else:
                f.write(f"{class_name:<25} | {'No Data':<20} | {'No Data':<18}\n")
        f.write("=" * 60 + "\n")


def main() -> None:
    # 1. Start from the directory where this script sits (src/data/)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    
    # 2. Back up dynamically until we find the parent directory holding 'COVID_19_dataset'
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "COVID_19_dataset").exists():
            project_root = parent
            break

    dataset_root = project_root / "COVID_19_dataset"
    output_figures_dir = project_root / "outputs" / "figures"
    output_report_path = project_root / "outputs" / "reports" / "pixel_stats_report.txt"

    print("=" * 70)
    print(f"Project Workspace Root: {project_root}")
    print(f"Dataset Target Location: {dataset_root}")
    print("=" * 70)

    try:
        # Step 1: Scan Directory Structure
        all_images, class_images = find_images(dataset_root)
        print(f"Total Unique Classes Found: {len(class_images)}")
        for cls, paths in class_images.items():
            print(f" - Class '{cls}': {len(paths)} images")
        print("-" * 70)

        # Step 2: Compute Statistics via Memory-Safe Accumulators
        global_mean, global_std, class_intensities, global_histogram = analyze_pixel_stats(
            all_images, class_images
        )

        # Step 3: Generate Figures
        print("Generating production-quality visualizations...")
        create_visualizations(output_figures_dir, global_histogram, class_intensities)
        print(f" Saved structural metrics plots to: {output_figures_dir}")

        # Step 4: Write Report Output
        save_report(output_report_path, global_mean, global_std, class_intensities)
        print(f" Saved comprehensive report summary to: {output_report_path}")

        # Step 5: Terminal Feedback Summary
        print("\n" + "=" * 50)
        print("        PIXEL ANALYSIS EXECUTION SUCCESSFUL")
        print("=" * 50)
        print("Calculated Global Normalization Parameters:")
        print(f"  - Mean [R, G, B]:  [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]")
        print(f"  - Std  [R, G, B]:  [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]")
        print("=" * 50)

    except Exception as e:
        print(f"\n[CRITICAL FAILURE]: {e}")


if __name__ == "__main__":
    main()