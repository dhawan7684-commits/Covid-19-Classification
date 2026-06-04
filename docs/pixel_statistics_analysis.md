# Pixel Intensity Statistics Analysis

## Objective
To analyze the global pixel distribution, establish exact normalization parameters for our PyTorch data loaders, and check for systemic class-wise brightness biases that could act as machine learning shortcuts.

---

## Global Normalization Values
To achieve stable gradients, faster loss convergence, and clean feature extraction, the custom values calculated below replace the standard ImageNet constants in our input transformations pipeline:

* **Global Channel Mean ($\mu$):** `[0.515781, 0.515781, 0.515781]`
* **Global Channel Standard Deviation ($\sigma$):** `[0.251694, 0.251694, 0.251694]`

*Note: The identical values across the R, G, and B configurations confirm that the underlying images are single-channel grayscale arrays replicated across a 3-channel matrix. This allows us to use standard pre-trained transfer learning networks seamlessly.*

---

## Visualizations & Diagnostic Reports

### 1. Global Pixel Intensity Distribution
The overall density profile of the dataset was mapped to evaluate contrast range and pixel distribution.

![Global Pixel Intensity Distribution](../outputs/figures/pixel_intensity_histogram.png)

* **Source Artifact:** `outputs/figures/pixel_intensity_histogram.png`
* **Analysis:** This histogram confirms that the pixel density curves for the Red, Green, and Blue channels overlap perfectly across all 256 gray levels. This visually proves identical channel replication. The smooth curve concentrated around the center ($0.515$) indicates healthy contrast balance with zero clipping artifacts at the pure black (0) or pure white (255) thresholds.

### 2. Class-Wise Brightness Bias Tracking (Shortcut Screen)
The mean pixel intensity of every image was calculated and grouped by its diagnostic class to identify any data collection discrepancies.

![Class-Wise Brightness Bias Boxplot](../outputs/figures/class_brightness_bias.png)

* **Source Artifact:** `outputs/figures/class_brightness_bias.png`
* **Statistical Breakdown:**
  | Class Name | Average Intensity Mean | Intensity Standard Deviation |
  | :--- | :--- | :--- |
  | **COVID** | 0.548539 | 0.100205 |
  | **Normal** | 0.507854 | 0.087883 |
  | **Viral Pneumonia** | 0.491653 | 0.074510 |

* **Analysis & Threat Assessment:** This boxplot visually highlights the slight structural delta among the cohorts. The `COVID` cohort features a shifted box profile centering at a higher mean ($\sim 0.548$) than `Viral Pneumonia` ($\sim 0.491$). This variation visually validates our risk profile: if unmitigated, our convolutional layers or self-attention blocks will focus on overall image luminosity instead of tracing actual ground-glass opacities or consolidations within the lung fields.

---

## Data Pipeline Mitigations

To neutralize this brightness shortcut and force both `EfficientNet-B0` and `Vision Transformer` to focus exclusively on structural pathology, we have integrated two specific safeguards into the **Phase 6 Data Pipeline**:

1. **Dynamic Random Brightness & Contrast Shifts:**
   We apply `Albumentations.ColorJitter(brightness=0.2, contrast=0.2, p=0.5)` during training. This randomizes illumination values across images dynamically, breaking the brightness-to-class correlation and forcing feature detection invariance.
   
2. **Exact Z-Score Normalization:**
   The transformations block maps features precisely to `mean=[0.515781, 0.515781, 0.515781]` and `std=[0.251694, 0.251694, 0.251694]`, centering your distributions for clean training optimization steps.