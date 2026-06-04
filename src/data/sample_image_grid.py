"""
sample_image_grid.py

Generate a publication-quality image grid showing randomly sampled
chest X-ray images from each class for dataset inspection.

Expected Project Structure:

project/
├── COVID_19_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   └── data/
│       └── sample_image_grid.py
├── docs/
└── outputs/
    └── figures/

Output:
outputs/figures/sample_images.png
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 5


def discover_classes(dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Discover classes automatically from the train split.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset root.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping class names to image paths.

    Raises
    ------
    FileNotFoundError
        If dataset or train directory is missing.
    ValueError
        If no valid class folders are found.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_path}"
        )

    train_dir = dataset_path / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory not found:\n{train_dir}"
        )

    class_images: Dict[str, List[Path]] = {}

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        image_paths = [
            file
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_paths:
            print(
                f"[WARNING] Empty class folder skipped: "
                f"{class_dir.name}"
            )
            continue

        class_images[class_dir.name] = image_paths

    if not class_images:
        raise ValueError(
            "No valid class folders containing images were found."
        )

    return class_images


def sample_images(
    class_images: Dict[str, List[Path]],
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> Dict[str, List[Path]]:
    """
    Randomly sample images from each class.

    Parameters
    ----------
    class_images : Dict[str, List[Path]]
        Available image paths per class.
    samples_per_class : int
        Number of images to sample.

    Returns
    -------
    Dict[str, List[Path]]
        Sampled image paths per class.
    """
    random.seed(RANDOM_SEED)

    sampled: Dict[str, List[Path]] = {}

    for class_name, images in class_images.items():
        if len(images) >= samples_per_class:
            sampled[class_name] = random.sample(
                images,
                samples_per_class,
            )
        else:
            print(
                f"[WARNING] Class '{class_name}' contains only "
                f"{len(images)} images."
            )

            sampled[class_name] = random.sample(
                images,
                len(images),
            )

    return sampled


def create_grid(
    sampled_images: Dict[str, List[Path]],
) -> plt.Figure:
    """
    Create publication-quality image grid.

    Parameters
    ----------
    sampled_images : Dict[str, List[Path]]
        Sampled image paths.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    class_names = list(sampled_images.keys())

    rows = len(class_names)
    cols = SAMPLES_PER_CLASS

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 9),
        dpi=300,
    )

    if rows == 1:
        axes = [axes]

    fig.suptitle(
        "Sample Images from Chest X-Ray Dataset",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, class_name in enumerate(class_names):
        samples = sampled_images[class_name]

        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]

            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx < len(samples):
                image_path = samples[col_idx]

                try:
                    image = Image.open(image_path).convert("L")

                    ax.imshow(
                        image,
                        cmap="gray",
                    )

                except Exception as error:
                    print(
                        f"[WARNING] Could not read image:\n"
                        f"{image_path}\n"
                        f"Reason: {error}"
                    )

                    ax.text(
                        0.5,
                        0.5,
                        "Unreadable\nImage",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            ax.axis("off")

        axes[row_idx][0].set_ylabel(
            class_name,
            rotation=90,
            fontsize=12,
            fontweight="bold",
            labelpad=20,
        )

    plt.tight_layout(
        rect=[0.03, 0.03, 1, 0.95]
    )

    return fig


def save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> None:
    """
    Save generated figure.

    Parameters
    ----------
    figure : plt.Figure
        Figure object.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_summary(
    class_images: Dict[str, List[Path]],
) -> None:
    """
    Print dataset summary.

    Parameters
    ----------
    class_images : Dict[str, List[Path]]
        Available images per class.
    """
    print("\n" + "=" * 40)
    print("SAMPLE IMAGE INSPECTION")
    print("=" * 40)
    print()

    print(
        f"{'Class Name':<20}"
        f"{'Images Available':>15}"
    )
    print("-" * 35)

    for class_name, images in class_images.items():
        print(
            f"{class_name:<20}"
            f"{len(images):>15}"
        )

    print()
    print(
        f"Sampled Images Per Class: "
        f"{SAMPLES_PER_CLASS}"
    )

    print("\n" + "=" * 40)


def main() -> None:
    """
    Main execution function.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]

        dataset_path = (
            project_root / "COVID_19_dataset"
        )

        output_path = (
            project_root
            / "outputs"
            / "figures"
            / "sample_images.png"
        )

        print(
            f"\nDataset Path:\n{dataset_path}\n"
        )

        class_images = discover_classes(
            dataset_path
        )

        sampled_images = sample_images(
            class_images
        )

        print_summary(class_images)

        figure = create_grid(
            sampled_images
        )

        save_figure(
            figure,
            output_path,
        )

        print(
            f"\n✓ Figure saved successfully:\n"
            f"{output_path}"
        )

    except Exception as error:
        print(f"\n[ERROR] {error}")
        raise


if __name__ == "__main__":
    main()