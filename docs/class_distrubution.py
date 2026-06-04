"""
class_distribution.py

Generates class distribution statistics and a publication-quality
visualization for the COVID-19 Chest X-Ray dataset.

Expected Dataset Structure:

ExCV/
├── COVID_19_dataset/
│   ├── train/
│   │   ├── COVID/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   ├── val/
│   │   ├── COVID/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   └── test/
│       ├── COVID/
│       ├── Normal/
│       └── Viral Pneumonia/
├── docs/
│   └── class_distribution.py
├── outputs/
└── src/

Output:
outputs/figures/class_distribution.png
outputs/class_distribution_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def count_images(dataset_path: Path) -> pd.DataFrame:
    """
    Count images for each class across train, val, and test splits.

    Parameters
    ----------
    dataset_path : Path
        Root dataset directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing counts per class and split.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_path}"
        )

    splits = ["train", "val", "test"]

    for split in splits:
        if not (dataset_path / split).exists():
            raise FileNotFoundError(
                f"Missing dataset split: {dataset_path / split}"
            )

    class_names = sorted(
        [
            folder.name
            for folder in (dataset_path / "train").iterdir()
            if folder.is_dir()
        ]
    )

    if not class_names:
        raise ValueError("No class folders found inside train/")

    records = []

    for class_name in class_names:
        split_counts: Dict[str, int] = {}

        for split in splits:
            class_dir = dataset_path / split / class_name

            count = sum(
                1
                for file in class_dir.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            split_counts[split] = count

        total = sum(split_counts.values())

        records.append(
            {
                "Class": class_name,
                "Train": split_counts["train"],
                "Validation": split_counts["val"],
                "Test": split_counts["test"],
                "Total": total,
            }
        )

    return pd.DataFrame(records)


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing class counts.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with percentages.
    """
    total_images = int(df["Total"].sum())

    if total_images == 0:
        raise ValueError("No images found in dataset.")

    df = df.copy()

    df["Percentage"] = (
        df["Total"] / total_images * 100
    ).round(2)

    return df


def plot_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate and save class distribution bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Statistics DataFrame.
    output_path : Path
        Save location.
    """
    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        df["Class"],
        df["Total"],
    )

    plt.title(
        "Class Distribution of Chest X-Ray Dataset",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    plt.xlabel(
        "Class Names",
        fontsize=12,
    )

    plt.ylabel(
        "Number of Images",
        fontsize=12,
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.6,
    )

    max_height = df["Total"].max()

    for bar in bars:
        height = int(bar.get_height())

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_height * 0.01),
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_results(
    df: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Save statistics CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Statistics DataFrame.
    output_directory : Path
        Output directory.
    """
    csv_path = output_directory / "class_distribution_summary.csv"

    df.to_csv(
        csv_path,
        index=False,
    )


def print_summary(df: pd.DataFrame) -> None:
    """
    Print dataset summary table.
    """
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION SUMMARY")
    print("=" * 50)

    summary = df[["Class", "Total"]].copy()
    summary.columns = ["Class", "Count"]

    print(summary.to_string(index=False))

    print("-" * 50)
    print(f"Total Images: {int(df['Total'].sum()):,}")
    print("=" * 50)

    print("\nClass Percentages:")

    for _, row in df.iterrows():
        print(
            f"{row['Class']:<20} "
            f"{row['Percentage']:>6.2f}%"
        )


def main() -> None:
    """
    Main execution function.
    """
    try:
        # Project Root (ExCV)
        project_root = Path(__file__).resolve().parent.parent

        # Dataset Location
        dataset_path = project_root / "COVID_19_dataset"

        # Output Directories
        output_dir = project_root / "outputs"
        figures_dir = output_dir / "figures"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure_path = (
            figures_dir / "class_distribution.png"
        )

        print("\nDataset Path:")
        print(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"\nDataset folder not found:\n{dataset_path}"
            )

        # Processing
        counts_df = count_images(dataset_path)

        stats_df = calculate_statistics(
            counts_df
        )

        print_summary(stats_df)

        plot_distribution(
            stats_df,
            figure_path,
        )

        save_results(
            stats_df,
            output_dir,
        )

        print(
            f"\n✓ Figure saved to:\n{figure_path}"
        )

        print(
            f"\n✓ CSV saved to:\n"
            f"{output_dir / 'class_distribution_summary.csv'}"
        )

    except Exception as error:
        print(f"\n[ERROR] {error}")
        raise


if __name__ == "__main__":
    main()