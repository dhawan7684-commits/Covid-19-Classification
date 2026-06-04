"""
image_dimension_analysis.py

Analyze image dimensions in a Chest X-Ray dataset and generate
statistics, reports, and visualizations.

Project Structure:

ExCV/
├── COVID_19_dataset/
├── src/
│   └── data/
│       └── image_dimension_analysis.py
├── outputs/
│   ├── reports/
│   └── figures/
└── docs/

Outputs:
outputs/reports/image_dimension_report.txt
outputs/figures/image_dimension_distribution.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_images(dataset_path: Path) -> List[Path]:
    """
    Recursively find all supported image files.

    Parameters
    ----------
    dataset_path : Path
        Dataset root directory.

    Returns
    -------
    List[Path]
        List of image paths.
    """
    image_paths: List[Path] = []

    for file_path in dataset_path.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            image_paths.append(file_path)

    return image_paths


def extract_dimensions(
    image_paths: List[Path],
) -> List[Tuple[int, int, float]]:
    """
    Extract image dimensions and aspect ratios.

    Parameters
    ----------
    image_paths : List[Path]
        Image paths.

    Returns
    -------
    List[Tuple[int, int, float]]
        (width, height, aspect_ratio)
    """
    dimensions: List[
        Tuple[int, int, float]
    ] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                width, height = image.size

                if height == 0:
                    continue

                aspect_ratio = (
                    width / height
                )

                dimensions.append(
                    (
                        width,
                        height,
                        aspect_ratio,
                    )
                )

        except Exception as error:
            print(
                f"[WARNING] Could not read:\n"
                f"{image_path}\n"
                f"Reason: {type(error).__name__}: "
                f"{error}\n"
            )

    return dimensions


def compute_statistics(
    dimensions: List[Tuple[int, int, float]],
) -> Dict[str, object]:
    """
    Compute image dimension statistics.

    Parameters
    ----------
    dimensions : List[Tuple[int, int, float]]
        Dimension data.

    Returns
    -------
    Dict[str, object]
        Statistics dictionary.
    """
    widths = [item[0] for item in dimensions]
    heights = [item[1] for item in dimensions]

    resolutions = [
        (item[0], item[1])
        for item in dimensions
    ]

    resolution_counter = Counter(
        resolutions
    )

    most_common_resolution = (
        resolution_counter.most_common(1)[0]
    )

    min_resolution = min(
        resolutions,
        key=lambda x: x[0] * x[1],
    )

    max_resolution = max(
        resolutions,
        key=lambda x: x[0] * x[1],
    )

    return {
        "total_images": len(dimensions),
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": mean(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": mean(heights),
        "min_resolution": min_resolution,
        "max_resolution": max_resolution,
        "most_common_resolution": (
            most_common_resolution[0]
        ),
        "most_common_count": (
            most_common_resolution[1]
        ),
        "unique_resolutions": len(
            resolution_counter
        ),
        "resolution_counter": (
            resolution_counter
        ),
    }


def create_visualization(
    resolution_counter: Counter,
    output_path: Path,
) -> None:
    """
    Create resolution frequency visualization.

    Parameters
    ----------
    resolution_counter : Counter
        Resolution frequencies.
    output_path : Path
        Figure output path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_resolutions = (
        resolution_counter.most_common(20)
    )

    labels = [
        f"{w}×{h}"
        for (w, h), _ in top_resolutions
    ]

    counts = [
        count
        for _, count in top_resolutions
    ]

    plt.figure(figsize=(14, 8))

    bars = plt.bar(
        range(len(labels)),
        counts,
    )

    plt.title(
        "Resolution Frequency Distribution",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel(
        "Resolution",
        fontsize=12,
    )

    plt.ylabel(
        "Number of Images",
        fontsize=12,
    )

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_report(
    statistics_data: Dict[str, object],
    report_path: Path,
) -> None:
    """
    Save image dimension report.

    Parameters
    ----------
    statistics_data : Dict
        Computed statistics.
    report_path : Path
        Output report path.
    """
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution_counter = (
        statistics_data[
            "resolution_counter"
        ]
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("=" * 80 + "\n")
        report.write(
            "IMAGE DIMENSION ANALYSIS REPORT\n"
        )
        report.write("=" * 80 + "\n\n")

        report.write(
            f"Total Images: "
            f"{statistics_data['total_images']:,}\n\n"
        )

        report.write(
            "WIDTH STATISTICS\n"
        )

        report.write(
            f"Minimum Width: "
            f"{statistics_data['min_width']}\n"
        )

        report.write(
            f"Maximum Width: "
            f"{statistics_data['max_width']}\n"
        )

        report.write(
            f"Average Width: "
            f"{statistics_data['avg_width']:.2f}\n\n"
        )

        report.write(
            "HEIGHT STATISTICS\n"
        )

        report.write(
            f"Minimum Height: "
            f"{statistics_data['min_height']}\n"
        )

        report.write(
            f"Maximum Height: "
            f"{statistics_data['max_height']}\n"
        )

        report.write(
            f"Average Height: "
            f"{statistics_data['avg_height']:.2f}\n\n"
        )

        min_w, min_h = (
            statistics_data[
                "min_resolution"
            ]
        )

        max_w, max_h = (
            statistics_data[
                "max_resolution"
            ]
        )

        common_w, common_h = (
            statistics_data[
                "most_common_resolution"
            ]
        )

        report.write(
            f"Minimum Resolution: "
            f"{min_w}x{min_h}\n"
        )

        report.write(
            f"Maximum Resolution: "
            f"{max_w}x{max_h}\n"
        )

        report.write(
            f"Most Common Resolution: "
            f"{common_w}x{common_h}\n"
        )

        report.write(
            f"Most Common Count: "
            f"{statistics_data['most_common_count']}\n"
        )

        report.write(
            f"Unique Resolutions: "
            f"{statistics_data['unique_resolutions']}\n\n"
        )

        report.write(
            "=" * 80 + "\n"
        )

        report.write(
            "RESOLUTION FREQUENCY TABLE\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        for (
            resolution,
            count,
        ) in resolution_counter.most_common():
            width, height = resolution

            report.write(
                f"{width}x{height} "
                f"-> {count} images\n"
            )


def print_summary(
    statistics_data: Dict[str, object],
) -> None:
    """
    Print terminal summary.

    Parameters
    ----------
    statistics_data : Dict
        Computed statistics.
    """
    min_w, min_h = (
        statistics_data[
            "min_resolution"
        ]
    )

    max_w, max_h = (
        statistics_data[
            "max_resolution"
        ]
    )

    common_w, common_h = (
        statistics_data[
            "most_common_resolution"
        ]
    )

    print("\n" + "=" * 80)
    print("IMAGE DIMENSION ANALYSIS")
    print("=" * 80)
    print()

    print(
        f"Total Images: "
        f"{statistics_data['total_images']:,}"
    )

    print()

    print(
        "Minimum Resolution:"
    )

    print(
        f"{min_w}x{min_h}"
    )

    print()

    print(
        "Maximum Resolution:"
    )

    print(
        f"{max_w}x{max_h}"
    )

    print()

    print(
        "Average Resolution:"
    )

    print(
        f"{statistics_data['avg_width']:.0f}"
        f"x"
        f"{statistics_data['avg_height']:.0f}"
    )

    print()

    print(
        "Most Common Resolution:"
    )

    print(
        f"{common_w}x{common_h}"
    )

    print()

    print(
        "Unique Resolutions:"
    )

    print(
        statistics_data[
            "unique_resolutions"
        ]
    )

    print("\n" + "=" * 80)


def main() -> None:
    """
    Main execution function.
    """
    try:
        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        dataset_path = (
            project_root
            / "COVID_19_dataset"
        )

        report_path = (
            project_root
            / "outputs"
            / "reports"
            / "image_dimension_report.txt"
        )

        figure_path = (
            project_root
            / "outputs"
            / "figures"
            / "image_dimension_distribution.png"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n"
                f"{dataset_path}"
            )

        image_paths = find_images(
            dataset_path
        )

        if not image_paths:
            raise ValueError(
                "No supported image files found."
            )

        dimensions = (
            extract_dimensions(
                image_paths
            )
        )

        if not dimensions:
            raise ValueError(
                "Could not extract dimensions from any image."
            )

        statistics_data = (
            compute_statistics(
                dimensions
            )
        )

        create_visualization(
            statistics_data[
                "resolution_counter"
            ],
            figure_path,
        )

        save_report(
            statistics_data,
            report_path,
        )

        print_summary(
            statistics_data
        )

        print(
            f"\nReport Saved:\n{report_path}"
        )

        print(
            f"\nFigure Saved:\n{figure_path}"
        )

    except Exception as error:
        print(
            f"\n[ERROR] "
            f"{type(error).__name__}: "
            f"{error}"
        )
        raise


if __name__ == "__main__":
    main()