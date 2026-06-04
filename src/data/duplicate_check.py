"""
duplicate_check.py

Detect duplicate images within each dataset split using perceptual hashing.

Project Structure:

ExCV/
├── COVID_19_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   └── data/
│       └── duplicate_check.py
├── outputs/
│   └── reports/
└── docs/

Output:
outputs/reports/duplicate_report.txt
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

import imagehash
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


def get_image_paths(split_path: Path) -> List[Path]:
    """
    Recursively collect all supported image files.

    Parameters
    ----------
    split_path : Path
        Dataset split directory.

    Returns
    -------
    List[Path]
        List of image paths.
    """
    image_paths: List[Path] = []

    for file_path in split_path.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            image_paths.append(file_path)

    return image_paths


def compute_hash(image_path: Path) -> Optional[str]:
    """
    Compute perceptual hash for an image.

    Parameters
    ----------
    image_path : Path
        Image file path.

    Returns
    -------
    Optional[str]
        Hash string if successful, otherwise None.
    """
    try:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image))

    except Exception as error:
        print(
            f"[WARNING] Failed to hash image:\n"
            f"{image_path}\n"
            f"Reason: {type(error).__name__}: {error}\n"
        )
        return None


def find_duplicates(
    image_paths: List[Path],
) -> Dict[str, List[Path]]:
    """
    Find duplicate images based on perceptual hash.

    Parameters
    ----------
    image_paths : List[Path]
        List of image paths.

    Returns
    -------
    Dict[str, List[Path]]
        Duplicate groups keyed by hash.
    """
    hash_groups: DefaultDict[str, List[Path]]
    hash_groups = defaultdict(list)

    for image_path in image_paths:
        image_hash = compute_hash(image_path)

        if image_hash is None:
            continue

        hash_groups[image_hash].append(image_path)

    duplicates = {
        image_hash: paths
        for image_hash, paths in hash_groups.items()
        if len(paths) > 1
    }

    return duplicates


def generate_report(
    duplicate_results: Dict[str, Dict[str, List[Path]]],
    report_path: Path,
) -> None:
    """
    Generate duplicate image report.

    Parameters
    ----------
    duplicate_results : Dict
        Duplicate groups for each split.
    report_path : Path
        Report file path.
    """
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_groups = len(
        duplicate_results["train"]
    )

    val_groups = len(
        duplicate_results["val"]
    )

    test_groups = len(
        duplicate_results["test"]
    )

    total_groups = (
        train_groups
        + val_groups
        + test_groups
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("=" * 80 + "\n")
        report.write("DUPLICATE IMAGE REPORT\n")
        report.write("=" * 80 + "\n\n")

        split_titles = {
            "train": "TRAIN SPLIT",
            "val": "VALIDATION SPLIT",
            "test": "TEST SPLIT",
        }

        for split_name in [
            "train",
            "val",
            "test",
        ]:
            duplicates = duplicate_results[
                split_name
            ]

            report.write(
                f"{split_titles[split_name]}\n\n"
            )

            report.write(
                f"Duplicate Groups Found: "
                f"{len(duplicates)}\n\n"
            )

            if not duplicates:
                report.write(
                    "No duplicate groups found.\n\n"
                )

            for group_idx, (
                _,
                paths,
            ) in enumerate(
                duplicates.items(),
                start=1,
            ):
                report.write(
                    f"Group {group_idx}:\n"
                )

                for path in paths:
                    report.write(
                        f"{path}\n"
                    )

                report.write("\n")

            report.write(
                "-" * 40 + "\n\n"
            )

        report.write("=" * 80 + "\n")
        report.write("SUMMARY\n\n")

        report.write(
            f"Train Duplicates: "
            f"{train_groups}\n"
        )

        report.write(
            f"Validation Duplicates: "
            f"{val_groups}\n"
        )

        report.write(
            f"Test Duplicates: "
            f"{test_groups}\n\n"
        )

        report.write(
            f"Total Duplicate Groups: "
            f"{total_groups}\n"
        )

        report.write(
            "\n" + "=" * 80 + "\n"
        )


def print_summary(
    train_count: int,
    val_count: int,
    test_count: int,
    duplicate_groups: int,
    report_path: Path,
) -> None:
    """
    Print execution summary.

    Parameters
    ----------
    train_count : int
        Number of train images scanned.
    val_count : int
        Number of validation images scanned.
    test_count : int
        Number of test images scanned.
    duplicate_groups : int
        Total duplicate groups.
    report_path : Path
        Report path.
    """
    print("\n" + "=" * 80)
    print("DUPLICATE IMAGE CHECK")
    print("=" * 80)
    print()

    print(
        f"Train Images Scanned: "
        f"{train_count:,}"
    )

    print(
        f"Validation Images Scanned: "
        f"{val_count:,}"
    )

    print(
        f"Test Images Scanned: "
        f"{test_count:,}"
    )

    print()

    print(
        f"Duplicate Groups Found: "
        f"{duplicate_groups:,}"
    )

    print()

    print("Report Saved:")
    print(report_path)

    print("\n" + "=" * 80)


def main() -> None:
    """
    Main execution function.
    """
    try:
        # ExCV/
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
            / "duplicate_report.txt"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n"
                f"{dataset_path}"
            )

        split_names = [
            "train",
            "val",
            "test",
        ]

        duplicate_results: Dict[
            str,
            Dict[str, List[Path]]
        ] = {}

        image_counts: Dict[str, int] = {}

        for split_name in split_names:
            split_path = (
                dataset_path / split_name
            )

            if not split_path.exists():
                raise FileNotFoundError(
                    f"Missing split folder:\n"
                    f"{split_path}"
                )

            print(
                f"Processing {split_name} split..."
            )

            image_paths = get_image_paths(
                split_path
            )

            image_counts[
                split_name
            ] = len(image_paths)

            duplicate_results[
                split_name
            ] = find_duplicates(
                image_paths
            )

        generate_report(
            duplicate_results,
            report_path,
        )

        total_duplicate_groups = (
            len(
                duplicate_results["train"]
            )
            + len(
                duplicate_results["val"]
            )
            + len(
                duplicate_results["test"]
            )
        )

        print_summary(
            train_count=image_counts[
                "train"
            ],
            val_count=image_counts[
                "val"
            ],
            test_count=image_counts[
                "test"
            ],
            duplicate_groups=(
                total_duplicate_groups
            ),
            report_path=report_path,
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