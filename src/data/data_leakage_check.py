"""
data_leakage_check.py

Detect image leakage across dataset splits using perceptual hashing.

Project Structure:

ExCV/
├── COVID_19_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/
│   └── reports/
├── src/
│   └── data/
│       └── data_leakage_check.py
└── docs/

Output:
outputs/reports/data_leakage_report.txt
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


def build_hash_index(
    split_name: str,
    split_path: Path,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Build hash index for a dataset split.

    Parameters
    ----------
    split_name : str
        Dataset split name.
    split_path : Path
        Split directory.

    Returns
    -------
    Dict[str, List[Dict[str, str]]]
        Hash index.
    """
    hash_index: DefaultDict[
        str,
        List[Dict[str, str]]
    ] = defaultdict(list)

    image_paths = get_image_paths(split_path)

    for image_path in image_paths:
        image_hash = compute_hash(image_path)

        if image_hash is None:
            continue

        try:
            class_name = image_path.parent.name

            hash_index[image_hash].append(
                {
                    "path": str(image_path),
                    "split": split_name,
                    "class": class_name,
                    "hash": image_hash,
                }
            )

        except Exception:
            continue

    return dict(hash_index)


def check_leakage(
    source_index: Dict[str, List[Dict[str, str]]],
    target_index: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    """
    Detect leakage between two dataset splits.

    Parameters
    ----------
    source_index : Dict
        Source split hash index.
    target_index : Dict
        Target split hash index.

    Returns
    -------
    List[Dict[str, object]]
        Leakage cases.
    """
    leakage_cases: List[
        Dict[str, object]
    ] = []

    common_hashes = (
        set(source_index.keys())
        & set(target_index.keys())
    )

    for image_hash in sorted(common_hashes):
        source_images = source_index[
            image_hash
        ]

        target_images = target_index[
            image_hash
        ]

        leakage_cases.append(
            {
                "hash": image_hash,
                "source": source_images,
                "target": target_images,
            }
        )

    return leakage_cases


def generate_report(
    train_val_leakage: List[Dict[str, object]],
    train_test_leakage: List[Dict[str, object]],
    val_test_leakage: List[Dict[str, object]],
    report_path: Path,
) -> None:
    """
    Generate leakage report.

    Parameters
    ----------
    train_val_leakage : List
        Train-validation leakage cases.
    train_test_leakage : List
        Train-test leakage cases.
    val_test_leakage : List
        Validation-test leakage cases.
    report_path : Path
        Output report path.
    """
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_cases = (
        len(train_val_leakage)
        + len(train_test_leakage)
        + len(val_test_leakage)
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("=" * 80 + "\n")
        report.write(
            "DATA LEAKAGE REPORT\n"
        )
        report.write("=" * 80 + "\n\n")

        sections = [
            (
                "TRAIN ↔ VALIDATION",
                train_val_leakage,
                "Train",
                "Validation",
            ),
            (
                "TRAIN ↔ TEST",
                train_test_leakage,
                "Train",
                "Test",
            ),
            (
                "VALIDATION ↔ TEST",
                val_test_leakage,
                "Validation",
                "Test",
            ),
        ]

        for (
            title,
            cases,
            left_label,
            right_label,
        ) in sections:

            report.write(
                f"{title}\n\n"
            )

            report.write(
                f"Leakage Cases Found: "
                f"{len(cases)}\n\n"
            )

            if not cases:
                report.write(
                    "No leakage detected.\n\n"
                )

            for idx, case in enumerate(
                cases,
                start=1,
            ):
                report.write(
                    f"Case {idx}:\n\n"
                )

                report.write(
                    f"Hash: {case['hash']}\n\n"
                )

                report.write(
                    f"{left_label}:\n"
                )

                for item in case["source"]:
                    report.write(
                        f"{item['path']}\n"
                    )

                report.write("\n")

                report.write(
                    f"{right_label}:\n"
                )

                for item in case["target"]:
                    report.write(
                        f"{item['path']}\n"
                    )

                report.write(
                    "\n---\n\n"
                )

        report.write(
            "SUMMARY\n\n"
        )

        report.write(
            f"Train ↔ Validation Leakage: "
            f"{len(train_val_leakage)}\n"
        )

        report.write(
            f"Train ↔ Test Leakage: "
            f"{len(train_test_leakage)}\n"
        )

        report.write(
            f"Validation ↔ Test Leakage: "
            f"{len(val_test_leakage)}\n\n"
        )

        report.write(
            f"Total Leakage Cases: "
            f"{total_cases}\n\n"
        )

        report.write(
            "=" * 80 + "\n"
        )


def print_summary(
    train_count: int,
    val_count: int,
    test_count: int,
    train_val_count: int,
    train_test_count: int,
    val_test_count: int,
    report_path: Path,
) -> None:
    """
    Print execution summary.

    Parameters
    ----------
    train_count : int
        Train images scanned.
    val_count : int
        Validation images scanned.
    test_count : int
        Test images scanned.
    train_val_count : int
        Train-validation leakage cases.
    train_test_count : int
        Train-test leakage cases.
    val_test_count : int
        Validation-test leakage cases.
    report_path : Path
        Report path.
    """
    total_cases = (
        train_val_count
        + train_test_count
        + val_test_count
    )

    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
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
        f"Train ↔ Validation Leakage: "
        f"{train_val_count}"
    )

    print(
        f"Train ↔ Test Leakage: "
        f"{train_test_count}"
    )

    print(
        f"Validation ↔ Test Leakage: "
        f"{val_test_count}"
    )

    print()

    print(
        f"Total Leakage Cases: "
        f"{total_cases}"
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
            / "data_leakage_report.txt"
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

        split_indices = {}

        image_counts = {}

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
                f"Building hash index for "
                f"{split_name}..."
            )

            image_paths = get_image_paths(
                split_path
            )

            image_counts[
                split_name
            ] = len(image_paths)

            split_indices[
                split_name
            ] = build_hash_index(
                split_name,
                split_path,
            )

        train_val_leakage = (
            check_leakage(
                split_indices["train"],
                split_indices["val"],
            )
        )

        train_test_leakage = (
            check_leakage(
                split_indices["train"],
                split_indices["test"],
            )
        )

        val_test_leakage = (
            check_leakage(
                split_indices["val"],
                split_indices["test"],
            )
        )

        generate_report(
            train_val_leakage,
            train_test_leakage,
            val_test_leakage,
            report_path,
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
            train_val_count=len(
                train_val_leakage
            ),
            train_test_count=len(
                train_test_leakage
            ),
            val_test_count=len(
                val_test_leakage
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