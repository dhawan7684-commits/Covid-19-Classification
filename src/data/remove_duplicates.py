"""
remove_duplicates.py

Detect and remove duplicate images within each dataset split using
perceptual hashing (pHash).

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
│       └── remove_duplicates.py
└── docs/

Outputs:
outputs/reports/duplicate_groups_before_deletion.txt
outputs/reports/deleted_duplicates.txt
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

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
        Image hash or None if hashing fails.
    """
    try:
        with Image.open(image_path) as image:
            return str(imagehash.phash(image))

    except Exception as error:
        print(
            f"[WARNING] Failed to hash:\n"
            f"{image_path}\n"
            f"Reason: {type(error).__name__}: {error}\n"
        )
        return None


def find_duplicate_groups(
    image_paths: List[Path],
) -> Dict[str, List[Path]]:
    """
    Find duplicate image groups using perceptual hashing.

    Parameters
    ----------
    image_paths : List[Path]
        Image file paths.

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
        image_hash: sorted(paths)
        for image_hash, paths in hash_groups.items()
        if len(paths) > 1
    }

    return duplicates


def save_duplicate_report(
    duplicate_groups: Dict[str, Dict[str, List[Path]]],
    report_path: Path,
) -> None:
    """
    Save duplicate groups before deletion.

    Parameters
    ----------
    duplicate_groups : Dict
        Duplicate groups for all splits.
    report_path : Path
        Output report path.
    """
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("=" * 80 + "\n")
        report.write(
            "DUPLICATE GROUPS BEFORE DELETION\n"
        )
        report.write("=" * 80 + "\n\n")

        for split_name in [
            "train",
            "val",
            "test",
        ]:
            report.write(
                f"{split_name.upper()} SPLIT\n\n"
            )

            groups = duplicate_groups[
                split_name
            ]

            if not groups:
                report.write(
                    "No duplicate groups found.\n\n"
                )
                continue

            for idx, (
                _,
                paths,
            ) in enumerate(
                groups.items(),
                start=1,
            ):
                report.write(
                    f"Group {idx}\n"
                )

                for path in paths:
                    report.write(
                        f"{path}\n"
                    )

                report.write("\n")

            report.write(
                "-" * 80 + "\n\n"
            )


def confirm_deletion() -> bool:
    """
    Ask user confirmation before deletion.

    Returns
    -------
    bool
        True if confirmed.
    """
    print("\n" + "=" * 80)
    print("WARNING")
    print("=" * 80)
    print(
        "Duplicate images will be permanently deleted."
    )
    print(
        "A backup report has already been generated."
    )
    print()

    response = input(
        "Type YES to continue: "
    ).strip()

    return response == "YES"


def remove_duplicates(
    duplicate_groups: Dict[str, Dict[str, List[Path]]],
    deletion_log_path: Path,
) -> Tuple[int, int]:
    """
    Remove duplicate files while keeping the first
    file alphabetically in each group.

    Parameters
    ----------
    duplicate_groups : Dict
        Duplicate groups by split.
    deletion_log_path : Path
        Output deletion log.

    Returns
    -------
    Tuple[int, int]
        (files_deleted, files_retained)
    """
    deletion_log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_deleted = 0
    files_retained = 0

    with open(
        deletion_log_path,
        "w",
        encoding="utf-8",
    ) as log:

        log.write("=" * 80 + "\n")
        log.write(
            "DELETED DUPLICATE FILES\n"
        )
        log.write("=" * 80 + "\n\n")

        for split_name in [
            "train",
            "val",
            "test",
        ]:
            log.write(
                f"{split_name.upper()} SPLIT\n\n"
            )

            for _, paths in (
                duplicate_groups[
                    split_name
                ].items()
            ):
                sorted_paths = sorted(paths)

                keep_file = sorted_paths[0]
                delete_files = sorted_paths[1:]

                files_retained += 1

                log.write(
                    f"KEEP: {keep_file}\n"
                )

                for file_path in delete_files:
                    try:
                        file_path.unlink()

                        files_deleted += 1

                        log.write(
                            f"DELETE: {file_path}\n"
                        )

                    except Exception as error:
                        log.write(
                            f"FAILED: {file_path}\n"
                        )
                        log.write(
                            f"REASON: "
                            f"{type(error).__name__}: "
                            f"{error}\n"
                        )

                log.write("\n")

    return files_deleted, files_retained


def print_summary(
    duplicate_groups_found: int,
    files_deleted: int,
    files_retained: int,
    deletion_log_path: Path,
) -> None:
    """
    Print final summary.

    Parameters
    ----------
    duplicate_groups_found : int
        Number of duplicate groups.
    files_deleted : int
        Number of files deleted.
    files_retained : int
        Number of files retained.
    deletion_log_path : Path
        Deletion log path.
    """
    print("\n" + "=" * 80)
    print("DUPLICATE REMOVAL REPORT")
    print("=" * 80)
    print()

    print(
        f"Duplicate Groups Found: "
        f"{duplicate_groups_found:,}"
    )

    print()

    print(
        f"Files Deleted: "
        f"{files_deleted:,}"
    )

    print(
        f"Files Retained: "
        f"{files_retained:,}"
    )

    print()

    print("Deletion Log Saved:")
    print(deletion_log_path)

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

        reports_dir = (
            project_root
            / "outputs"
            / "reports"
        )

        duplicate_report_path = (
            reports_dir
            / "duplicate_groups_before_deletion.txt"
        )

        deletion_log_path = (
            reports_dir
            / "deleted_duplicates.txt"
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

        duplicate_groups: Dict[
            str,
            Dict[str, List[Path]]
        ] = {}

        total_groups = 0

        print("\nScanning dataset...\n")

        for split_name in split_names:
            split_path = (
                dataset_path / split_name
            )

            if not split_path.exists():
                raise FileNotFoundError(
                    f"Missing split folder:\n"
                    f"{split_path}"
                )

            image_paths = get_image_paths(
                split_path
            )

            split_duplicates = (
                find_duplicate_groups(
                    image_paths
                )
            )

            duplicate_groups[
                split_name
            ] = split_duplicates

            total_groups += len(
                split_duplicates
            )

            print(
                f"{split_name.upper():<10}"
                f"Duplicate Groups: "
                f"{len(split_duplicates)}"
            )

        save_duplicate_report(
            duplicate_groups,
            duplicate_report_path,
        )

        print(
            "\nBackup report created:"
        )
        print(
            duplicate_report_path
        )

        if total_groups == 0:
            print(
                "\nNo duplicate groups found."
            )
            return

        if not confirm_deletion():
            print(
                "\nOperation cancelled by user."
            )
            return

        files_deleted, files_retained = (
            remove_duplicates(
                duplicate_groups,
                deletion_log_path,
            )
        )

        print_summary(
            duplicate_groups_found=total_groups,
            files_deleted=files_deleted,
            files_retained=files_retained,
            deletion_log_path=deletion_log_path,
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