"""
remove_data_leakage.py

Remove confirmed cross-split data leakage from the Chest X-Ray dataset.

Policy:
- Never delete train images.
- Keep validation images for validation-test leakage cases.
- Remove validation/test images according to confirmed leakage list.

Project Structure:

ExCV/
├── COVID_19_dataset/
├── outputs/
│   └── reports/
├── src/
│   └── data/
│       └── remove_data_leakage.py
└── docs/

Output:
outputs/reports/leakage_cleanup_log.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


def collect_removals(dataset_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Collect files to keep and remove based on confirmed leakage cases.

    Parameters
    ----------
    dataset_path : Path
        Dataset root directory.

    Returns
    -------
    Tuple[List[Path], List[Path]]
        (files_to_keep, files_to_remove)
    """
    keep_files: List[Path] = []
    remove_files_list: List[Path] = []

    # -----------------------------
    # TRAIN ↔ VALIDATION
    # Keep TRAIN, Remove VALIDATION
    # -----------------------------

    keep_files.extend(
        [
            dataset_path / "train" / "COVID" / "COVID-3374.png",
            dataset_path / "train" / "COVID" / "COVID-533.png",
            dataset_path / "train" / "COVID" / "COVID-442.png",
            dataset_path / "train" / "COVID" / "COVID-1123.png",
        ]
    )

    remove_files_list.extend(
        [
            dataset_path / "val" / "COVID" / "COVID-3375.png",
            dataset_path / "val" / "COVID" / "COVID-898.png",
            dataset_path / "val" / "COVID" / "COVID-1267.png",
            dataset_path / "val" / "COVID" / "COVID-1416.png",
        ]
    )

    # -----------------------------
    # TRAIN ↔ TEST
    # Keep TRAIN, Remove TEST
    # -----------------------------

    keep_files.extend(
        [
            dataset_path / "train" / "COVID" / "COVID-2715.png",
            dataset_path / "train" / "COVID" / "COVID-126.png",
            dataset_path / "train" / "COVID" / "COVID-766.png",
            dataset_path / "train" / "COVID" / "COVID-1188.png",
            dataset_path / "train" / "COVID" / "COVID-1431.png",
        ]
    )

    remove_files_list.extend(
        [
            dataset_path / "test" / "COVID" / "COVID-2716.png",
            dataset_path / "test" / "COVID" / "COVID-230.png",
            dataset_path / "test" / "COVID" / "COVID-114.png",
            dataset_path / "test" / "COVID" / "COVID-487.png",
            dataset_path / "test" / "COVID" / "COVID-315.png",
        ]
    )

    # -----------------------------
    # VALIDATION ↔ TEST
    # Keep VALIDATION, Remove TEST
    # -----------------------------

    keep_files.extend(
        [
            dataset_path / "val" / "COVID" / "COVID-2677.png",
            dataset_path / "val" / "COVID" / "COVID-517.png",
        ]
    )

    remove_files_list.extend(
        [
            dataset_path / "test" / "COVID" / "COVID-2678.png",
            dataset_path / "test" / "COVID" / "COVID-56.png",
        ]
    )

    return keep_files, remove_files_list


def confirm_removal(files_to_remove: List[Path]) -> bool:
    """
    Display files to be removed and request confirmation.

    Parameters
    ----------
    files_to_remove : List[Path]
        Files scheduled for deletion.

    Returns
    -------
    bool
        True if user confirms.
    """
    print("\n" + "=" * 80)
    print("FILES SCHEDULED FOR REMOVAL")
    print("=" * 80)

    for file_path in files_to_remove:
        print(file_path)

    print("\n" + "=" * 80)

    response = input(
        "Type YES to continue: "
    ).strip()

    return response == "YES"


def remove_files(
    files_to_remove: List[Path],
) -> Tuple[int, List[str]]:
    """
    Remove files safely.

    Parameters
    ----------
    files_to_remove : List[Path]
        Files to delete.

    Returns
    -------
    Tuple[int, List[str]]
        (removed_count, log_entries)
    """
    removed_count = 0
    log_entries: List[str] = []

    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_path.unlink()

                removed_count += 1

                log_entries.append(
                    f"REMOVED : {file_path}"
                )

            else:
                log_entries.append(
                    f"MISSING : {file_path}"
                )

        except Exception as error:
            log_entries.append(
                f"FAILED  : {file_path}"
            )

            log_entries.append(
                f"REASON  : "
                f"{type(error).__name__}: "
                f"{error}"
            )

    return removed_count, log_entries


def save_log(
    log_path: Path,
    keep_files: List[Path],
    files_to_remove: List[Path],
    log_entries: List[str],
) -> None:
    """
    Save cleanup log.

    Parameters
    ----------
    log_path : Path
        Output log path.
    keep_files : List[Path]
        Files retained.
    files_to_remove : List[Path]
        Files targeted for deletion.
    log_entries : List[str]
        Operation log entries.
    """
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log:

        log.write("=" * 80 + "\n")
        log.write(
            "DATA LEAKAGE CLEANUP LOG\n"
        )
        log.write("=" * 80 + "\n\n")

        log.write("FILES KEPT\n\n")

        for path in keep_files:
            log.write(f"{path}\n")

        log.write(
            "\n" + "-" * 80 + "\n\n"
        )

        log.write(
            "FILES SELECTED FOR REMOVAL\n\n"
        )

        for path in files_to_remove:
            log.write(f"{path}\n")

        log.write(
            "\n" + "-" * 80 + "\n\n"
        )

        log.write("DELETION RESULTS\n\n")

        for entry in log_entries:
            log.write(entry + "\n")

        log.write(
            "\n" + "=" * 80 + "\n"
        )


def print_summary(
    removed_count: int,
    kept_count: int,
    log_path: Path,
) -> None:
    """
    Print final summary.

    Parameters
    ----------
    removed_count : int
        Number of files removed.
    kept_count : int
        Number of files retained.
    log_path : Path
        Log file path.
    """
    print("\n" + "=" * 80)
    print("DATA LEAKAGE CLEANUP")
    print("=" * 80)
    print()

    print(
        f"Files Removed: "
        f"{removed_count}"
    )

    print(
        f"Files Kept: "
        f"{kept_count}"
    )

    print()

    print("Log Saved:")
    print(log_path)

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

        log_path = (
            project_root
            / "outputs"
            / "reports"
            / "leakage_cleanup_log.txt"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n"
                f"{dataset_path}"
            )

        keep_files, files_to_remove = (
            collect_removals(
                dataset_path
            )
        )

        if not confirm_removal(
            files_to_remove
        ):
            print(
                "\nOperation cancelled by user."
            )
            return

        removed_count, log_entries = (
            remove_files(
                files_to_remove
            )
        )

        save_log(
            log_path=log_path,
            keep_files=keep_files,
            files_to_remove=files_to_remove,
            log_entries=log_entries,
        )

        print_summary(
            removed_count=removed_count,
            kept_count=len(
                keep_files
            ),
            log_path=log_path,
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