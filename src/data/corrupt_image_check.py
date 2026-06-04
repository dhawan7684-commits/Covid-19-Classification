"""
corrupt_image_check.py

Verify the integrity of all images in the Chest X-Ray dataset and
generate a report of corrupt or unreadable image files.

Project Structure:

ExCV/
├── COVID_19_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   └── data/
│       └── corrupt_image_check.py
├── outputs/
│   └── reports/
└── docs/

Output:
outputs/reports/corrupt_images.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, UnidentifiedImageError


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


def check_image(
    image_path: Path,
) -> Optional[Tuple[str, str, str]]:
    """
    Verify image integrity using Pillow.

    Parameters
    ----------
    image_path : Path
        Image file path.

    Returns
    -------
    Optional[Tuple[str, str, str]]
        None if valid, otherwise returns:
        (image_path, exception_type, exception_message)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        return None

    except UnidentifiedImageError as error:
        return (
            str(image_path),
            type(error).__name__,
            str(error),
        )

    except (OSError, IOError) as error:
        return (
            str(image_path),
            type(error).__name__,
            str(error),
        )

    except Exception as error:
        return (
            str(image_path),
            type(error).__name__,
            str(error),
        )


def save_report(
    corrupt_files: List[Tuple[str, str, str]],
    report_path: Path,
) -> None:
    """
    Save corrupt image report.

    Parameters
    ----------
    corrupt_files : List[Tuple[str, str, str]]
        Corrupt image records.
    report_path : Path
        Output report file.
    """
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("=" * 80 + "\n")
        file.write("CORRUPT IMAGE REPORT\n")
        file.write("=" * 80 + "\n\n")

        if not corrupt_files:
            file.write(
                "No corrupt images were found.\n"
            )
            return

        for index, (
            image_path,
            exception_type,
            exception_message,
        ) in enumerate(corrupt_files, start=1):
            file.write(f"[{index}]\n")
            file.write(
                f"Image Path     : {image_path}\n"
            )
            file.write(
                f"Exception Type : {exception_type}\n"
            )
            file.write(
                f"Message        : {exception_message}\n"
            )
            file.write("-" * 80 + "\n")


def print_summary(
    total_images: int,
    corrupt_images: int,
) -> None:
    """
    Print execution summary.

    Parameters
    ----------
    total_images : int
        Total images checked.
    corrupt_images : int
        Number of corrupt images.
    """
    print("\n" + "=" * 50)
    print("CORRUPT IMAGE CHECK SUMMARY")
    print("=" * 50)
    print(
        f"Total Images Checked : {total_images:,}"
    )
    print(
        f"Corrupt Images Found : {corrupt_images:,}"
    )
    print("=" * 50)


def main() -> None:
    """
    Main execution function.
    """
    try:
        # --------------------------------------------------
        # Project Root
        # Script Location:
        # ExCV/src/data/corrupt_image_check.py
        #
        # parents[0] -> data
        # parents[1] -> src
        # parents[2] -> ExCV
        # --------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]

        dataset_path = (
            project_root / "COVID_19_dataset"
        )

        report_path = (
            project_root
            / "outputs"
            / "reports"
            / "corrupt_images.txt"
        )

        print("\n" + "=" * 50)
        print("DATASET INTEGRITY CHECK")
        print("=" * 50)
        print(f"Project Root : {project_root}")
        print(f"Dataset Path : {dataset_path}")
        print("=" * 50)

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

        print(
            f"\nFound {len(image_paths):,} images."
        )
        print("Checking integrity...\n")

        corrupt_files: List[
            Tuple[str, str, str]
        ] = []

        for image_path in image_paths:
            result = check_image(
                image_path
            )

            if result is not None:
                corrupt_files.append(
                    result
                )

        save_report(
            corrupt_files,
            report_path,
        )

        print_summary(
            total_images=len(image_paths),
            corrupt_images=len(
                corrupt_files
            ),
        )

        print(
            f"\nReport saved to:\n"
            f"{report_path}"
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