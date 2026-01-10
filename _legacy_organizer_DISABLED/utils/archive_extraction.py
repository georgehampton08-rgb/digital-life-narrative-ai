"""Archive extraction utilities for Digital Life Narrative AI.

Handles automatic extraction of ZIP archives, including multi-part archives
from Google Takeout.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class ArchiveExtractionError(Exception):
    """Error occurred during archive extraction."""

    pass


def is_multi_part_archive(file_path: Path) -> bool:
    """Check if a file is part of a multi-part ZIP archive.

    Google Takeout uses naming like:
    - takeout-20260106T162534Z-9-001.zip
    - takeout-20260106T162534Z-9-002.zip

    Args:
        file_path: Path to check.

    Returns:
        True if file appears to be part of multi-part archive.
    """
    name = file_path.stem
    # Check for patterns like -001, -002 at the end
    return bool(name and (name.endswith("-001") or "-00" in name[-4:]))


def find_archive_parts(first_part: Path) -> list[Path]:
    """Find all parts of a multi-part archive.

    Args:
        first_part: Path to the first part (e.g., *-001.zip).

    Returns:
        List of all parts in order.
    """
    if not is_multi_part_archive(first_part):
        return [first_part]

    parent = first_part.parent
    base_name = first_part.stem[:-4]  # Remove -001

    parts = []
    part_num = 1
    while True:
        part_path = parent / f"{base_name}-{part_num:03d}.zip"
        if not part_path.exists():
            break
        parts.append(part_path)
        part_num += 1

    return parts if parts else [first_part]


def detect_archives_in_directory(directory: Path) -> list[Path]:
    """Find archive files in a directory.

    Detects:
    - Single ZIP files
    - First part of multi-part ZIPs (e.g., *-001.zip)

    Args:
        directory: Directory to search.

    Returns:
        List of archive files (first part only for multi-part).
    """
    if not directory.exists() or not directory.is_dir():
        return []

    archives = []
    seen_bases = set()

    for zip_file in sorted(directory.glob("*.zip")):
        # For multi-part, only include the first part
        if is_multi_part_archive(zip_file):
            base_name = zip_file.stem[:-4]
            if base_name not in seen_bases:
                archives.append(zip_file)
                seen_bases.add(base_name)
        else:
            archives.append(zip_file)

    return archives


def extract_archive(
    archive_path: Path,
    extract_to: Path | None = None,
    cleanup_on_error: bool = True,
) -> Path:
    """Extract a ZIP archive to a directory.

    Handles both single and multi-part archives automatically.

    Args:
        archive_path: Path to the archive or first part.
        extract_to: Directory to extract to. If None, creates temp directory.
        cleanup_on_error: If True, remove extract_to if extraction fails.

    Returns:
        Path to the extracted directory.

    Raises:
        ArchiveExtractionError: If extraction fails.
    """
    if not archive_path.exists():
        raise ArchiveExtractionError(f"Archive not found: {archive_path}")

    # Determine extraction directory
    if extract_to is None:
        extract_to = Path(tempfile.mkdtemp(prefix="organizer_extract_"))
        logger.info(f"Extracting to temporary directory: {extract_to}")
    else:
        extract_to.mkdir(parents=True, exist_ok=True)

    try:
        # Find all parts if multi-part
        parts = find_archive_parts(archive_path)

        if len(parts) > 1:
            logger.info(
                f"Detected multi-part archive: {len(parts)} parts "
                f"({archive_path.name} to {parts[-1].name})"
            )

        # Extract first part (or only part)
        logger.info(f"Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)

        # For multi-part, Python's zipfile should handle automatically if supported
        # Otherwise, we'd need to concatenate or use 7zip

        logger.info(f"Extraction complete: {extract_to}")
        return extract_to

    except zipfile.BadZipFile as e:
        error_msg = f"Invalid or corrupted ZIP file: {archive_path}"
        logger.error(f"{error_msg}: {e}")

        if cleanup_on_error and extract_to.exists():
            shutil.rmtree(extract_to, ignore_errors=True)

        raise ArchiveExtractionError(error_msg) from e

    except Exception as e:
        error_msg = f"Failed to extract {archive_path}"
        logger.error(f"{error_msg}: {e}")

        if cleanup_on_error and extract_to.exists():
            shutil.rmtree(extract_to, ignore_errors=True)

        raise ArchiveExtractionError(error_msg) from e


def extract_with_7zip(archive_path: Path, extract_to: Path) -> None:
    """Extract using 7-Zip command-line tool (if available).

    This is a fallback for multi-part archives that Python's zipfile can't handle.

    Args:
        archive_path: Path to first part of archive.
        extract_to: Directory to extract to.

    Raises:
        ArchiveExtractionError: If 7-Zip not available or extraction fails.
    """
    import subprocess

    # Try to find 7-Zip
    seven_zip_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z",  # Try PATH
    ]

    seven_zip = None
    for path in seven_zip_paths:
        try:
            result = subprocess.run(
                [path, "--help"], capture_output=True, timeout=5, check=False
            )
            if result.returncode == 0:
                seven_zip = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if not seven_zip:
        raise ArchiveExtractionError(
            "7-Zip not found. Multi-part archives require 7-Zip. "
            "Download from https://www.7-zip.org/"
        )

    # Run 7-Zip extraction
    try:
        logger.info(f"Extracting with 7-Zip: {archive_path}")
        result = subprocess.run(
            [seven_zip, "x", str(archive_path), f"-o{extract_to}", "-y"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            check=False,
        )

        if result.returncode != 0:
            raise ArchiveExtractionError(
                f"7-Zip extraction failed: {result.stderr or result.stdout}"
            )

        logger.info("7-Zip extraction complete")

    except subprocess.TimeoutExpired as e:
        raise ArchiveExtractionError("Extraction timed out (> 5 minutes)") from e
    except Exception as e:
        raise ArchiveExtractionError(f"7-Zip extraction error: {e}") from e
