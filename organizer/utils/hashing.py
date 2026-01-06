"""File hashing utilities for Digital Life Narrative AI.

Provides efficient hash computation for file deduplication.

Example:
    >>> from organizer.utils.hashing import compute_file_hash, find_duplicates
    >>> hash_val = compute_file_hash(Path("photo.jpg"))
    >>> duplicates = find_duplicates(media_items)
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from organizer.models import MediaItem

logger = logging.getLogger(__name__)


# =============================================================================
# Hash Computation
# =============================================================================


def compute_file_hash(
    path: Path,
    algorithm: str = "md5",
    chunk_size: int = 8192,
) -> str:
    """Compute hash of file contents.

    Reads file in chunks to handle large files efficiently.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm (md5, sha1, sha256, etc.).
        chunk_size: Size of chunks to read (bytes).

    Returns:
        Hexadecimal hash string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If algorithm is not supported.

    Example:
        >>> hash_val = compute_file_hash(Path("photo.jpg"))
        >>> print(hash_val)  # "d41d8cd98f00b204e9800998ecf8427e"
    """
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_quick_hash(path: Path, sample_size: int = 65536) -> str:
    """Compute quick hash for large files.

    Hashes first + last sample_size bytes plus the file size.
    Much faster for large files but may have false positives.

    Args:
        path: Path to the file.
        sample_size: Bytes to read from start and end (default: 64KB).

    Returns:
        Hexadecimal hash string.

    Raises:
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> quick_hash = compute_quick_hash(Path("large_video.mp4"))
    """
    file_size = path.stat().st_size
    hasher = hashlib.md5()

    # Include file size in hash
    hasher.update(str(file_size).encode())

    with open(path, "rb") as f:
        # Read first chunk
        first_chunk = f.read(sample_size)
        hasher.update(first_chunk)

        # Read last chunk (if file is large enough)
        if file_size > sample_size * 2:
            f.seek(-sample_size, 2)  # Seek from end
            last_chunk = f.read(sample_size)
            hasher.update(last_chunk)
        elif file_size > sample_size:
            # File is between sample_size and sample_size*2
            f.seek(sample_size)
            remaining = f.read()
            hasher.update(remaining)

    return hasher.hexdigest()


def compute_content_hash(data: bytes, algorithm: str = "md5") -> str:
    """Compute hash of in-memory data.

    Args:
        data: Bytes to hash.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


# =============================================================================
# Deduplication
# =============================================================================


def find_duplicates(items: list["MediaItem"]) -> dict[str, list["MediaItem"]]:
    """Find duplicate media items by file hash.

    Groups items by their file hash and returns groups with more
    than one item (i.e., duplicates).

    Args:
        items: List of MediaItem objects.

    Returns:
        Dict mapping hash to list of duplicate MediaItems.
        Only includes hashes with 2+ items.

    Example:
        >>> duplicates = find_duplicates(media_items)
        >>> for hash_val, dupe_items in duplicates.items():
        ...     print(f"Hash {hash_val}: {len(dupe_items)} copies")
    """
    by_hash: dict[str, list["MediaItem"]] = defaultdict(list)

    for item in items:
        if item.file_hash:
            by_hash[item.file_hash].append(item)
        elif item.file_path and item.file_path.exists():
            try:
                file_hash = compute_file_hash(item.file_path)
                by_hash[file_hash].append(item)
            except Exception as e:
                logger.debug(f"Could not hash {item.file_path}: {e}")

    # Return only duplicates (2+ items with same hash)
    return {
        hash_val: items_list
        for hash_val, items_list in by_hash.items()
        if len(items_list) > 1
    }


def find_duplicates_quick(items: list["MediaItem"]) -> dict[str, list["MediaItem"]]:
    """Find potential duplicates using quick hash.

    Uses quick hash for speed, may have false positives.
    Good for initial screening of large collections.

    Args:
        items: List of MediaItem objects.

    Returns:
        Dict mapping hash to list of potentially duplicate MediaItems.
    """
    by_hash: dict[str, list["MediaItem"]] = defaultdict(list)

    for item in items:
        if item.file_path and item.file_path.exists():
            try:
                file_hash = compute_quick_hash(item.file_path)
                by_hash[file_hash].append(item)
            except Exception as e:
                logger.debug(f"Could not quick-hash {item.file_path}: {e}")

    return {
        hash_val: items_list
        for hash_val, items_list in by_hash.items()
        if len(items_list) > 1
    }


def deduplicate_items(items: list["MediaItem"]) -> tuple[list["MediaItem"], int]:
    """Remove duplicate items from a list.

    Keeps the first occurrence of each unique file.

    Args:
        items: List of MediaItem objects.

    Returns:
        Tuple of (deduplicated list, number of duplicates removed).

    Example:
        >>> unique_items, removed = deduplicate_items(all_items)
        >>> print(f"Removed {removed} duplicates")
    """
    seen_hashes: set[str] = set()
    unique_items: list["MediaItem"] = []
    duplicates_count = 0

    for item in items:
        # Get or compute hash
        file_hash = item.file_hash
        if not file_hash and item.file_path and item.file_path.exists():
            try:
                file_hash = compute_file_hash(item.file_path)
            except Exception:
                # Can't hash, include the item
                unique_items.append(item)
                continue

        if file_hash:
            if file_hash in seen_hashes:
                duplicates_count += 1
                logger.debug(f"Duplicate: {item.file_path}")
            else:
                seen_hashes.add(file_hash)
                unique_items.append(item)
        else:
            # No hash, include the item
            unique_items.append(item)

    return unique_items, duplicates_count
