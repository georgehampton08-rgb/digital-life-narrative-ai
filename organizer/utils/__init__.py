"""Utility functions for Digital Life Narrative AI.

This package provides common utilities used throughout the application.

Submodules:
    logging: Centralized logging configuration
    hashing: File hashing and deduplication
    datetime_utils: Date/time parsing and formatting (planned)
    file_utils: File system operations (planned)
"""

from organizer.utils.archive_extraction import (
    ArchiveExtractionError,
    detect_archives_in_directory,
    extract_archive,
    is_multi_part_archive,
)
from organizer.utils.hashing import (
    compute_content_hash,
    compute_file_hash,
    compute_quick_hash,
    deduplicate_items,
    find_duplicates,
    find_duplicates_quick,
)
from organizer.utils.logging import (
    LogContext,
    add_file_handler,
    get_logger,
    log_context,
    set_level,
    setup_logging,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "log_context",
    "set_level",
    "add_file_handler",
    # Hashing
    "compute_file_hash",
    "compute_quick_hash",
    "compute_content_hash",
    "find_duplicates",
    "find_duplicates_quick",
    "deduplicate_items",
    # Archive extraction
    "extract_archive",
    "detect_archives_in_directory",
    "is_multi_part_archive",
    "ArchiveExtractionError",
]
