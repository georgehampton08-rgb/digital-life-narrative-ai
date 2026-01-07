"""Parser base infrastructure for Digital Life Narrative AI.

This module provides the abstract base class and registry system for all platform parsers.
Parsers extract media metadata from platform exports and normalize into Memory objects.

Classes:
    BaseParser: Abstract base class all parsers inherit from
    ParserRegistry: Singleton registry for parser discovery
    ParseStatus: Enum for parse operation status
    ParseResult: Complete result of parsing operation
    ParseProgress: Progress tracking for callbacks

Functions:
    parse_directory: High-level function to parse a directory
    register_parser: Decorator for registering parser classes

Example:
    >>> @register_parser
    ... class MyParser(BaseParser):
    ...     platform = SourcePlatform.INSTAGRAM
    ...     version = "1.0.0"
    ...
    ...     def can_parse(self, root: Path) -> bool:
    ...         return (root / "instagram.json").exists()
    ...
    ...     def parse(self, root: Path, progress=None) -> ParseResult:
    ...         # Parse logic here
    ...         pass
    >>>
    >>> results = parse_directory(Path("/exports"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Type

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

from src.core import (
    Memory,
    MediaType,
    SourcePlatform,
    Location,
    GeoPoint,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ParseStatus(str, Enum):
    """Status of a parse operation.

    Attributes:
        SUCCESS: Completed without errors
        PARTIAL: Completed with some warnings/skipped files
        FAILED: Could not complete parsing
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParseWarning:
    """A non-fatal issue encountered during parsing.

    Attributes:
        file_path: File that caused the warning (None if general)
        message: Human-readable warning message
        warning_type: Category (e.g., "missing_metadata", "unreadable_file")
        recoverable: Whether this issue can be ignored
    """

    file_path: Path | None
    message: str
    warning_type: str
    recoverable: bool = True


@dataclass
class ParseError:
    """A fatal issue that prevented parsing a file.

    Attributes:
        file_path: File that caused the error (None if general)
        message: Human-readable error message
        error_type: Category of error
        original_exception: Original exception if available
    """

    file_path: Path | None
    message: str
    error_type: str
    original_exception: Exception | None = None


@dataclass
class ParseProgress:
    """Progress information for callbacks.

    Attributes:
        current: Files processed so far
        total: Estimated total files
        current_file: Name of file being processed
        stage: Current stage (e.g., "scanning", "parsing")
    """

    current: int
    total: int
    current_file: str | None = None
    stage: str = "parsing"

    def percentage(self) -> float:
        """Calculate percentage complete.

        Returns:
            Percentage from 0.0 to 100.0
        """
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0


@dataclass
class ParseResult:
    """The complete result of running a parser.

    Attributes:
        platform: Platform this parser handles
        status: Overall parse status
        memories: Successfully parsed memories
        warnings: Non-fatal issues encountered
        errors: Fatal issues encountered
        files_processed: Total files processed
        files_skipped: Files explicitly skipped
        parse_duration_seconds: Time taken to parse
        root_path: Root directory that was parsed
        parser_version: Version of the parser
    """

    platform: SourcePlatform
    status: ParseStatus
    memories: list[Memory] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)
    errors: list[ParseError] = field(default_factory=list)
    files_processed: int = 0
    files_skipped: int = 0
    parse_duration_seconds: float = 0.0
    root_path: Path = field(default_factory=lambda: Path("."))
    parser_version: str = "unknown"

    def success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Ratio of memories to total attempts (0.0 to 1.0)
        """
        total_attempts = len(self.memories) + len(self.errors)
        if total_attempts == 0:
            return 1.0
        return len(self.memories) / total_attempts

    def to_summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Parse Result for {self.platform.value}:",
            f"  Status: {self.status.value}",
            f"  Memories: {len(self.memories)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Errors: {len(self.errors)}",
            f"  Files Processed: {self.files_processed}",
            f"  Files Skipped: {self.files_skipped}",
            f"  Success Rate: {self.success_rate():.1%}",
            f"  Duration: {self.parse_duration_seconds:.2f}s",
        ]
        return "\n".join(lines)

    def has_critical_errors(self) -> bool:
        """Check if parse had critical errors.

        Returns:
            True if FAILED or error rate > 50%
        """
        if self.status == ParseStatus.FAILED:
            return True
        return self.success_rate() < 0.5


# =============================================================================
# Type Aliases
# =============================================================================


ProgressCallback = Callable[[ParseProgress], None]


# =============================================================================
# Exceptions
# =============================================================================


class ParserError(Exception):
    """Base exception for parser issues.

    Attributes:
        message: Error message
        parser: Parser name that raised the error
        path: File/directory that caused the issue
    """

    def __init__(
        self,
        message: str,
        parser: str | None = None,
        path: Path | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.parser = parser
        self.path = path


class ParserNotFoundError(ParserError):
    """No parser registered for requested platform."""

    pass


class ParseAbortedError(ParserError):
    """Parsing was aborted due to errors."""

    pass


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseParser(ABC):
    """Abstract base class for all platform parsers.

    Subclasses must:
    - Set class attributes: platform, version, supported_extensions, description
    - Implement: can_parse(), parse(), get_signature_files()

    Subclasses inherit helper methods for common operations like EXIF extraction,
    file hashing, datetime parsing, etc.

    Example:
        >>> class SnapchatParser(BaseParser):
        ...     platform = SourcePlatform.SNAPCHAT
        ...     version = "1.0.0"
        ...     supported_extensions = {".jpg", ".mp4"}
        ...     description = "Snapchat Memories export parser"
        ...
        ...     def can_parse(self, root: Path) -> bool:
        ...         return (root / "memories_history.json").exists()
        ...
        ...     def parse(self, root: Path, progress=None) -> ParseResult:
        ...         # Implementation here
        ...         pass
        ...
        ...     def get_signature_files(self) -> list[str]:
        ...         return ["memories_history.json"]
    """

    # Class attributes - must be set by subclass
    platform: ClassVar[SourcePlatform]
    version: ClassVar[str] = "1.0.0"
    supported_extensions: ClassVar[set[str]] = set()
    description: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize parser.

        Sets up logger with parser-specific name. No heavy work here.
        """
        parser_name = self.__class__.__name__
        self._logger = logging.getLogger(f"{__name__}.{parser_name}")

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def can_parse(self, root: Path) -> bool:
        """Check if this parser can handle the given directory.

        Should be FAST - no deep scanning. Look for signature files/folders.

        Args:
            root: Directory to check

        Returns:
            True if this looks like our export format
        """
        pass

    @abstractmethod
    def parse(
        self,
        root: Path,
        progress: ProgressCallback | None = None,
    ) -> ParseResult:
        """Parse directory and extract memories.

        Must be fault-tolerant: one bad file should not abort parsing.
        Call progress callback periodically if provided.

        Args:
            root: Root directory to parse
            progress: Optional progress callback

        Returns:
            ParseResult with memories and status
        """
        pass

    @abstractmethod
    def get_signature_files(self) -> list[str]:
        """Get list of signature files/patterns for detection.

        Used by can_parse() and auto-detection.

        Returns:
            List of filenames or patterns (e.g., ["memories_history.json"])
        """
        pass

    # =========================================================================
    # EXIF Extraction Helpers
    # =========================================================================

    def _extract_exif_datetime(self, image_path: Path) -> datetime | None:
        """Extract DateTimeOriginal from image EXIF.

        Args:
            image_path: Path to image file

        Returns:
            Datetime from EXIF or None if not available
        """
        if not PILLOW_AVAILABLE:
            return None

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif:
                    return None

                # Try DateTimeOriginal (36867) first, then DateTime (306)
                for tag_id in [36867, 306]:
                    if tag_id in exif:
                        dt_str = exif[tag_id]
                        return self._parse_datetime_string(dt_str)

                return None
        except Exception as e:
            self._logger.debug(f"EXIF datetime extraction failed for {image_path.name}: {e}")
            return None

    def _extract_exif_location(self, image_path: Path) -> GeoPoint | None:
        """Extract GPS coordinates from EXIF.

        Args:
            image_path: Path to image file

        Returns:
            GeoPoint with coordinates or None if not available
        """
        if not PILLOW_AVAILABLE:
            return None

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif:
                    return None

                # GPS info is in tag 34853
                gps_info = exif.get(34853)
                if not gps_info:
                    return None

                # Extract latitude
                lat = gps_info.get(2)  # GPSLatitude
                lat_ref = gps_info.get(1)  # GPSLatitudeRef (N/S)

                # Extract longitude
                lon = gps_info.get(4)  # GPSLongitude
                lon_ref = gps_info.get(3)  # GPSLongitudeRef (E/W)

                if not all([lat, lat_ref, lon, lon_ref]):
                    return None

                # Convert DMS to decimal
                latitude = self._dms_to_decimal(lat, lat_ref)
                longitude = self._dms_to_decimal(lon, lon_ref)

                return GeoPoint(latitude=latitude, longitude=longitude)

        except Exception as e:
            self._logger.debug(f"EXIF location extraction failed for {image_path.name}: {e}")
            return None

    def _dms_to_decimal(self, dms: tuple, ref: str) -> float:
        """Convert DMS (degrees, minutes, seconds) to decimal.

        Args:
            dms: Tuple of (degrees, minutes, seconds)
            ref: Reference (N/S/E/W)

        Returns:
            Decimal degrees
        """
        degrees, minutes, seconds = dms
        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        if ref in ["S", "W"]:
            decimal = -decimal
        return decimal

    def _extract_exif_metadata(self, image_path: Path) -> dict[str, Any]:
        """Extract all available EXIF as dict.

        Args:
            image_path: Path to image file

        Returns:
            Dict of EXIF data, empty if unavailable
        """
        if not PILLOW_AVAILABLE:
            return {}

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif:
                    return {}

                metadata = {}
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = value

                return metadata

        except Exception as e:
            self._logger.debug(f"EXIF metadata extraction failed for {image_path.name}: {e}")
            return {}

    # =========================================================================
    # Datetime Parsing Helpers
    # =========================================================================

    def _parse_datetime_string(
        self,
        dt_string: str,
        formats: list[str] | None = None,
    ) -> datetime | None:
        """Parse datetime from string using multiple formats.

        Args:
            dt_string: Datetime string to parse
            formats: Custom formats to try (uses defaults if None)

        Returns:
            Parsed datetime or None
        """
        if not dt_string:
            return None

        # Default formats
        if formats is None:
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y:%m:%d %H:%M:%S",  # EXIF style
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%B %d, %Y at %I:%M%p",  # Snapchat style
                "%B %d, %Y at %I:%M %p",
                "%Y/%m/%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
            ]

        for fmt in formats:
            try:
                dt = datetime.strptime(dt_string, fmt)
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue

        return None

    def _parse_datetime_from_filename(self, filename: str) -> datetime | None:
        """Extract datetime from filename patterns.

        Handles common patterns:
        - IMG_20190615_143022.jpg
        - 2019-06-15_14-30-22.png
        - PXL_20210615_143022000.jpg
        - VID_20190615_143022.mp4
        - Screenshot_20190615-143022.png

        Args:
            filename: Filename to parse

        Returns:
            Parsed datetime or None
        """
        patterns = [
            # IMG_YYYYMMDD_HHMMSS
            r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})",
            # YYYY-MM-DD_HH-MM-SS
            r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})",
            # YYYYMMDD_HHMMSS
            r"(\d{8})[_-](\d{6})",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 6:
                        year, month, day, hour, minute, second = groups
                        return datetime(
                            int(year),
                            int(month),
                            int(day),
                            int(hour),
                            int(minute),
                            int(second),
                            tzinfo=timezone.utc,
                        )
                    elif len(groups) == 2:
                        # YYYYMMDD_HHMMSS format
                        date_part, time_part = groups
                        year = int(date_part[:4])
                        month = int(date_part[4:6])
                        day = int(date_part[6:8])
                        hour = int(time_part[:2])
                        minute = int(time_part[2:4])
                        second = int(time_part[4:6])
                        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                except (ValueError, IndexError):
                    continue

        return None

    # =========================================================================
    # File Hashing Helpers
    # =========================================================================

    def _compute_file_hash(
        self,
        file_path: Path,
        algorithm: str = "md5",
    ) -> str:
        """Compute hash of file contents.

        Reads in 64KB chunks to handle large files efficiently.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            Hex digest string
        """
        hash_obj = hashlib.new(algorithm)

        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):  # 64KB chunks
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self._logger.warning(f"Hash computation failed for {file_path}: {e}")
            return ""

    def _compute_quick_hash(self, file_path: Path) -> str:
        """Fast hash for large files.

        Hashes first 64KB + last 64KB + file size.
        Good for quick deduplication, not cryptographic.

        Args:
            file_path: Path to file

        Returns:
            Hex digest string
        """
        try:
            file_size = file_path.stat().st_size
            hash_obj = hashlib.md5()

            # Add file size
            hash_obj.update(str(file_size).encode())

            with open(file_path, "rb") as f:
                # First 64KB
                chunk = f.read(65536)
                hash_obj.update(chunk)

                # Last 64KB if file is large enough
                if file_size > 131072:  # 128KB
                    f.seek(-65536, 2)  # Seek from end
                    chunk = f.read(65536)
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()
        except Exception as e:
            self._logger.warning(f"Quick hash failed for {file_path}: {e}")
            return ""

    # =========================================================================
    # File Type Detection
    # =========================================================================

    def _is_media_file(self, path: Path) -> bool:
        """Check if file extension suggests media.

        Args:
            path: File path to check

        Returns:
            True if likely a media file
        """
        media_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".heic",
            ".webp",
            ".bmp",
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v",
            ".wmv",
            ".flv",
            ".mp3",
            ".m4a",
            ".wav",
            ".aac",
            ".flac",
            ".ogg",
        }
        return path.suffix.lower() in media_extensions

    def _is_metadata_file(self, path: Path) -> bool:
        """Check if file is metadata (not media).

        Args:
            path: File path to check

        Returns:
            True if likely a metadata file
        """
        metadata_extensions = {".json", ".xml", ".html", ".txt", ".csv"}
        return path.suffix.lower() in metadata_extensions

    def _should_skip_file(self, path: Path) -> bool:
        """Check if file should be skipped.

        Skips hidden files, system files, etc.

        Args:
            path: File path to check

        Returns:
            True if file should be skipped
        """
        # Hidden files
        if path.name.startswith("."):
            return True

        # System files
        skip_names = {
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            ".Trashes",
            ".Spotlight-V100",
            ".fseventsd",
        }
        if path.name in skip_names:
            return True

        # macOS resource forks
        if "__MACOSX" in path.parts:
            return True

        # Zero-byte files
        try:
            if path.stat().st_size == 0:
                return True
        except OSError:
            return True

        return False

    def _detect_media_type(self, path: Path) -> MediaType:
        """Determine MediaType from file extension.

        Args:
            path: File path

        Returns:
            Detected MediaType
        """
        ext = path.suffix.lower()

        photo_ext = {".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp", ".bmp"}
        video_ext = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv"}
        audio_ext = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}

        if ext in photo_ext:
            return MediaType.PHOTO
        elif ext in video_ext:
            return MediaType.VIDEO
        elif ext in audio_ext:
            return MediaType.AUDIO
        elif "screenshot" in path.name.lower():
            return MediaType.SCREENSHOT
        else:
            return MediaType.UNKNOWN

    # =========================================================================
    # JSON Handling
    # =========================================================================

    def _safe_json_load(self, path: Path) -> dict | list | None:
        """Safely load JSON file.

        Handles missing files, invalid JSON, encoding issues.
        Tries UTF-8, UTF-8-sig (BOM), then latin-1.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON or None on failure
        """
        if not path.exists():
            return None

        encodings = ["utf-8", "utf-8-sig", "latin-1"]

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as e:
                self._logger.warning(f"JSON load failed for {path}: {e}")
                return None

        self._logger.warning(f"Could not decode JSON file: {path}")
        return None

    # =========================================================================
    # Memory Creation
    # =========================================================================

    def _create_memory(self, **kwargs) -> Memory:
        """Factory helper to create Memory with common defaults.

        Automatically sets source_platform from self.platform.

        Args:
            **kwargs: Arguments to pass to Memory constructor

        Returns:
            Memory instance
        """
        kwargs.setdefault("source_platform", self.platform)
        return Memory(**kwargs)

    # =========================================================================
    # Directory Scanning
    # =========================================================================

    def _scan_directory(
        self,
        root: Path,
        recursive: bool = True,
    ) -> Iterator[Path]:
        """Yield all files in directory.

        Skips hidden files and system files. Yields files only, not directories.

        Args:
            root: Root directory to scan
            recursive: Whether to descend into subdirectories

        Yields:
            File paths
        """
        try:
            if recursive:
                for item in root.rglob("*"):
                    if item.is_file() and not self._should_skip_file(item):
                        yield item
            else:
                for item in root.iterdir():
                    if item.is_file() and not self._should_skip_file(item):
                        yield item
        except PermissionError as e:
            self._logger.warning(f"Permission denied scanning {root}: {e}")

    def _count_files(self, root: Path) -> int:
        """Count total files for progress estimation.

        Quick scan, skips hidden/system files.

        Args:
            root: Root directory

        Returns:
            Estimated file count
        """
        count = 0
        try:
            for _ in self._scan_directory(root):
                count += 1
        except Exception as e:
            self._logger.warning(f"File counting failed for {root}: {e}")
        return count


# =============================================================================
# Parser Registry
# =============================================================================


class ParserRegistry:
    """Singleton registry for parser discovery and lookup.

    Parsers register themselves using the @register decorator.

    Example:
        >>> @ParserRegistry.register
        ... class MyParser(BaseParser):
        ...     platform = SourcePlatform.INSTAGRAM
        ...     ...
        >>>
        >>> parser = ParserRegistry.get_parser(SourcePlatform.INSTAGRAM)
    """

    _parsers: ClassVar[dict[SourcePlatform, Type[BaseParser]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register(cls, parser_class: Type[BaseParser]) -> Type[BaseParser]:
        """Register a parser class.

        Can be used as decorator.

        Args:
            parser_class: Parser class to register

        Returns:
            The parser class (for decorator pattern)

        Raises:
            ValueError: If parser doesn't have required attributes
        """
        # Validate required class attributes
        if not hasattr(parser_class, "platform"):
            raise ValueError(f"{parser_class.__name__} must define 'platform' class attribute")

        if not hasattr(parser_class, "version"):
            raise ValueError(f"{parser_class.__name__} must define 'version' class attribute")

        # Register
        platform = parser_class.platform
        cls._parsers[platform] = parser_class
        logger.debug(f"Registered parser: {parser_class.__name__} for {platform.value}")

        return parser_class

    @classmethod
    def get_parser(cls, platform: SourcePlatform) -> BaseParser | None:
        """Get parser instance for a platform.

        Args:
            platform: Platform to get parser for

        Returns:
            Parser instance or None
        """
        parser_class = cls._parsers.get(platform)
        if parser_class:
            return parser_class()
        return None

    @classmethod
    def get_parser_class(cls, platform: SourcePlatform) -> Type[BaseParser] | None:
        """Get parser class for a platform.

        Args:
            platform: Platform to get parser class for

        Returns:
            Parser class or None
        """
        return cls._parsers.get(platform)

    @classmethod
    def list_parsers(cls) -> list[SourcePlatform]:
        """Return list of all registered platforms.

        Returns:
            List of SourcePlatform values
        """
        return list(cls._parsers.keys())

    @classmethod
    def detect_parsers(cls, root: Path) -> list[BaseParser]:
        """Auto-detect which parsers can handle a directory.

        Args:
            root: Directory to check

        Returns:
            List of parser instances that can handle this directory
        """
        matching_parsers = []

        for parser_class in cls._parsers.values():
            try:
                parser = parser_class()
                if parser.can_parse(root):
                    matching_parsers.append(parser)
            except Exception as e:
                logger.warning(f"Error checking {parser_class.__name__}.can_parse(): {e}")

        return matching_parsers

    @classmethod
    def get_all_parsers(cls) -> list[BaseParser]:
        """Return instances of all registered parsers.

        Returns:
            List of all parser instances
        """
        return [parser_class() for parser_class in cls._parsers.values()]

    @classmethod
    def clear(cls) -> None:
        """Clear registry (for testing)."""
        cls._parsers.clear()


# =============================================================================
# Module-Level Functions
# =============================================================================


def parse_directory(
    root: Path,
    platform: SourcePlatform | None = None,
    progress: ProgressCallback | None = None,
) -> list[ParseResult]:
    """Parse a directory with appropriate parser(s).

    If platform specified: use that parser only.
    If platform is None: auto-detect and run all matching parsers.

    Args:
        root: Directory to parse
        platform: Specific platform to use (None for auto-detect)
        progress: Optional progress callback

    Returns:
        List of ParseResult (one per parser that ran)

    Raises:
        ParserError: If directory doesn't exist
        ParserNotFoundError: If specified platform has no parser
    """
    if not root.exists():
        raise ParserError(
            f"Directory does not exist: {root}",
            path=root,
        )

    results = []

    if platform is not None:
        # Use specific parser
        parser = ParserRegistry.get_parser(platform)
        if parser is None:
            raise ParserNotFoundError(
                f"No parser registered for {platform.value}",
                parser=platform.value,
            )

        try:
            result = parser.parse(root, progress)
            results.append(result)
        except Exception as e:
            logger.error(f"Parser {parser.__class__.__name__} failed: {e}")
            # Return FAILED result instead of crashing
            results.append(
                ParseResult(
                    platform=platform,
                    status=ParseStatus.FAILED,
                    errors=[
                        ParseError(
                            file_path=root,
                            message=str(e),
                            error_type="parser_exception",
                            original_exception=e,
                        )
                    ],
                    root_path=root,
                    parser_version=parser.version,
                )
            )
    else:
        # Auto-detect
        parsers = ParserRegistry.detect_parsers(root)

        if not parsers:
            logger.warning(f"No parsers detected for {root}")
            return []

        for parser in parsers:
            try:
                result = parser.parse(root, progress)
                results.append(result)
            except Exception as e:
                logger.error(f"Parser {parser.__class__.__name__} failed: {e}")
                results.append(
                    ParseResult(
                        platform=parser.platform,
                        status=ParseStatus.FAILED,
                        errors=[
                            ParseError(
                                file_path=root,
                                message=str(e),
                                error_type="parser_exception",
                                original_exception=e,
                            )
                        ],
                        root_path=root,
                        parser_version=parser.version,
                    )
                )

    return results


def register_parser(parser_class: Type[BaseParser]) -> Type[BaseParser]:
    """Module-level decorator alias for ParserRegistry.register.

    Args:
        parser_class: Parser class to register

    Returns:
        The parser class (for decorator pattern)
    """
    return ParserRegistry.register(parser_class)
