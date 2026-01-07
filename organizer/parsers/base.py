"""Base parser interface and registry for Digital Life Narrative AI.

This module defines the abstract base class for all platform parsers and
provides a registry for dynamic parser lookup. Each platform-specific
parser normalizes data into MediaItem objects.

Parsing philosophy:
- Be lenient: if a file can't be parsed, log it and continue
- Preserve original metadata in MediaItem.original_metadata
- Prefer embedded metadata (EXIF) over filename-derived dates
- Generate unique IDs for all items
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from PIL import Image
from PIL.ExifTags import TAGS

from organizer.models import (
    Confidence,
    GeoLocation,
    MediaItem,
    ParseResult,
    SourcePlatform,
)

if TYPE_CHECKING:
    from organizer.config import AppConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ParserError(Exception):
    """Custom exception for parsing failures.

    Provides context about where and why parsing failed.

    Attributes:
        platform: The platform being parsed when error occurred.
        file_path: The specific file that caused the error (if applicable).
        message: Human-readable error description.
        original_exception: The underlying exception that was caught.
    """

    def __init__(
        self,
        message: str,
        platform: SourcePlatform,
        file_path: Path | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        """Initialize ParserError.

        Args:
            message: Human-readable error description.
            platform: The platform being parsed.
            file_path: Optional path to the file that caused the error.
            original_exception: Optional underlying exception.
        """
        self.platform = platform
        self.file_path = file_path
        self.message = message
        self.original_exception = original_exception

        # Build full message
        full_message = f"[{platform.value}] {message}"
        if file_path:
            full_message += f" (file: {file_path})"
        if original_exception:
            full_message += f" - {type(original_exception).__name__}: {original_exception}"

        super().__init__(full_message)


# =============================================================================
# Base Parser
# =============================================================================


class BaseParser(ABC):
    """Abstract base class for all platform parsers.

    Defines the interface that all platform-specific parsers must implement,
    and provides common utility methods for parsing operations.

    Class Attributes:
        platform: The SourcePlatform this parser handles (set by subclasses).
        supported_extensions: File extensions this parser can process.

    Example:
        ```python
        class SnapchatParser(BaseParser):
            platform = SourcePlatform.SNAPCHAT
            supported_extensions = {".jpg", ".mp4", ".json"}

            def parse(self, root_path: Path, progress_callback=None) -> ParseResult:
                # Implementation here
                pass
        ```
    """

    platform: SourcePlatform
    supported_extensions: set[str] = set()

    # Common datetime formats to try when parsing
    DATETIME_FORMATS: list[str] = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S",  # EXIF format
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y%m%d_%H%M%S",  # Filename format
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
    ]

    def __init__(self) -> None:
        """Initialize the parser."""
        self._errors: list[str] = []
        self._stats: dict[str, int] = {
            "total_files": 0,
            "parsed": 0,
            "skipped": 0,
            "errors": 0,
        }

    @abstractmethod
    def parse(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParseResult:
        """Parse all media items from the given root path.

        This is the main parsing method. Implementations should scan the
        root_path, extract all relevant media items, and normalize them
        into MediaItem objects.

        Args:
            root_path: Root directory of the export to parse.
            progress_callback: Optional callback for progress updates.
                Called with (current_count, estimated_total).

        Returns:
            ParseResult containing all parsed items, errors, and stats.
        """
        pass

    @abstractmethod
    def can_parse(self, root_path: Path) -> bool:
        """Check if this parser can handle the given path.

        Performs a quick validation to determine if the export at
        root_path is compatible with this parser.

        Args:
            root_path: Path to check.

        Returns:
            True if this parser can handle the export.
        """
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_item_id(self) -> str:
        """Generate a unique ID for a media item.

        Returns:
            New UUID string.
        """
        return str(uuid4())

    def _extract_exif_datetime(self, image_path: Path) -> datetime | None:
        """Extract datetime from image EXIF data.

        Attempts to read DateTimeOriginal, DateTimeDigitized, or DateTime
        from the image's EXIF metadata.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted datetime (timezone-aware UTC) or None if not found.
        """
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return None

                # Map tag IDs to names
                exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

                # Try different EXIF date fields in order of preference
                date_fields = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]

                for field in date_fields:
                    if field in exif:
                        dt_str = exif[field]
                        if isinstance(dt_str, str):
                            # EXIF format: "YYYY:MM:DD HH:MM:SS"
                            parsed = self._safe_parse_datetime(dt_str)
                            if parsed:
                                return parsed

        except Exception as e:
            logger.debug(f"Failed to extract EXIF from {image_path}: {e}")

        return None

    def _calculate_file_hash(
        self,
        file_path: Path,
        algorithm: str = "md5",
    ) -> str:
        """Calculate hash of a file for deduplication.

        Reads file in chunks to handle large files efficiently.

        Args:
            file_path: Path to the file to hash.
            algorithm: Hash algorithm to use (md5, sha256, etc.).

        Returns:
            Hexadecimal hash string.

        Raises:
            ParserError: If file cannot be read.
        """
        try:
            hasher = hashlib.new(algorithm)
            chunk_size = 65536  # 64KB chunks

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hasher.update(chunk)

            return hasher.hexdigest()

        except Exception as e:
            raise ParserError(
                f"Failed to calculate hash for {file_path.name}",
                platform=self.platform,
                file_path=file_path,
                original_exception=e,
            )

    def _safe_parse_datetime(
        self,
        dt_string: str,
        formats: list[str] | None = None,
    ) -> datetime | None:
        """Safely parse a datetime string trying multiple formats.

        Args:
            dt_string: The datetime string to parse.
            formats: Optional list of formats to try. Uses DATETIME_FORMATS if None.

        Returns:
            Parsed datetime (timezone-aware UTC) or None if parsing fails.
        """
        if not dt_string or not isinstance(dt_string, str):
            return None

        # Clean up the string
        dt_string = dt_string.strip()

        formats_to_try = formats or self.DATETIME_FORMATS

        for fmt in formats_to_try:
            try:
                parsed = datetime.strptime(dt_string, fmt)
                # Make timezone-aware (assume UTC if no timezone)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue

        # Try Unix timestamp (seconds or milliseconds)
        try:
            timestamp = float(dt_string)
            # Check if milliseconds (> year 3000 in seconds)
            if timestamp > 32503680000:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            pass

        logger.debug(f"Could not parse datetime: {dt_string}")
        return None

    def _normalize_location(
        self,
        raw_location: dict[str, Any] | str | None,
    ) -> GeoLocation | None:
        """Normalize various location formats into GeoLocation.

        Handles different location representations from various platforms.

        Args:
            raw_location: Raw location data (dict with lat/lon, string, etc.).

        Returns:
            Normalized GeoLocation or None if location can't be parsed.
        """
        if raw_location is None:
            return None

        if isinstance(raw_location, str):
            # Just a location string, no coordinates
            if raw_location.strip():
                return GeoLocation(raw_location_string=raw_location.strip())
            return None

        if isinstance(raw_location, dict):
            # Try to extract coordinates and place names
            lat = None
            lon = None
            place_name = None
            country = None

            # Common key variations for latitude
            lat_keys = ["latitude", "lat", "Latitude", "geoDataExif.latitude"]
            for key in lat_keys:
                if key in raw_location:
                    try:
                        lat = float(raw_location[key])
                        break
                    except (ValueError, TypeError):
                        pass

            # Common key variations for longitude
            lon_keys = ["longitude", "lon", "lng", "Longitude", "geoDataExif.longitude"]
            for key in lon_keys:
                if key in raw_location:
                    try:
                        lon = float(raw_location[key])
                        break
                    except (ValueError, TypeError):
                        pass

            # Place name keys
            place_keys = ["place_name", "placeName", "name", "locality", "city", "address"]
            for key in place_keys:
                if key in raw_location and raw_location[key]:
                    place_name = str(raw_location[key])
                    break

            # Country keys
            country_keys = ["country", "countryName", "country_code"]
            for key in country_keys:
                if key in raw_location and raw_location[key]:
                    country = str(raw_location[key])
                    break

            if lat is not None or lon is not None or place_name or country:
                return GeoLocation(
                    latitude=lat,
                    longitude=lon,
                    place_name=place_name,
                    country=country,
                    raw_location_string=str(raw_location),
                )

        return None

    def _extract_datetime_from_filename(self, filename: str) -> datetime | None:
        """Attempt to extract datetime from a filename.

        Looks for common date patterns in filenames.

        Args:
            filename: The filename to parse.

        Returns:
            Extracted datetime or None if no date found.
        """
        # Common filename date patterns
        patterns = [
            # IMG_20230615_143022.jpg
            r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
            # 2023-06-15_14-30-22.jpg
            r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})",
            # 2023-06-15 143022.jpg
            r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})(\d{2})(\d{2})",
            # IMG_20230615.jpg (date only)
            r"(\d{4})(\d{2})(\d{2})(?!\d)",
            # 2023-06-15.jpg (date only)
            r"(\d{4})-(\d{2})-(\d{2})(?!\d)",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    if len(groups) >= 6:
                        # Full datetime
                        dt = datetime(
                            year=int(groups[0]),
                            month=int(groups[1]),
                            day=int(groups[2]),
                            hour=int(groups[3]),
                            minute=int(groups[4]),
                            second=int(groups[5]),
                            tzinfo=timezone.utc,
                        )
                    else:
                        # Date only
                        dt = datetime(
                            year=int(groups[0]),
                            month=int(groups[1]),
                            day=int(groups[2]),
                            tzinfo=timezone.utc,
                        )
                    return dt
                except ValueError:
                    continue

        return None

    def _log_error(self, message: str, file_path: Path | None = None) -> None:
        """Log an error and add to error list.

        Args:
            message: Error message.
            file_path: Optional file path for context.
        """
        error_msg = message
        if file_path:
            error_msg = f"{file_path.name}: {message}"

        self._errors.append(error_msg)
        self._stats["errors"] += 1
        logger.warning(f"[{self.platform.value}] {error_msg}")

    def _create_parse_result(
        self,
        items: list[MediaItem],
        duration: float,
    ) -> ParseResult:
        """Create a ParseResult from parsed items.

        Args:
            items: List of parsed MediaItem objects.
            duration: Time taken to parse in seconds.

        Returns:
            Completed ParseResult.
        """
        self._stats["parsed"] = len(items)

        return ParseResult(
            source_platform=self.platform,
            items=items,
            parse_errors=self._errors.copy(),
            stats=self._stats.copy(),
            parse_duration_seconds=duration,
        )


# =============================================================================
# Parser Registry
# =============================================================================


class ParserRegistry:
    """Registry for parser classes.

    Provides dynamic lookup and registration of platform parsers.
    Use as a decorator or call register() directly.

    Example:
        ```python
        @ParserRegistry.register
        class SnapchatParser(BaseParser):
            platform = SourcePlatform.SNAPCHAT
            ...
        ```
    """

    _parsers: dict[SourcePlatform, type[BaseParser]] = {}

    @classmethod
    def register(
        cls,
        parser_class: type[BaseParser],
    ) -> type[BaseParser]:
        """Register a parser class.

        Can be used as a decorator or called directly.

        Args:
            parser_class: The parser class to register.

        Returns:
            The same parser class (for decorator use).

        Raises:
            ValueError: If parser has no platform set.
        """
        if not hasattr(parser_class, "platform") or parser_class.platform is None:
            raise ValueError(
                f"Parser class {parser_class.__name__} must define 'platform' attribute"
            )

        platform = parser_class.platform
        cls._parsers[platform] = parser_class
        logger.debug(f"Registered parser for {platform.value}: {parser_class.__name__}")

        return parser_class

    @classmethod
    def get_parser(cls, platform: SourcePlatform) -> BaseParser | None:
        """Get an instance of the parser for a platform.

        Args:
            platform: The source platform.

        Returns:
            Parser instance or None if no parser registered.
        """
        parser_class = cls._parsers.get(platform)
        if parser_class:
            return parser_class()
        return None

    @classmethod
    def get_all_parsers(cls) -> list[BaseParser]:
        """Get instances of all registered parsers.

        Returns:
            List of parser instances.
        """
        return [parser_class() for parser_class in cls._parsers.values()]

    @classmethod
    def list_supported_platforms(cls) -> list[SourcePlatform]:
        """List all platforms with registered parsers.

        Returns:
            List of supported SourcePlatform values.
        """
        return list(cls._parsers.keys())

    @classmethod
    def is_registered(cls, platform: SourcePlatform) -> bool:
        """Check if a parser is registered for a platform.

        Args:
            platform: The platform to check.

        Returns:
            True if a parser is registered.
        """
        return platform in cls._parsers


# =============================================================================
# High-Level Parsing Functions
# =============================================================================


def parse_all_sources(
    paths: list[Path],
    config: AppConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[ParseResult]:
    """Parse all sources from multiple input paths.

    Auto-detects platforms for each path and runs appropriate parsers.
    Handles errors gracefully — one bad file won't stop everything.

    Args:
        paths: List of paths to parse.
        config: Optional application configuration.
        progress_callback: Optional callback for progress.
            Called with (platform_name, current, total).

    Returns:
        List of ParseResult for all successfully parsed sources.
    """
    from organizer.detection import detect_export_source

    results: list[ParseResult] = []

    for path in paths:
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            continue

        if not path.is_dir():
            logger.warning(f"Path is not a directory: {path}")
            continue

        # Detect sources in this path
        try:
            detections = detect_export_source(path)
        except Exception as e:
            logger.error(f"Detection failed for {path}: {e}")
            continue

        if not detections:
            logger.info(f"No recognized export format in: {path}")
            continue

        # Parse each detected source
        for detection in detections:
            parser = ParserRegistry.get_parser(detection.platform)

            if not parser:
                logger.warning(f"No parser available for {detection.platform.value}")
                continue

            logger.info(
                f"Parsing {detection.platform.value} export "
                f"({detection.confidence.value} confidence) from {detection.root_path}"
            )

            try:
                start_time = time.time()

                # Create progress wrapper if callback provided
                parser_progress = None
                if progress_callback:

                    def parser_progress(current: int, total: int) -> None:
                        progress_callback(detection.platform.value, current, total)

                result = parser.parse(detection.root_path, parser_progress)
                results.append(result)

                duration = time.time() - start_time
                logger.info(
                    f"Parsed {len(result.items)} items from "
                    f"{detection.platform.value} in {duration:.2f}s"
                )

            except ParserError as e:
                logger.error(f"Parser error: {e}")
                # Create empty result with error
                results.append(
                    ParseResult(
                        source_platform=detection.platform,
                        items=[],
                        parse_errors=[str(e)],
                        stats={"total_files": 0, "parsed": 0, "skipped": 0, "errors": 1},
                        parse_duration_seconds=0.0,
                    )
                )

            except Exception as e:
                logger.error(f"Unexpected error parsing {detection.platform.value}: {e}")
                results.append(
                    ParseResult(
                        source_platform=detection.platform,
                        items=[],
                        parse_errors=[f"Unexpected error: {e}"],
                        stats={"total_files": 0, "parsed": 0, "skipped": 0, "errors": 1},
                        parse_duration_seconds=0.0,
                    )
                )

    return results


def parse_single_path(
    path: Path,
    platform: SourcePlatform | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ParseResult | None:
    """Parse a single path, optionally with explicit platform.

    Convenience function for parsing a single export.

    Args:
        path: Path to the export directory.
        platform: Optional explicit platform (auto-detected if None).
        progress_callback: Optional progress callback.

    Returns:
        ParseResult or None if parsing fails.
    """
    from organizer.detection import detect_export_source

    if not path.exists() or not path.is_dir():
        logger.error(f"Invalid path: {path}")
        return None

    # Determine platform
    if platform is None:
        detections = detect_export_source(path)
        if not detections:
            logger.error(f"Could not detect platform for: {path}")
            return None
        platform = detections[0].platform
        root_path = detections[0].root_path
    else:
        root_path = path

    # Get parser
    parser = ParserRegistry.get_parser(platform)
    if not parser:
        logger.error(f"No parser for platform: {platform.value}")
        return None

    try:
        return parser.parse(root_path, progress_callback)
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return None
