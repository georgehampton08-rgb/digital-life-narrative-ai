"""Local files parser for Digital Life Narrative AI.

This module provides a fallback parser for generic local photo/video directories.
It handles media from DSLRs, phone transfers, external drives, or any folder that
isn't from a specific platform export.

This is the "catch-all" parser that extracts metadata from:
- EXIF data in images
- Filename patterns (embedded dates)
- Folder structure hints
- File system timestamps (last resort)

Typical usage:
    >>> from pathlib import Path
    >>> from src.parsers.local_files import LocalFilesParser
    >>>
    >>> parser = LocalFilesParser()
    >>> if parser.can_parse(Path("/my_photos")):
    ...     result = parser.parse(Path("/my_photos"))
    ...     print(f"Parsed {len(result.memories)} memories")
"""

import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from PIL.ExifTags import TAGS

from src.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    SourcePlatform,
)
from src.parsers.base import (
    BaseParser,
    ParseError,
    ParseProgress,
    ParseResult,
    ParseStatus,
    ParseWarning,
    ProgressCallback,
    register_parser,
)

logger = logging.getLogger(__name__)

# Filename patterns for datetime extraction
# Format: (regex_pattern, format_description)
FILENAME_PATTERNS = [
    # Android standard: IMG_20190615_143022.jpg
    (r"IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", "YYYYMMDD_HHMMSS"),
    # Google Pixel: PXL_20190615_143022123.jpg
    (r"PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", "YYYYMMDD_HHMMSS"),
    # iOS Screenshot: Screenshot 2019-06-15 at 14.30.22.png
    (
        r"Screenshot[_ ](\d{4})-(\d{2})-(\d{2})[_ ]at[_ ](\d{1,2})\.(\d{2})\.(\d{2})",
        "YYYY-MM-DD_HH.MM.SS",
    ),
    # Generic: 2019-06-15 14-30-22.jpg
    (r"(\d{4})-(\d{2})-(\d{2})[_ ](\d{2})[-.](\d{2})[-.](\d{2})", "YYYY-MM-DD_HH-MM-SS"),
    # WhatsApp: IMG-20190615-WA0001.jpg
    (r"IMG-(\d{4})(\d{2})(\d{2})-WA\d+", "YYYYMMDD"),
    # Video: VID_20190615_143022.mp4
    (r"VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", "YYYYMMDD_HHMMSS"),
    # Date only: 20190615.jpg
    (r"(\d{4})(\d{2})(\d{2})", "YYYYMMDD"),
]


@register_parser
class LocalFilesParser(BaseParser):
    """Parser for local photo/video directories.

    This is a fallback parser that handles generic media directories without
    platform-specific structure. It extracts metadata from EXIF, filenames,
    folder structure, and file system timestamps.

    Attributes:
        platform: Source platform (LOCAL).
        version: Parser version.
        supported_extensions: File extensions this parser handles.
        description: Human-readable parser description.
    """

    platform: ClassVar[SourcePlatform] = SourcePlatform.LOCAL
    version: ClassVar[str] = "1.1.0"
    supported_extensions: ClassVar[set[str]] = {
        # Standard Images
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".heic",
        ".heif",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
        # RAW formats - Nikon
        ".nef",
        ".nrw",
        # RAW formats - Canon
        ".cr2",
        ".cr3",
        ".crw",
        # RAW formats - Sony
        ".arw",
        ".srf",
        ".sr2",
        # RAW formats - Fuji
        ".raf",
        # RAW formats - Olympus
        ".orf",
        # RAW formats - Panasonic
        ".rw2",
        # RAW formats - Pentax
        ".pef",
        ".ptx",
        # RAW formats - Generic/Adobe
        ".dng",
        ".raw",
        # Videos
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".3gp",
        ".wmv",
        ".flv",
        ".webm",
        # Audio
        ".mp3",
        ".m4a",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
    }
    description: ClassVar[str] = "Parser for local photo/video directories"

    def can_parse(self, root: Path) -> bool:
        """Check if this parser can handle the given directory.

        Args:
            root: Directory to check.

        Returns:
            True if directory contains at least 3 media files, False otherwise.

        Notes:
            This is a fallback parser with very loose requirements.
            It accepts any directory with media files.
        """
        if not root.is_dir():
            return False

        # Count media files
        media_count = 0
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                media_count += 1
                if media_count >= 3:  # At least 3 media files
                    return True

        return False

    def get_signature_files(self) -> list[str]:
        """Get list of signature files/directories.

        Returns:
            Empty list (this is a fallback parser with no signatures).
        """
        return []

    def parse(self, root: Path, progress: ProgressCallback | None = None) -> ParseResult:
        """Parse local media directory into Memory objects.

        Args:
            root: Root directory of local media.
            progress: Optional callback for progress reporting.

        Returns:
            ParseResult containing extracted memories, warnings, and errors.

        Processing steps:
            1. Scan for all media files
            2. For each file, extract EXIF, filename, and path metadata
            3. Create Memory objects with best available data
            4. Return comprehensive result
        """
        memories: list[Memory] = []
        warnings: list[ParseWarning] = []
        errors: list[ParseError] = []

        def report_progress(stage: str, current: int = 0, total: int = 0):
            """Helper to report progress if callback provided."""
            if progress:
                progress(ParseProgress(current=current, total=total, stage=stage))

        report_progress("Scanning local media directory...")
        logger.info(f"Starting local files parse of {root}")

        # Scan for media files
        media_files = list(self._scan_media_files(root))
        total_files = len(media_files)

        report_progress(f"Found {total_files} media files", 0, total_files)
        logger.info(f"Found {total_files} media files to process")

        # Process each file
        for idx, file_path in enumerate(media_files):
            try:
                memory = self._create_memory_from_file(file_path)
                if memory:
                    memories.append(memory)

                # Report progress periodically
                if idx % 100 == 0:
                    report_progress("Processing media files...", idx, total_files)

            except Exception as e:
                # Escape path for Rich markup (contains [...] that could be interpreted as tags)
                escaped_path = str(file_path).replace("[", "\\[").replace("]", "\\]")
                logger.warning(f"Error processing {escaped_path}: {e}", exc_info=True)
                errors.append(
                    ParseError(
                        file_path=file_path,
                        message=f"Failed to process file: {e}",
                        error_type="file_parse_error",
                        original_exception=e,
                    )
                )

        # Build result
        status = ParseStatus.SUCCESS
        if errors:
            status = ParseStatus.PARTIAL if memories else ParseStatus.FAILED

        result = ParseResult(
            platform=SourcePlatform.LOCAL,
            status=status,
            memories=memories,
            warnings=warnings,
            errors=errors,
            files_processed=total_files,
            root_path=root,
            parser_version=self.version,
        )

        report_progress(
            f"Completed: {len(memories)} memories extracted", len(memories), len(memories)
        )
        logger.info(f"Parse complete: {status.value}, {len(memories)} memories")

        return result

    def _scan_media_files(self, root: Path) -> Iterator[Path]:
        """Recursively scan for media files.

        Args:
            root: Root directory to scan.

        Yields:
            Path objects for each media file found.

        Notes:
            - Skips hidden files (starting with '.')
            - Skips system files (.DS_Store, Thumbs.db)
            - Handles permission errors gracefully
        """
        skip_dirs = {"__MACOSX", ".git", ".svn", "node_modules"}
        skip_files = {".DS_Store", "Thumbs.db", "desktop.ini"}

        for dirpath, dirnames, filenames in os.walk(root):
            # Remove directories to skip
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip_dirs]

            for filename in filenames:
                # Skip hidden and system files
                if filename.startswith(".") or filename in skip_files:
                    continue

                file_path = Path(dirpath) / filename

                # Check if media file
                if file_path.suffix.lower() in self.supported_extensions:
                    yield file_path

    def _extract_exif_data(self, image_path: Path) -> dict[str, Any]:
        """Extract EXIF data from image file.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary of EXIF data with human-readable tag names.
            Returns empty dict if no EXIF or on error.

        Notes:
            Uses Pillow's lazy loading for performance.
            RAW formats may have limited EXIF support.
        """
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()

                if not exif_data:
                    return {}

                # Ensure exif_data is a dictionary (some files return unexpected types)
                if not isinstance(exif_data, dict):
                    logger.debug(
                        f"EXIF data is not a dict for {image_path}, type: {type(exif_data)}"
                    )
                    return {}

                # Decode tag IDs to readable names
                decoded_exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    decoded_exif[tag_name] = value

                return decoded_exif

        except Exception as e:
            logger.debug(f"Could not extract EXIF from {image_path}: {e}")
            return {}

    def _extract_datetime_from_exif(
        self, exif: dict[str, Any]
    ) -> tuple[datetime | None, ConfidenceLevel]:
        """Extract datetime from EXIF data.

        Args:
            exif: EXIF data dictionary.

        Returns:
            Tuple of (datetime, confidence_level).

        Priority order:
            1. DateTimeOriginal (when photo taken) → HIGH confidence
            2. DateTimeDigitized (when digitized) → HIGH confidence
            3. DateTime (last modification) → MEDIUM confidence

        Notes:
            EXIF times are typically local time without timezone.
            Returns UTC-aware datetime assuming local time.
        """
        # Try DateTimeOriginal first (most reliable)
        dt_str = exif.get("DateTimeOriginal")
        if dt_str:
            dt = self._parse_exif_datetime(dt_str)
            if dt and self._validate_datetime(dt):
                return (dt, ConfidenceLevel.HIGH)

        # Try DateTimeDigitized
        dt_str = exif.get("DateTimeDigitized")
        if dt_str:
            dt = self._parse_exif_datetime(dt_str)
            if dt and self._validate_datetime(dt):
                return (dt, ConfidenceLevel.HIGH)

        # Fall back to DateTime
        dt_str = exif.get("DateTime")
        if dt_str:
            dt = self._parse_exif_datetime(dt_str)
            if dt and self._validate_datetime(dt):
                return (dt, ConfidenceLevel.MEDIUM)

        return (None, ConfidenceLevel.LOW)

    def _parse_exif_datetime(self, dt_str: str) -> datetime | None:
        """Parse EXIF datetime string.

        Args:
            dt_str: EXIF datetime string (e.g., "2019:06:15 14:30:22").

        Returns:
            Timezone-aware datetime object, or None if unparseable.
        """
        try:
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            # Assume local time, convert to UTC-aware
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug(f"Could not parse EXIF datetime: {dt_str}")
            return None

    def _validate_datetime(self, dt: datetime) -> bool:
        """Validate that datetime is reasonable.

        Args:
            dt: Datetime to validate.

        Returns:
            True if datetime seems valid, False otherwise.

        Checks:
            - Not in the future
            - Not before 1990 (likely error)
            - Not exactly 1970-01-01 (Unix epoch default)
        """
        now = datetime.now(timezone.utc)
        epoch_start = datetime(1970, 1, 1, tzinfo=timezone.utc)
        min_date = datetime(1990, 1, 1, tzinfo=timezone.utc)

        # Check for future date
        if dt > now:
            logger.warning(f"Future date detected: {dt}")
            return False

        # Check for epoch default
        if dt == epoch_start:
            logger.debug("Unix epoch date detected (likely error)")
            return False

        # Check for unreasonably old date
        if dt < min_date:
            logger.debug(f"Very old date detected: {dt}")
            return False

        return True

    def _extract_gps_from_exif(self, exif: dict[str, Any]) -> GeoPoint | None:
        """Extract GPS coordinates from EXIF.

        Args:
            exif: EXIF data dictionary.

        Returns:
            GeoPoint object, or None if no valid GPS data.

        Notes:
            EXIF GPS uses Degrees/Minutes/Seconds format that must be
            converted to decimal degrees.
        """
        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None

        try:
            # Extract latitude
            lat_dms = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef (N or S)

            # Extract longitude
            lon_dms = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef (E or W)

            if not all([lat_dms, lat_ref, lon_dms, lon_ref]):
                return None

            # Convert DMS to decimal
            latitude = self._dms_to_decimal(lat_dms, lat_ref)
            longitude = self._dms_to_decimal(lon_dms, lon_ref)

            # Extract altitude if available
            altitude = None
            if 6 in gps_info:  # GPSAltitude
                altitude = float(gps_info[6])

            return GeoPoint(latitude=latitude, longitude=longitude, altitude=altitude)

        except Exception as e:
            logger.warning(f"Error parsing GPS data: {e}")
            return None

    def _dms_to_decimal(self, dms: tuple, ref: str) -> float:
        """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees.

        Args:
            dms: Tuple of ((degrees, divisor), (minutes, divisor), (seconds, divisor)).
            ref: Reference (N/S for latitude, E/W for longitude).

        Returns:
            Decimal degrees (negative for S and W).
        """
        degrees = float(dms[0][0]) / float(dms[0][1])
        minutes = float(dms[1][0]) / float(dms[1][1])
        seconds = float(dms[2][0]) / float(dms[2][1])

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        # Apply negative for South and West
        if ref in ["S", "W"]:
            decimal = -decimal

        return decimal

    def _extract_camera_info(self, exif: dict[str, Any]) -> dict[str, str | None]:
        """Extract camera make and model from EXIF.

        Args:
            exif: EXIF data dictionary.

        Returns:
            Dictionary with 'make' and 'model' keys.
        """
        make = exif.get("Make")
        model = exif.get("Model")

        # Clean up strings
        if make:
            make = str(make).strip().replace("\x00", "")
        if model:
            model = str(model).strip().replace("\x00", "")

        return {"make": make, "model": model}

    def _parse_datetime_from_filename(
        self, filename: str
    ) -> tuple[datetime | None, ConfidenceLevel]:
        """Extract datetime from filename patterns.

        Args:
            filename: Filename to parse.

        Returns:
            Tuple of (datetime, confidence_level).
            Returns (None, LOW) if no pattern matches.

        Patterns:
            - IMG_20190615_143022.jpg (Android)
            - PXL_20190615_143022123.jpg (Google Pixel)
            - Screenshot 2019-06-15 at 14.30.22.png (iOS)
            - 2019-06-15 14-30-22.jpg (Generic)
            - IMG-20190615-WA0001.jpg (WhatsApp)
            - VID_20190615_143022.mp4 (Video)
        """
        for pattern, _format_desc in FILENAME_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()

                    # Build datetime based on captured groups
                    if len(groups) >= 6:
                        # Full datetime
                        year, month, day, hour, minute, second = [int(g) for g in groups[:6]]
                        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                    elif len(groups) == 3:
                        # Date only
                        year, month, day = [int(g) for g in groups[:3]]
                        dt = datetime(year, month, day, tzinfo=timezone.utc)
                    else:
                        continue

                    if self._validate_datetime(dt):
                        return (dt, ConfidenceLevel.MEDIUM)

                except (ValueError, IndexError) as e:
                    logger.debug(f"Pattern matched but datetime invalid: {e}")
                    continue

        return (None, ConfidenceLevel.LOW)

    def _parse_datetime_from_path(self, path: Path) -> tuple[datetime | None, ConfidenceLevel]:
        """Infer datetime from folder path.

        Args:
            path: File path.

        Returns:
            Tuple of (datetime, confidence_level).
            Returns partial datetime (year or year-month) with LOW confidence.

        Patterns:
            - /2019/06/... → June 2019
            - /2019-06-15/... → June 15, 2019
            - /Photos from 2019/... → 2019
        """
        path_str = str(path)

        # Try to find year-month-day pattern
        match = re.search(r"/(\d{4})-(\d{2})-(\d{2})/", path_str)
        if match:
            try:
                year, month, day = [int(g) for g in match.groups()]
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                if self._validate_datetime(dt):
                    return (dt, ConfidenceLevel.LOW)
            except ValueError:
                pass

        # Try year/month pattern
        match = re.search(r"/(\d{4})/(\d{2})/", path_str)
        if match:
            try:
                year, month = [int(g) for g in match.groups()]
                dt = datetime(year, month, 1, tzinfo=timezone.utc)
                if self._validate_datetime(dt):
                    return (dt, ConfidenceLevel.LOW)
            except ValueError:
                pass

        # Try just year
        match = re.search(r"/(\d{4})/", path_str)
        if match:
            try:
                year = int(match.group(1))
                dt = datetime(year, 1, 1, tzinfo=timezone.utc)
                if self._validate_datetime(dt):
                    return (dt, ConfidenceLevel.LOW)
            except ValueError:
                pass

        return (None, ConfidenceLevel.LOW)

    def _infer_context_from_path(self, path: Path) -> dict[str, Any]:
        """Extract contextual hints from folder structure.

        Args:
            path: File path.

        Returns:
            Dictionary with hints: event_hint, year_hint, month_hint, source_hint.

        Examples:
            /photos/Paris 2019/img.jpg → event_hint: "Paris 2019"
            /DCIM/100CANON/img.jpg → source_hint: "100CANON"
        """
        context = {"event_hint": None, "year_hint": None, "month_hint": None, "source_hint": None}

        parts = path.parts

        # Look for year in path
        for part in parts:
            if re.match(r"^\d{4}$", part):
                context["year_hint"] = int(part)
                break

        # Look for DCIM/camera folders
        for part in parts:
            if "DCIM" in part.upper() or "CANON" in part.upper() or "NIKON" in part.upper():
                context["source_hint"] = part
                break

        # Look for event-like folder names (non-numeric, not generic)
        generic_folders = {
            "photos",
            "pictures",
            "images",
            "videos",
            "media",
            "camera roll",
            "downloads",
        }
        for part in parts[-3:]:  # Check last 3 parts
            if part.lower() not in generic_folders and not re.match(r"^\d+$", part):
                # Might be an event folder
                context["event_hint"] = part
                break

        return context

    def _get_file_system_datetime(self, path: Path) -> datetime | None:
        """Get filesystem timestamp as last resort.

        Args:
            path: File path.

        Returns:
            Timezone-aware datetime from file modification time.

        Notes:
            Uses modification time (mtime).
            Always has INFERRED confidence (least reliable).
        """
        try:
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            return mtime
        except Exception as e:
            logger.warning(f"Could not get file timestamp for {path}: {e}")
            return None

    def _detect_screenshot(self, filename: str, exif: dict[str, Any]) -> bool:
        """Detect if image is a screenshot.

        Args:
            filename: Filename to check.
            exif: EXIF data dictionary.

        Returns:
            True if likely a screenshot, False otherwise.

        Detection signals:
            - Filename contains "Screenshot" or "screen shot"
            - EXIF Software is "iOS" or "Android" without camera make
        """
        # Check filename
        if "screenshot" in filename.lower() or "screen shot" in filename.lower():
            return True

        # Check EXIF
        software = exif.get("Software", "").lower()
        make = exif.get("Make", "").lower()

        return bool(("ios" in software or "android" in software) and not make)

    def _create_memory_from_file(self, file_path: Path) -> Memory | None:
        """Create Memory from a single file.

        Args:
            file_path: Path to media file.

        Returns:
            Memory object, or None if file cannot be processed.

        Strategy:
            1. Extract EXIF if available
            2. Get datetime: EXIF → filename → path → filesystem
            3. Get location: EXIF GPS
            4. Get camera info: EXIF Make/Model
            5. Detect media type and screenshot status
            6. Infer context from path
            7. Create Memory with best available data
        """
        # Determine media type from extension
        ext = file_path.suffix.lower()

        # Standard images
        if ext in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".heic",
            ".heif",
            ".webp",
            ".tiff",
            ".tif",
            ".bmp",
        }:
            media_type = MediaType.PHOTO
        # RAW formats (all camera brands)
        elif ext in {
            ".nef",
            ".nrw",
            ".cr2",
            ".cr3",
            ".crw",
            ".arw",
            ".srf",
            ".sr2",
            ".raf",
            ".orf",
            ".rw2",
            ".pef",
            ".ptx",
            ".dng",
            ".raw",
        }:
            media_type = MediaType.PHOTO  # RAW is still a photo
        # Videos
        elif ext in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".wmv", ".flv", ".webm"}:
            media_type = MediaType.VIDEO
        # Audio
        elif ext in {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}:
            media_type = MediaType.AUDIO
        else:
            media_type = MediaType.UNKNOWN

        # Extract EXIF for images (skip videos for performance)
        exif = {}
        if media_type == MediaType.PHOTO:
            exif = self._extract_exif_data(file_path)

        # Get datetime with cascading fallback
        created_at = None
        created_at_confidence = ConfidenceLevel.LOW

        # Try EXIF
        if exif:
            created_at, created_at_confidence = self._extract_datetime_from_exif(exif)

        # Try filename
        if not created_at:
            created_at, created_at_confidence = self._parse_datetime_from_filename(file_path.name)

        # Try path
        if not created_at:
            created_at, _ = self._parse_datetime_from_path(file_path)
            created_at_confidence = ConfidenceLevel.LOW

        # Last resort: filesystem
        if not created_at:
            created_at = self._get_file_system_datetime(file_path)
            created_at_confidence = ConfidenceLevel.LOW

        # Get location from EXIF
        location = None
        if exif:
            geopoint = self._extract_gps_from_exif(exif)
            if geopoint:
                location = Location(geo_point=geopoint, confidence=ConfidenceLevel.HIGH)

        # Get camera info
        camera_info = self._extract_camera_info(exif) if exif else {}

        # Detect screenshot
        if media_type == MediaType.PHOTO and self._detect_screenshot(file_path.name, exif):
            media_type = MediaType.SCREENSHOT

        # Infer context
        context = self._infer_context_from_path(file_path)

        # Create Memory
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=media_type,
            created_at=created_at,
            created_at_confidence=created_at_confidence,
            source_path=str(file_path),
            location=location,
            album_name=context.get("event_hint"),
            original_metadata={
                "exif": exif,
                "camera_make": camera_info.get("make"),
                "camera_model": camera_info.get("model"),
                "context": context,
            },
        )

        return memory
