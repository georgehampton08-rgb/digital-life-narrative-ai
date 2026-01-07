"""Local photos/media parser for Digital Life Narrative AI.

This parser handles local photo folders – photos from DSLRs, phone transfers,
external drives, etc. that aren't from any specific platform export. This is
the "fallback" parser for generic media.

Focus areas:
- EXIF extraction for images (timestamps, GPS, camera info)
- Filename parsing for dates
- Folder name inference for context
- Basic video metadata extraction
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from organizer.models import (
    Confidence,
    GeoLocation,
    MediaItem,
    MediaType,
    ParseResult,
    SourcePlatform,
)
from organizer.parsers.base import BaseParser, ParserRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Supported media extensions
IMAGE_EXTENSIONS = frozenset(
    {
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
    }
)

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".3gp",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
    }
)

AUDIO_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".wma",
    }
)

# Files to skip
SKIP_FILES = frozenset(
    {
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
        ".picasa.ini",
        "albumdata.xml",
        ".nomedia",
        ".thumbnails",
    }
)

SKIP_PREFIXES = (".", "~", "__")

# Common filename date patterns (compiled for performance)
FILENAME_DATE_PATTERNS = [
    # Android/Pixel: IMG_20190615_143022.jpg, VID_20190615_143022.mp4
    (re.compile(r"(?:IMG|VID|PXL)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"), True),
    # Screenshot patterns
    (
        re.compile(r"Screenshot[_\s-]?(\d{4})-?(\d{2})-?(\d{2})[_\s-]?(\d{2})-?(\d{2})-?(\d{2})"),
        True,
    ),
    # ISO format: 2019-06-15_14-30-22.jpg or 2019-06-15 14.30.22.jpg
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_\s\.](\d{2})[\.\-](\d{2})[\.\-](\d{2})"), True),
    # WhatsApp: IMG-20190615-WA0001.jpg
    (re.compile(r"IMG-(\d{4})(\d{2})(\d{2})-WA\d+"), False),
    # Date only patterns
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})(?!\d)"), False),
    (re.compile(r"(\d{4})(\d{2})(\d{2})(?!\d)"), False),
]

# Folder patterns for context inference
YEAR_PATTERN = re.compile(r"^(19|20)\d{2}$")
YEAR_MONTH_PATTERN = re.compile(r"^(19|20)\d{2}[_\-](0[1-9]|1[0-2])$")
LOCATION_PATTERNS = [
    re.compile(r"(.+?)\s+(19|20)\d{2}$"),  # "Paris 2019"
    re.compile(r"(19|20)\d{2}\s+(.+)$"),  # "2019 Paris"
]
EVENT_KEYWORDS = frozenset(
    {
        "wedding",
        "birthday",
        "vacation",
        "trip",
        "holiday",
        "christmas",
        "easter",
        "thanksgiving",
        "graduation",
        "party",
        "concert",
        "festival",
    }
)


# =============================================================================
# Local Photos Parser
# =============================================================================


@ParserRegistry.register
class LocalPhotosParser(BaseParser):
    """Parser for local/generic media folders.

    Fallback parser for media that doesn't come from a specific platform.
    Focuses on EXIF extraction, filename parsing, and folder context inference.

    Attributes:
        platform: SourcePlatform.LOCAL
        supported_extensions: Set of file extensions to process

    Example:
        ```python
        parser = LocalPhotosParser()
        result = parser.parse(Path("/path/to/photos"))
        print(f"Found {len(result.items)} media items")
        ```
    """

    platform = SourcePlatform.LOCAL
    supported_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

    def __init__(self) -> None:
        """Initialize the local photos parser."""
        super().__init__()

    def can_parse(self, root_path: Path) -> bool:
        """Check if directory contains parseable media files.

        This is a fallback parser, so it can parse any directory.

        Args:
            root_path: Path to check.

        Returns:
            True (fallback parser).
        """
        return True

    def parse(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParseResult:
        """Parse local media folder and extract all media items.

        Args:
            root_path: Root directory to scan.
            progress_callback: Optional callback for progress updates.

        Returns:
            ParseResult containing all parsed items.
        """
        start_time = time.time()
        all_items: list[MediaItem] = []

        logger.info(f"Starting local media parse: {root_path}")

        # Reset state
        self._errors = []
        self._stats = {"total_files": 0, "parsed": 0, "skipped": 0, "errors": 0}

        # Collect all media files
        media_files = self._collect_media_files(root_path)
        total_files = len(media_files)
        self._stats["total_files"] = total_files

        logger.info(f"Found {total_files} media files to process")

        # Process each file
        for idx, file_path in enumerate(media_files):
            try:
                item = self._process_file(file_path, root_path)
                if item:
                    all_items.append(item)

                if progress_callback and idx % 100 == 0:
                    progress_callback(idx + 1, total_files)

            except Exception as e:
                self._log_error(f"Failed to process: {e}", file_path)

        if progress_callback:
            progress_callback(total_files, total_files)

        duration = time.time() - start_time
        logger.info(f"Local parse complete: {len(all_items)} items in {duration:.2f}s")

        return self._create_parse_result(all_items, duration)

    # =========================================================================
    # File Collection
    # =========================================================================

    def _collect_media_files(self, root_path: Path) -> list[Path]:
        """Collect all media files from directory tree.

        Skips hidden files, system files, and unsupported formats.

        Args:
            root_path: Root directory to scan.

        Returns:
            List of media file paths.
        """
        media_files: list[Path] = []

        try:
            for file_path in root_path.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip hidden/system files
                if self._should_skip_file(file_path):
                    continue

                # Check extension
                if file_path.suffix.lower() in self.supported_extensions:
                    media_files.append(file_path)

        except PermissionError as e:
            logger.warning(f"Permission denied scanning: {e}")

        return media_files

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped.

        Args:
            file_path: Path to check.

        Returns:
            True if file should be skipped.
        """
        name_lower = file_path.name.lower()

        # Skip known system files
        if name_lower in SKIP_FILES:
            return True

        # Skip files with certain prefixes
        if any(file_path.name.startswith(prefix) for prefix in SKIP_PREFIXES):
            return True

        # Skip files in hidden directories
        for parent in file_path.parents:
            if parent.name.startswith("."):
                return True

        return False

    # =========================================================================
    # File Processing
    # =========================================================================

    def _process_file(self, file_path: Path, root_path: Path) -> MediaItem | None:
        """Process a single media file.

        Args:
            file_path: Path to the media file.
            root_path: Root parse directory for context.

        Returns:
            MediaItem or None if processing fails.
        """
        ext = file_path.suffix.lower()

        # Determine media type
        if ext in IMAGE_EXTENSIONS:
            media_type = MediaType.PHOTO
        elif ext in VIDEO_EXTENSIONS:
            media_type = MediaType.VIDEO
        elif ext in AUDIO_EXTENSIONS:
            media_type = MediaType.AUDIO
        else:
            media_type = MediaType.UNKNOWN

        # Extract metadata based on type
        if media_type == MediaType.PHOTO:
            return self._process_image(file_path, root_path)
        elif media_type == MediaType.VIDEO:
            return self._process_video(file_path, root_path)
        elif media_type == MediaType.AUDIO:
            return self._process_audio(file_path, root_path)
        else:
            return self._process_generic(file_path, root_path, media_type)

    def _process_image(self, file_path: Path, root_path: Path) -> MediaItem:
        """Process an image file with full EXIF extraction.

        Args:
            file_path: Path to the image.
            root_path: Root parse directory.

        Returns:
            MediaItem with extracted metadata.
        """
        # Extract EXIF data
        exif_data = self._extract_full_exif(file_path)

        # Get timestamp from EXIF or fallback
        timestamp, timestamp_confidence = self._get_image_timestamp(file_path, exif_data)

        # Get location from EXIF GPS
        location, location_confidence = self._get_image_location(exif_data)

        # Infer additional context from folder path
        folder_hints = self._infer_from_folder_path(file_path)

        # If no location from EXIF, try folder hints
        if location is None and folder_hints.get("location_hint"):
            location = GeoLocation(place_name=folder_hints["location_hint"])
            location_confidence = Confidence.LOW

        # Build caption from EXIF description
        caption = exif_data.get("parsed", {}).get("description")

        # Original metadata
        original_metadata: dict[str, Any] = {"source": "local"}
        if exif_data.get("raw"):
            # Store subset of EXIF to avoid huge metadata
            original_metadata["exif"] = self._summarize_exif(exif_data["raw"])
        if folder_hints:
            original_metadata["folder_hints"] = folder_hints

        # Calculate hash
        file_hash = None
        try:
            file_hash = self._calculate_file_hash(file_path)
        except Exception:
            pass

        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=file_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=location,
            location_confidence=location_confidence,
            people=[],  # Would need face detection for this
            caption=caption,
            original_metadata=original_metadata,
            file_hash=file_hash,
        )

    def _process_video(self, file_path: Path, root_path: Path) -> MediaItem:
        """Process a video file.

        Args:
            file_path: Path to the video.
            root_path: Root parse directory.

        Returns:
            MediaItem with extracted metadata.
        """
        # Extract what metadata we can
        video_metadata = self._extract_video_metadata(file_path)

        # Get timestamp
        timestamp, timestamp_confidence = self._get_file_timestamp(file_path)

        # Infer from folder
        folder_hints = self._infer_from_folder_path(file_path)

        # Check for location hint from folder
        location = None
        location_confidence = Confidence.LOW
        if folder_hints.get("location_hint"):
            location = GeoLocation(place_name=folder_hints["location_hint"])

        # Original metadata
        original_metadata: dict[str, Any] = {"source": "local"}
        if video_metadata:
            original_metadata["video"] = video_metadata
        if folder_hints:
            original_metadata["folder_hints"] = folder_hints

        # Calculate hash
        file_hash = None
        try:
            file_hash = self._calculate_file_hash(file_path)
        except Exception:
            pass

        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.VIDEO,
            file_path=file_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=location,
            location_confidence=location_confidence,
            people=[],
            caption=None,
            original_metadata=original_metadata,
            file_hash=file_hash,
        )

    def _process_audio(self, file_path: Path, root_path: Path) -> MediaItem:
        """Process an audio file.

        Args:
            file_path: Path to the audio file.
            root_path: Root parse directory.

        Returns:
            MediaItem with extracted metadata.
        """
        # Get timestamp from filename or file dates
        timestamp, timestamp_confidence = self._get_file_timestamp(file_path)

        # Infer from folder
        folder_hints = self._infer_from_folder_path(file_path)

        original_metadata: dict[str, Any] = {"source": "local"}
        if folder_hints:
            original_metadata["folder_hints"] = folder_hints

        # Calculate hash
        file_hash = None
        try:
            file_hash = self._calculate_file_hash(file_path)
        except Exception:
            pass

        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.AUDIO,
            file_path=file_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=None,
            location_confidence=Confidence.LOW,
            people=[],
            caption=None,
            original_metadata=original_metadata,
            file_hash=file_hash,
        )

    def _process_generic(
        self,
        file_path: Path,
        root_path: Path,
        media_type: MediaType,
    ) -> MediaItem:
        """Process a generic/unknown media file.

        Args:
            file_path: Path to the file.
            root_path: Root parse directory.
            media_type: Pre-determined media type.

        Returns:
            MediaItem with basic metadata.
        """
        timestamp, timestamp_confidence = self._get_file_timestamp(file_path)
        folder_hints = self._infer_from_folder_path(file_path)

        file_hash = None
        try:
            file_hash = self._calculate_file_hash(file_path)
        except Exception:
            pass

        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.LOCAL,
            media_type=media_type,
            file_path=file_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=None,
            location_confidence=Confidence.LOW,
            people=[],
            caption=None,
            original_metadata={"source": "local", "folder_hints": folder_hints},
            file_hash=file_hash,
        )

    # =========================================================================
    # EXIF Extraction
    # =========================================================================

    def _extract_full_exif(self, image_path: Path) -> dict[str, Any]:
        """Extract full EXIF data from an image.

        Uses Pillow's lazy loading to avoid loading entire image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with 'raw' (original EXIF) and 'parsed' (extracted values).
        """
        result: dict[str, Any] = {"raw": {}, "parsed": {}}

        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return result

                # Ensure exif_data is a dictionary (some files return unexpected types)
                if not isinstance(exif_data, dict):
                    logger.debug(
                        f"EXIF data is not a dict for {image_path}, type: {type(exif_data)}"
                    )
                    return result

                # Map numeric tags to names
                raw_exif: dict[str, Any] = {}
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    raw_exif[tag_name] = value

                result["raw"] = raw_exif

                # Parse specific fields
                parsed = result["parsed"]

                # Timestamps
                for field in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                    if field in raw_exif:
                        dt = self._safe_parse_datetime(str(raw_exif[field]))
                        if dt:
                            parsed["datetime"] = dt
                            parsed["datetime_field"] = field
                            break

                # Camera info
                if "Make" in raw_exif:
                    parsed["camera_make"] = str(raw_exif["Make"]).strip()
                if "Model" in raw_exif:
                    parsed["camera_model"] = str(raw_exif["Model"]).strip()

                # Description/Caption
                for field in ["ImageDescription", "UserComment"]:
                    if field in raw_exif and raw_exif[field]:
                        desc = str(raw_exif[field]).strip()
                        if desc and desc.lower() not in ["", "ascii", "binary"]:
                            parsed["description"] = desc
                            break

                # Artist
                if "Artist" in raw_exif:
                    parsed["artist"] = str(raw_exif["Artist"]).strip()

                # GPS Info
                if "GPSInfo" in raw_exif:
                    gps_info = raw_exif["GPSInfo"]
                    location = self._parse_gps_exif(gps_info)
                    if location:
                        parsed["location"] = location

        except Exception as e:
            logger.debug(f"Failed to extract EXIF from {image_path}: {e}")

        return result

    def _parse_gps_exif(self, gps_info: dict[Any, Any]) -> GeoLocation | None:
        """Parse GPS EXIF data into GeoLocation.

        EXIF GPS stores coordinates as degrees/minutes/seconds with
        N/S/E/W references.

        Args:
            gps_info: GPS info dict from EXIF.

        Returns:
            GeoLocation or None if parsing fails.
        """
        try:
            # Ensure gps_info is a dictionary (can be a list in some RAW files)
            if not isinstance(gps_info, dict):
                logger.debug(f"GPS info is not a dict, type: {type(gps_info)}")
                return None

            # Decode GPS tags if needed
            decoded_gps: dict[str, Any] = {}
            for tag_id, value in gps_info.items():
                tag_name = GPSTAGS.get(tag_id, str(tag_id))
                decoded_gps[tag_name] = value

            # Extract latitude
            lat = None
            if "GPSLatitude" in decoded_gps and "GPSLatitudeRef" in decoded_gps:
                lat = self._dms_to_decimal(
                    decoded_gps["GPSLatitude"],
                    decoded_gps["GPSLatitudeRef"],
                )

            # Extract longitude
            lon = None
            if "GPSLongitude" in decoded_gps and "GPSLongitudeRef" in decoded_gps:
                lon = self._dms_to_decimal(
                    decoded_gps["GPSLongitude"],
                    decoded_gps["GPSLongitudeRef"],
                )

            if lat is not None and lon is not None:
                return GeoLocation(
                    latitude=lat,
                    longitude=lon,
                    raw_location_string=str(decoded_gps),
                )

        except Exception as e:
            logger.debug(f"Failed to parse GPS EXIF: {e}")

        return None

    def _dms_to_decimal(
        self,
        dms: tuple[Any, ...],
        ref: str,
    ) -> float | None:
        """Convert degrees/minutes/seconds to decimal degrees.

        Args:
            dms: Tuple of (degrees, minutes, seconds) - may be IFDRational.
            ref: Reference direction (N/S/E/W).

        Returns:
            Decimal degrees or None.
        """
        try:
            # Handle IFDRational or tuple format
            def to_float(val: Any) -> float:
                if hasattr(val, "numerator") and hasattr(val, "denominator"):
                    if val.denominator == 0:
                        return 0.0
                    return float(val.numerator) / float(val.denominator)
                return float(val)

            degrees = to_float(dms[0])
            minutes = to_float(dms[1])
            seconds = to_float(dms[2]) if len(dms) > 2 else 0.0

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            # Apply direction
            if ref.upper() in ("S", "W"):
                decimal = -decimal

            return decimal

        except Exception as e:
            logger.debug(f"Failed to convert DMS to decimal: {e}")
            return None

    def _summarize_exif(self, raw_exif: dict[str, Any]) -> dict[str, Any]:
        """Create a smaller summary of EXIF for storage.

        Args:
            raw_exif: Full raw EXIF dict.

        Returns:
            Summarized EXIF with key fields only.
        """
        keep_fields = {
            "DateTimeOriginal",
            "DateTime",
            "Make",
            "Model",
            "ExposureTime",
            "FNumber",
            "ISOSpeedRatings",
            "FocalLength",
            "ImageWidth",
            "ImageHeight",
        }

        summary = {}
        for field in keep_fields:
            if field in raw_exif:
                value = raw_exif[field]
                # Convert non-serializable types
                if hasattr(value, "numerator"):
                    value = f"{value.numerator}/{value.denominator}"
                summary[field] = str(value) if not isinstance(value, (int, float, str)) else value

        return summary

    # =========================================================================
    # Timestamp Extraction
    # =========================================================================

    def _get_image_timestamp(
        self,
        file_path: Path,
        exif_data: dict[str, Any],
    ) -> tuple[datetime | None, Confidence]:
        """Get timestamp for an image from multiple sources.

        Priority:
        1. EXIF DateTimeOriginal (highest confidence)
        2. Filename parsing
        3. File modification time (lowest confidence)

        Args:
            file_path: Path to the image.
            exif_data: Extracted EXIF data.

        Returns:
            Tuple of (timestamp, confidence level).
        """
        # Priority 1: EXIF datetime
        parsed = exif_data.get("parsed", {})
        if "datetime" in parsed:
            return parsed["datetime"], Confidence.HIGH

        # Priority 2: Filename parsing
        filename_dt = self._parse_filename_date(file_path.name)
        if filename_dt:
            return filename_dt, Confidence.MEDIUM

        # Priority 3: Folder year
        folder_hints = self._infer_from_folder_path(file_path)
        if folder_hints.get("year"):
            year = folder_hints["year"]
            month = folder_hints.get("month", 1)
            return datetime(year, month, 1, tzinfo=timezone.utc), Confidence.LOW

        # Priority 4: File modification time
        try:
            mtime = file_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc), Confidence.LOW
        except Exception:
            pass

        return None, Confidence.LOW

    def _get_file_timestamp(
        self,
        file_path: Path,
    ) -> tuple[datetime | None, Confidence]:
        """Get timestamp for a non-image file.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (timestamp, confidence level).
        """
        # Try filename parsing first
        filename_dt = self._parse_filename_date(file_path.name)
        if filename_dt:
            return filename_dt, Confidence.MEDIUM

        # Folder year
        folder_hints = self._infer_from_folder_path(file_path)
        if folder_hints.get("year"):
            year = folder_hints["year"]
            month = folder_hints.get("month", 1)
            return datetime(year, month, 1, tzinfo=timezone.utc), Confidence.LOW

        # File modification time
        try:
            mtime = file_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc), Confidence.LOW
        except Exception:
            pass

        return None, Confidence.LOW

    def _get_image_location(
        self,
        exif_data: dict[str, Any],
    ) -> tuple[GeoLocation | None, Confidence]:
        """Get location from EXIF data.

        Args:
            exif_data: Extracted EXIF data.

        Returns:
            Tuple of (GeoLocation, confidence level).
        """
        parsed = exif_data.get("parsed", {})
        if "location" in parsed:
            return parsed["location"], Confidence.HIGH

        return None, Confidence.LOW

    def _parse_filename_date(self, filename: str) -> datetime | None:
        """Parse date/time from filename using common patterns.

        Supports formats:
        - IMG_20190615_143022.jpg (Android)
        - PXL_20190615_143022.jpg (Pixel)
        - 2019-06-15_14-30-22.jpg
        - Screenshot_2019-06-15-14-30-22.png
        - IMG-20190615-WA0001.jpg (WhatsApp)

        Args:
            filename: The filename to parse.

        Returns:
            Parsed datetime or None.
        """
        for pattern, has_time in FILENAME_DATE_PATTERNS:
            match = pattern.search(filename)
            if match:
                groups = match.groups()
                try:
                    year = int(groups[0])
                    month = int(groups[1])
                    day = int(groups[2])

                    # Validate date components
                    if not (1900 <= year <= 2100):
                        continue
                    if not (1 <= month <= 12):
                        continue
                    if not (1 <= day <= 31):
                        continue

                    if has_time and len(groups) >= 6:
                        hour = int(groups[3])
                        minute = int(groups[4])
                        second = int(groups[5])

                        if not (0 <= hour <= 23):
                            continue
                        if not (0 <= minute <= 59):
                            continue
                        if not (0 <= second <= 59):
                            continue

                        return datetime(
                            year,
                            month,
                            day,
                            hour,
                            minute,
                            second,
                            tzinfo=timezone.utc,
                        )
                    else:
                        return datetime(year, month, day, tzinfo=timezone.utc)

                except (ValueError, IndexError):
                    continue

        return None

    # =========================================================================
    # Folder Context Inference
    # =========================================================================

    def _infer_from_folder_path(self, file_path: Path) -> dict[str, Any]:
        """Infer context from parent folder names.

        Extracts:
        - Year/month from folder names
        - Location hints from folder names
        - Event hints from folder names

        Args:
            file_path: Path to the media file.

        Returns:
            Dict with inferred hints.
        """
        hints: dict[str, Any] = {}

        for parent in file_path.parents:
            name = parent.name

            # Skip empty or root-like names
            if not name or name in (".", "..", "/"):
                continue

            # Check for year folder
            if YEAR_PATTERN.match(name):
                hints["year"] = int(name)
                continue

            # Check for year-month folder
            if YEAR_MONTH_PATTERN.match(name):
                parts = re.split(r"[-_]", name)
                hints["year"] = int(parts[0])
                hints["month"] = int(parts[1])
                continue

            # Check for location patterns like "Paris 2019"
            for pattern in LOCATION_PATTERNS:
                match = pattern.match(name)
                if match:
                    groups = match.groups()
                    # Determine which group is the location
                    for group in groups:
                        if not group.isdigit():
                            hints["location_hint"] = group.strip()
                            break
                    break

            # Check for event keywords
            name_lower = name.lower()
            for keyword in EVENT_KEYWORDS:
                if keyword in name_lower:
                    hints["event_hint"] = name
                    break

        return hints

    # =========================================================================
    # Video Metadata
    # =========================================================================

    def _extract_video_metadata(self, video_path: Path) -> dict[str, Any]:
        """Extract basic video metadata.

        Currently extracts:
        - File size
        - Creation/modification times
        - Filename-derived date

        For full metadata (duration, resolution, codec), ffprobe would be needed.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with video metadata.
        """
        metadata: dict[str, Any] = {}

        try:
            stat = video_path.stat()
            metadata["file_size_bytes"] = stat.st_size
            metadata["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)

            # File times
            metadata["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            # Try to get creation time (platform-dependent)
            try:
                ctime = stat.st_birthtime  # macOS
                metadata["ctime"] = datetime.fromtimestamp(ctime, tz=timezone.utc).isoformat()
            except AttributeError:
                # Windows/Linux use st_ctime differently
                metadata["ctime"] = datetime.fromtimestamp(
                    stat.st_ctime, tz=timezone.utc
                ).isoformat()

            # Filename date
            filename_dt = self._parse_filename_date(video_path.name)
            if filename_dt:
                metadata["filename_date"] = filename_dt.isoformat()

        except Exception as e:
            logger.debug(f"Failed to extract video metadata from {video_path}: {e}")

        # Note: For full video metadata (duration, resolution, codec),
        # would need ffprobe or similar tool:
        # ffprobe -v quiet -print_format json -show_format -show_streams video.mp4
        metadata["note"] = "Full video metadata requires ffprobe"

        return metadata
