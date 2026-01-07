"""Google Photos (Takeout) export parser for Digital Life Narrative AI.

Handles Google Photos exports from Google Takeout. These exports have
JSON sidecar files containing rich metadata for each photo/video.

Google Takeout structure:
    Takeout/
    └── Google Photos/
        ├── Photos from 2019/
        │   ├── IMG_1234.jpg
        │   └── IMG_1234.jpg.json  (sidecar metadata)
        ├── Album - Trip to Paris/
        │   └── photo.jpg
        └── print-subscriptions.json
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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

# Media file extensions supported by Google Photos
GOOGLE_PHOTOS_MEDIA_EXTENSIONS = frozenset(
    {
        # Images
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".heic",
        ".heif",
        ".bmp",
        ".tiff",
        ".tif",
        ".raw",
        ".dng",
        # Videos
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".3gp",
        # Live photos motion component
        ".mp",
    }
)

# Files to skip during parsing
SKIP_FILES = frozenset(
    {
        "print-subscriptions.json",
        "shared_album_comments.json",
        "user-generated-memory-titles.json",
        "metadata.json",
        ".picasa.ini",
        "Thumbs.db",
        ".DS_Store",
    }
)

# Folders that indicate album context
ALBUM_FOLDER_PATTERNS = [
    r"^Album[- _](.+)$",
    r"^(.+) - Album$",
]


# =============================================================================
# Google Photos Parser
# =============================================================================


@ParserRegistry.register
class GooglePhotosParser(BaseParser):
    """Parser for Google Photos/Takeout exports.

    Handles the JSON sidecar metadata pattern used by Google Takeout,
    extracting rich metadata including:
    - Precise timestamps (photoTakenTime)
    - GPS coordinates (geoData)
    - People tags
    - Descriptions and captions
    - Album context

    Attributes:
        platform: SourcePlatform.GOOGLE_PHOTOS
        supported_extensions: Set of file extensions to process

    Example:
        ```python
        parser = GooglePhotosParser()
        if parser.can_parse(takeout_path):
            result = parser.parse(takeout_path)
            print(f"Found {len(result.items)} media items")
        ```
    """

    platform = SourcePlatform.GOOGLE_PHOTOS
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".mp4",
        ".mov",
        ".heic",
        ".webp",
        ".json",
        ".heif",
        ".avi",
        ".mkv",
    }

    def __init__(self) -> None:
        """Initialize the Google Photos parser."""
        super().__init__()
        self._processed_hashes: set[str] = set()

    def can_parse(self, root_path: Path) -> bool:
        """Check if this parser can handle the given path.

        Looks for Google Photos-specific structure.

        Args:
            root_path: Path to check.

        Returns:
            True if Google Photos export detected.
        """
        if not root_path.exists() or not root_path.is_dir():
            return False

        # Check for "Google Photos" folder
        google_photos_dir = root_path / "Google Photos"
        if google_photos_dir.exists():
            return True

        # Check in Takeout subfolder
        takeout_dir = root_path / "Takeout" / "Google Photos"
        if takeout_dir.exists():
            return True

        # Check for Google-specific JSON sidecars
        for json_file in root_path.glob("*.json"):
            if self._is_google_sidecar(json_file):
                return True

        return False

    def parse(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParseResult:
        """Parse Google Photos export and extract all media items.

        Args:
            root_path: Root directory of the Google Photos export.
            progress_callback: Optional callback for progress updates.

        Returns:
            ParseResult containing all parsed items.
        """
        start_time = time.time()
        all_items: list[MediaItem] = []

        logger.info(f"Starting Google Photos parse: {root_path}")

        # Reset state
        self._errors = []
        self._stats = {"total_files": 0, "parsed": 0, "skipped": 0, "errors": 0}
        self._processed_hashes = set()

        # Find the actual Google Photos directory
        photos_root = self._find_photos_root(root_path)
        if not photos_root:
            logger.warning("Could not find Google Photos directory")
            return self._create_parse_result([], time.time() - start_time)

        logger.debug(f"Using photos root: {photos_root}")

        # Collect all media files
        media_files = self._collect_media_files(photos_root)
        total_files = len(media_files)
        self._stats["total_files"] = total_files

        logger.info(f"Found {total_files} media files to process")

        # Process each media file
        for idx, media_path in enumerate(media_files):
            try:
                item = self._process_media_file(media_path, photos_root)
                if item:
                    # Deduplication check
                    if item.file_hash and item.file_hash in self._processed_hashes:
                        self._stats["skipped"] += 1
                        logger.debug(f"Skipping duplicate: {media_path.name}")
                        continue

                    if item.file_hash:
                        self._processed_hashes.add(item.file_hash)

                    all_items.append(item)

                if progress_callback and idx % 50 == 0:
                    progress_callback(idx + 1, total_files)

            except Exception as e:
                self._log_error(f"Failed to process: {e}", media_path)

        if progress_callback:
            progress_callback(total_files, total_files)

        duration = time.time() - start_time
        logger.info(f"Google Photos parse complete: {len(all_items)} items in {duration:.2f}s")

        return self._create_parse_result(all_items, duration)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _find_photos_root(self, root_path: Path) -> Path | None:
        """Find the actual Google Photos directory.

        Args:
            root_path: Starting path to search from.

        Returns:
            Path to Google Photos directory or None.
        """
        # Direct path
        if (root_path / "Google Photos").exists():
            return root_path / "Google Photos"

        # Takeout structure
        if (root_path / "Takeout" / "Google Photos").exists():
            return root_path / "Takeout" / "Google Photos"

        # Check if root_path itself is the Google Photos folder
        if root_path.name == "Google Photos":
            return root_path

        # Check if we're inside Takeout
        if root_path.name == "Takeout":
            gp = root_path / "Google Photos"
            if gp.exists():
                return gp

        # Search one level deep
        for subdir in root_path.iterdir():
            if subdir.is_dir() and subdir.name == "Google Photos":
                return subdir

        # Fallback to root if it contains year folders
        year_pattern = re.compile(r"^Photos from \d{4}$")
        for subdir in root_path.iterdir():
            if subdir.is_dir() and year_pattern.match(subdir.name):
                return root_path

        return root_path  # Use root as fallback

    def _collect_media_files(self, photos_root: Path) -> list[Path]:
        """Collect all media files from the export.

        Args:
            photos_root: Root directory to scan.

        Returns:
            List of media file paths.
        """
        media_files: list[Path] = []

        for file_path in photos_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip known non-media files
            if file_path.name in SKIP_FILES:
                continue

            # Skip JSON files (these are sidecars, not media)
            if file_path.suffix.lower() == ".json":
                continue

            # Check if it's a media file
            if file_path.suffix.lower() in GOOGLE_PHOTOS_MEDIA_EXTENSIONS:
                media_files.append(file_path)

        return media_files

    def _process_media_file(
        self,
        media_path: Path,
        photos_root: Path,
    ) -> MediaItem | None:
        """Process a single media file with its sidecar.

        Args:
            media_path: Path to the media file.
            photos_root: Root of the Google Photos export.

        Returns:
            MediaItem or None if processing fails.
        """
        # Find sidecar metadata
        sidecar_path = self._find_sidecar(media_path)
        sidecar_data: dict[str, Any] = {}

        if sidecar_path:
            sidecar_data = self._parse_sidecar(sidecar_path)

        # Determine media type
        media_type = self._determine_media_type(media_path)

        # Extract timestamp
        timestamp, timestamp_confidence = self._extract_timestamp(media_path, sidecar_data)

        # Extract location
        location, location_confidence = self._extract_location(media_path, sidecar_data)

        # Extract people
        people = self._extract_people(sidecar_data)

        # Extract description/caption
        caption = sidecar_data.get("description") or sidecar_data.get("title")

        # Get album context
        album = self._extract_album_info(media_path)

        # Calculate file hash
        file_hash = None
        try:
            file_hash = self._calculate_file_hash(media_path)
        except Exception:
            pass

        # Build original metadata
        original_metadata: dict[str, Any] = {}
        if sidecar_data:
            original_metadata["sidecar"] = sidecar_data
        if album:
            original_metadata["album"] = album

        # Detect if this is a screenshot
        is_screenshot = self._is_screenshot(media_path, sidecar_data)
        if is_screenshot:
            original_metadata["is_screenshot"] = True

        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=media_type,
            file_path=media_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=location,
            location_confidence=location_confidence,
            people=people,
            caption=caption,
            original_metadata=original_metadata,
            file_hash=file_hash,
        )

    def _find_sidecar(self, media_path: Path) -> Path | None:
        """Find the JSON sidecar file for a media file.

        Google Takeout uses various sidecar naming patterns:
        - filename.ext.json (most common)
        - filename.json (without extension)
        - FILENAME.EXT.json (case variations)

        Args:
            media_path: Path to the media file.

        Returns:
            Path to sidecar or None if not found.
        """
        parent = media_path.parent
        name = media_path.name
        stem = media_path.stem

        # Try different sidecar patterns
        patterns = [
            # Standard: filename.ext.json
            f"{name}.json",
            # Without extension: filename.json
            f"{stem}.json",
            # Case variations
            f"{name.lower()}.json",
            f"{name.upper()}.json",
            # Edited versions
            f"{stem}-edited{media_path.suffix}.json",
            f"{stem}-EDITED{media_path.suffix}.json",
            # With parenthetical suffix (Google adds (1), (2) for duplicates)
            re.sub(r"\(\d+\)$", "", stem) + ".json" if "(" in stem else None,
        ]

        for pattern in patterns:
            if pattern is None:
                continue
            sidecar = parent / pattern
            if sidecar.exists():
                return sidecar

        # Try case-insensitive search
        try:
            for file in parent.iterdir():
                if file.suffix.lower() == ".json":
                    # Check if it matches our media file
                    file_lower = file.stem.lower()
                    name_lower = name.lower()
                    stem_lower = stem.lower()

                    if file_lower == name_lower or file_lower == stem_lower:
                        return file
        except PermissionError:
            pass

        return None

    def _parse_sidecar(self, sidecar_path: Path) -> dict[str, Any]:
        """Parse a Google Photos JSON sidecar file.

        Args:
            sidecar_path: Path to the JSON sidecar.

        Returns:
            Parsed metadata dictionary.
        """
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return {}

            return data

        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Failed to parse sidecar {sidecar_path}: {e}")
            return {}

    def _is_google_sidecar(self, json_path: Path) -> bool:
        """Check if a JSON file is a Google Photos sidecar.

        Args:
            json_path: Path to JSON file.

        Returns:
            True if it's a Google Photos sidecar.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return False

            # Check for Google-specific fields
            google_fields = [
                "photoTakenTime",
                "geoData",
                "geoDataExif",
                "googlePhotosOrigin",
                "imageViews",
            ]

            return any(field in data for field in google_fields)

        except (json.JSONDecodeError, IOError):
            return False

    def _parse_google_timestamp(self, ts_obj: dict[str, Any]) -> datetime | None:
        """Parse a Google Photos timestamp object.

        Google provides timestamps as:
        {
            "timestamp": "1577836800",  (Unix epoch in seconds)
            "formatted": "Jan 1, 2020, 12:00:00 AM UTC"
        }

        Args:
            ts_obj: Timestamp object from sidecar.

        Returns:
            Parsed datetime or None.
        """
        if not isinstance(ts_obj, dict):
            return None

        # Prefer Unix timestamp (more precise)
        timestamp_str = ts_obj.get("timestamp")
        if timestamp_str:
            try:
                timestamp = int(timestamp_str)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OSError):
                pass

        # Fall back to formatted string
        formatted = ts_obj.get("formatted")
        if formatted:
            return self._safe_parse_datetime(formatted)

        return None

    def _extract_timestamp(
        self,
        media_path: Path,
        sidecar_data: dict[str, Any],
    ) -> tuple[datetime | None, Confidence]:
        """Extract timestamp from sidecar or fallback sources.

        Priority:
        1. photoTakenTime from sidecar (highest confidence)
        2. creationTime from sidecar
        3. EXIF data from image
        4. Filename parsing
        5. File modification time (lowest confidence)

        Args:
            media_path: Path to the media file.
            sidecar_data: Parsed sidecar metadata.

        Returns:
            Tuple of (timestamp, confidence level).
        """
        # Priority 1: photoTakenTime (original capture time)
        photo_taken = sidecar_data.get("photoTakenTime")
        if photo_taken:
            timestamp = self._parse_google_timestamp(photo_taken)
            if timestamp:
                return timestamp, Confidence.HIGH

        # Priority 2: creationTime (upload time, less preferred)
        creation = sidecar_data.get("creationTime")
        if creation:
            timestamp = self._parse_google_timestamp(creation)
            if timestamp:
                return timestamp, Confidence.MEDIUM

        # Priority 3: EXIF data
        if media_path.suffix.lower() in {".jpg", ".jpeg", ".heic", ".heif", ".png"}:
            exif_time = self._extract_exif_datetime(media_path)
            if exif_time:
                return exif_time, Confidence.HIGH

        # Priority 4: Filename parsing
        filename_time = self._extract_datetime_from_filename(media_path.name)
        if filename_time:
            return filename_time, Confidence.MEDIUM

        # Priority 5: Folder name (e.g., "Photos from 2019")
        folder_year = self._extract_year_from_folder(media_path)
        if folder_year:
            # Create a date for the start of that year
            return datetime(folder_year, 1, 1, tzinfo=timezone.utc), Confidence.LOW

        # Priority 6: File modification time
        try:
            mtime = media_path.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=timezone.utc), Confidence.LOW
        except Exception:
            pass

        return None, Confidence.LOW

    def _extract_year_from_folder(self, file_path: Path) -> int | None:
        """Extract year from folder name like 'Photos from 2019'.

        Args:
            file_path: Path to check parent folders.

        Returns:
            Year as integer or None.
        """
        for parent in file_path.parents:
            match = re.search(r"Photos from (\d{4})", parent.name)
            if match:
                return int(match.group(1))

            # Also check for just year folders
            if parent.name.isdigit() and 1900 <= int(parent.name) <= 2100:
                return int(parent.name)

        return None

    def _extract_location(
        self,
        media_path: Path,
        sidecar_data: dict[str, Any],
    ) -> tuple[GeoLocation | None, Confidence]:
        """Extract location from sidecar or EXIF.

        Args:
            media_path: Path to the media file.
            sidecar_data: Parsed sidecar metadata.

        Returns:
            Tuple of (GeoLocation, confidence level).
        """
        # Try geoData from sidecar (Google's processed location)
        geo_data = sidecar_data.get("geoData")
        if geo_data and isinstance(geo_data, dict):
            lat = geo_data.get("latitude")
            lon = geo_data.get("longitude")

            # Check for valid coordinates (0,0 means no location)
            if lat and lon and not (lat == 0 and lon == 0):
                try:
                    location = GeoLocation(
                        latitude=float(lat),
                        longitude=float(lon),
                        raw_location_string=str(geo_data),
                    )
                    return location, Confidence.HIGH
                except (ValueError, TypeError):
                    pass

        # Try geoDataExif (EXIF-based location)
        geo_exif = sidecar_data.get("geoDataExif")
        if geo_exif and isinstance(geo_exif, dict):
            lat = geo_exif.get("latitude")
            lon = geo_exif.get("longitude")

            if lat and lon and not (lat == 0 and lon == 0):
                try:
                    location = GeoLocation(
                        latitude=float(lat),
                        longitude=float(lon),
                        raw_location_string=str(geo_exif),
                    )
                    return location, Confidence.MEDIUM
                except (ValueError, TypeError):
                    pass

        return None, Confidence.LOW

    def _extract_people(self, sidecar_data: dict[str, Any]) -> list[str]:
        """Extract people tags from sidecar.

        Args:
            sidecar_data: Parsed sidecar metadata.

        Returns:
            List of people names.
        """
        people: list[str] = []

        people_data = sidecar_data.get("people", [])
        if isinstance(people_data, list):
            for person in people_data:
                if isinstance(person, dict):
                    name = person.get("name")
                    if name:
                        people.append(name)
                elif isinstance(person, str):
                    people.append(person)

        return people

    def _extract_album_info(self, file_path: Path) -> str | None:
        """Extract album name from folder structure.

        Checks if file is in an album-named folder.

        Args:
            file_path: Path to the media file.

        Returns:
            Album name or None.
        """
        for parent in file_path.parents:
            # Check album patterns
            for pattern in ALBUM_FOLDER_PATTERNS:
                match = re.match(pattern, parent.name, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            # Also check for explicit album naming
            if parent.name.lower().startswith("album"):
                # Extract the part after "album"
                album_name = re.sub(r"^album[- _]*", "", parent.name, flags=re.IGNORECASE)
                if album_name:
                    return album_name.strip()

        return None

    def _determine_media_type(self, media_path: Path) -> MediaType:
        """Determine the media type from file extension.

        Args:
            media_path: Path to the media file.

        Returns:
            Appropriate MediaType.
        """
        ext = media_path.suffix.lower()

        if ext in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".heic",
            ".heif",
            ".bmp",
            ".tiff",
            ".tif",
        }:
            return MediaType.PHOTO
        elif ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}:
            return MediaType.VIDEO
        else:
            return MediaType.UNKNOWN

    def _is_screenshot(
        self,
        media_path: Path,
        sidecar_data: dict[str, Any],
    ) -> bool:
        """Detect if a file is a screenshot.

        Args:
            media_path: Path to the media file.
            sidecar_data: Parsed sidecar metadata.

        Returns:
            True if file appears to be a screenshot.
        """
        name_lower = media_path.name.lower()

        # Common screenshot naming patterns
        screenshot_patterns = [
            "screenshot",
            "screen shot",
            "screen_shot",
            "captura",  # Spanish
            "schermafbeelding",  # Dutch
            "bildschirmfoto",  # German
        ]

        for pattern in screenshot_patterns:
            if pattern in name_lower:
                return True

        # Check folder name
        for parent in media_path.parents:
            if "screenshot" in parent.name.lower():
                return True

        # Check sidecar for screenshot indicators
        title = sidecar_data.get("title", "").lower()
        for pattern in screenshot_patterns:
            if pattern in title:
                return True

        return False
