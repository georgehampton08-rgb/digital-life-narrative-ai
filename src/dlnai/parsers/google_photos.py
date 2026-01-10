"""Google Photos parser for Digital Life Narrative AI.

This module provides a concrete parser that converts Google Photos Takeout
exports into normalized Memory objects. It handles the unique "sidecar JSON"
pattern where each photo/video has an accompanying JSON file with rich metadata
including EXIF data, location, people tags, and timestamps.

Google Photos exports are often the richest data source because they preserve
EXIF data, face detection results, location information, and album organization.

Typical usage:
    >>> from pathlib import Path
    >>> from dlnai.parsers.google_photos import GooglePhotosParser
    >>>
    >>> parser = GooglePhotosParser()
    >>> if parser.can_parse(Path("/exports/google_photos")):
    ...     result = parser.parse(Path("/exports/google_photos"))
    ...     print(f"Parsed {len(result.memories)} memories")
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

from dlnai.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)
from dlnai.parsers.base import (
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


@register_parser
class GooglePhotosParser(BaseParser):
    """Parser for Google Photos Takeout exports.

    Handles the sidecar JSON pattern where each media file has a corresponding
    .json file containing rich metadata including timestamps, location, people
    tags (from face recognition), and album information.

    Attributes:
        platform: Source platform (GOOGLE_PHOTOS).
        version: Parser version.
        supported_extensions: File extensions this parser handles.
        description: Human-readable parser description.
    """

    platform: ClassVar[SourcePlatform] = SourcePlatform.GOOGLE_PHOTOS
    version: ClassVar[str] = "1.0.0"
    supported_extensions: ClassVar[set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".heic",
        ".webp",
        ".mp4",
        ".mov",
        ".json",
    }
    description: ClassVar[str] = "Parser for Google Photos Takeout exports"

    def can_parse(self, root: Path) -> bool:
        """Check if this parser can handle the given directory.

        Args:
            root: Directory to check.

        Returns:
            True if directory appears to be a Google Photos export, False otherwise.

        Detection criteria:
            - "Google Photos" directory exists at root or under "Takeout/", OR
            - Contains .json sidecars with Google Photos structure, OR
            - Contains "Photos from YYYY" directories
        """
        # Check for Google Photos directory
        if (root / "Google Photos").is_dir():
            return True

        if (root / "Takeout" / "Google Photos").is_dir():
            return True

        # Check for "Photos from YYYY" pattern
        for item in root.iterdir():
            if item.is_dir() and re.match(r"Photos from \d{4}", item.name):
                return True

        # Check for Google Photos specific marker
        if (root / "print-subscriptions.json").exists():
            return True

        # Check for sidecar pattern (at least a few .jpg.json files)
        json_sidecars = 0
        for item in root.rglob("*.jpg.json"):
            json_sidecars += 1
            if json_sidecars >= 3:  # At least 3 sidecars
                return True

        return False

    def get_signature_files(self) -> List[str]:
        """Get list of signature files/directories for Google Photos exports.

        Returns:
            List of file/directory patterns that indicate a Google Photos export.
        """
        return ["Google Photos/", "Takeout/", "print-subscriptions.json", "Photos from */"]

    def parse(self, root: Path, progress: ProgressCallback | None = None) -> ParseResult:
        """Parse Google Photos export into Memory objects.

        Args:
            root: Root directory of the Google Photos export.
            progress: Optional callback for progress reporting.

        Returns:
            ParseResult containing extracted memories, warnings, and errors.

        Processing steps:
            1. Find Google Photos root directory
            2. Scan for all media files with sidecars
            3. Parse each sidecar and create Memory objects
            4. Detect Live Photos (paired .jpg + .mp4)
            5. Deduplicate copies
            6. Return comprehensive result
        """
        memories: List[Memory] = []
        warnings: List[ParseWarning] = []
        errors: List[ParseError] = []

        def report_progress(stage: str, current: int = 0, total: int = 0):
            """Helper to report progress if callback provided."""
            if progress:
                progress(ParseProgress(current=current, total=total, stage=stage))

        report_progress("Scanning Google Photos export...")
        logger.info(f"Starting Google Photos parse of {root}")

        # Find Google Photos root
        photos_root = self._find_google_photos_root(root)
        if not photos_root:
            error = ParseError(
                file_path=root,
                message="Could not find Google Photos directory",
                error_type="directory_not_found",
            )
            errors.append(error)
            return ParseResult(
                platform=SourcePlatform.GOOGLE_PHOTOS,
                status=ParseStatus.FAILED,
                memories=[],
                warnings=warnings,
                errors=errors,
            )

        logger.info(f"Found Google Photos root at {photos_root}")

        # Scan for media files
        report_progress("Scanning for media files...")
        media_pairs = list(self._scan_for_media_files(photos_root))
        total_files = len(media_pairs)

        report_progress(f"Found {total_files} media files", 0, total_files)
        logger.info(f"Found {total_files} media files to process")

        # Process each media file
        for idx, (media_path, sidecar_path) in enumerate(media_pairs):
            try:
                # Parse sidecar if available
                sidecar_data = None
                if sidecar_path:
                    sidecar_data = self._parse_sidecar(sidecar_path)
                    if not sidecar_data:
                        warnings.append(
                            ParseWarning(
                                message=f"Invalid sidecar JSON", file_path=str(sidecar_path)
                            )
                        )

                # Create Memory from pair
                memory = self._create_memory_from_pair(media_path, sidecar_data)
                memories.append(memory)

                # Report progress periodically
                if idx % 100 == 0:
                    report_progress(f"Processing media files...", idx, total_files)

            except Exception as e:
                logger.warning(f"Error processing {media_path}: {e}", exc_info=True)
                errors.append(
                    ParseError(
                        file_path=media_path,
                        message=f"Failed to process media file: {e}",
                        error_type="media_parse_error",
                        original_exception=e,
                    )
                )

        # Post-processing: Detect Live Photos
        report_progress("Detecting Live Photos...")
        memories = self._pair_live_photos(memories)

        # Deduplicate
        report_progress("Deduplicating memories...")
        unique_memories = self._deduplicate_memories(memories)
        duplicates_removed = len(memories) - len(unique_memories)

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate memories")

        # Build result
        status = ParseStatus.SUCCESS
        if errors:
            status = ParseStatus.PARTIAL if unique_memories else ParseStatus.FAILED

        result = ParseResult(
            platform=SourcePlatform.GOOGLE_PHOTOS,
            status=status,
            memories=unique_memories,
            warnings=warnings,
            errors=errors,
            files_processed=total_files,
            root_path=photos_root,
            parser_version=self.version,
        )

        report_progress(
            f"Completed: {len(unique_memories)} memories extracted",
            len(unique_memories),
            len(unique_memories),
        )
        logger.info(f"Parse complete: {status.value}, {len(unique_memories)} memories")

        return result

    def _find_google_photos_root(self, root: Path) -> Optional[Path]:
        """Find the actual Google Photos directory within the export.

        Args:
            root: Root directory of export.

        Returns:
            Path to Google Photos directory, or None if not found.

        Checks:
            - root itself
            - root/Google Photos
            - root/Takeout/Google Photos
        """
        # Check if root is the Google Photos dir
        if (root / "Photos from 2019").is_dir() or (root / "print-subscriptions.json").exists():
            return root

        # Check for Google Photos subdirectory
        gp_dir = root / "Google Photos"
        if gp_dir.is_dir():
            return gp_dir

        # Check for Takeout/Google Photos
        takeout_gp = root / "Takeout" / "Google Photos"
        if takeout_gp.is_dir():
            return takeout_gp

        return None

    def _scan_for_media_files(self, photos_root: Path) -> Iterator[Tuple[Path, Optional[Path]]]:
        """Scan for media files and their corresponding sidecar JSONs.

        Args:
            photos_root: Root directory of Google Photos export.

        Yields:
            Tuples of (media_file_path, sidecar_json_path).
            sidecar_json_path is None if no sidecar found.

        Notes:
            - Uses generator to handle large exports efficiently
            - Skips JSON files (they're sidecars, not media)
            - Skips metadata.json files (album metadata)
        """
        for file_path in photos_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip JSON files themselves
            if file_path.suffix.lower() == ".json":
                continue

            # Check if it's a media file
            if file_path.suffix.lower() not in self.supported_extensions:
                continue

            # Find corresponding sidecar
            sidecar = self._find_sidecar(file_path)

            yield (file_path, sidecar)

    def _find_sidecar(self, media_path: Path) -> Optional[Path]:
        """Find JSON sidecar for a media file.

        Args:
            media_path: Path to media file.

        Returns:
            Path to sidecar JSON, or None if not found.

        Tries multiple naming patterns:
            1. IMG_1234.jpg → IMG_1234.jpg.json
            2. IMG_1234.jpg → IMG_1234.json
            3. Case variations
        """
        parent = media_path.parent
        name = media_path.name

        # Try: filename + .json (e.g., IMG_1234.jpg.json)
        sidecar1 = parent / f"{name}.json"
        if sidecar1.exists():
            return sidecar1

        # Try: stem + .json (e.g., IMG_1234.json)
        stem = media_path.stem
        sidecar2 = parent / f"{stem}.json"
        if sidecar2.exists():
            return sidecar2

        # Try case-insensitive search (for some exports)
        for item in parent.iterdir():
            if item.suffix.lower() == ".json":
                # Check if this JSON is for our media file
                json_stem = item.stem
                if json_stem.lower() == name.lower() or json_stem.lower() == stem.lower():
                    return item

        return None

    def _parse_sidecar(self, sidecar_path: Path) -> Optional[Dict[str, Any]]:
        """Parse and validate sidecar JSON.

        Args:
            sidecar_path: Path to sidecar JSON file.

        Returns:
            Parsed JSON as dictionary, or None if invalid.

        Validation:
            Must have at least one of:
            - "photoTakenTime" or "creationTime"
            - "title"
        """
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate it's a Google Photos sidecar
            if "photoTakenTime" in data or "creationTime" in data or "title" in data:
                return data

            logger.debug(f"JSON {sidecar_path} doesn't appear to be a Google Photos sidecar")
            return None

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {sidecar_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading {sidecar_path}: {e}")
            return None

    def _extract_datetime_from_sidecar(
        self, sidecar: Dict[str, Any]
    ) -> Tuple[Optional[datetime], ConfidenceLevel]:
        """Extract best timestamp from sidecar.

        Args:
            sidecar: Parsed sidecar JSON.

        Returns:
            Tuple of (datetime, confidence_level).

        Preference order:
            1. photoTakenTime (when photo was actually taken) → HIGH confidence
            2. creationTime (when added to Google Photos) → MEDIUM confidence
        """
        # Try photoTakenTime first (most reliable)
        photo_taken = sidecar.get("photoTakenTime")
        if photo_taken:
            dt = self._parse_google_timestamp(photo_taken)
            if dt:
                return (dt, ConfidenceLevel.HIGH)

        # Fall back to creationTime
        creation = sidecar.get("creationTime")
        if creation:
            dt = self._parse_google_timestamp(creation)
            if dt:
                return (dt, ConfidenceLevel.MEDIUM)

        return (None, ConfidenceLevel.LOW)

    def _extract_location_from_sidecar(self, sidecar: Dict[str, Any]) -> Optional[Location]:
        """Extract location from sidecar.

        Args:
            sidecar: Parsed sidecar JSON.

        Returns:
            Location object, or None if no valid location data.

        Preference order:
            1. geoDataExif (from original EXIF) → HIGH confidence
            2. geoData (might be Google-inferred) → MEDIUM confidence

        Handles:
            - lat/lon of 0, 0 (invalid, returns None)
            - Missing altitude (acceptable)
        """
        # Try geoDataExif first (from EXIF, most reliable)
        geo_exif = sidecar.get("geoDataExif")
        if geo_exif and isinstance(geo_exif, dict):
            lat = geo_exif.get("latitude")
            lon = geo_exif.get("longitude")

            if lat is not None and lon is not None:
                # Check for invalid 0,0 coordinates
                if lat == 0.0 and lon == 0.0:
                    pass  # Invalid, try next
                else:
                    altitude = geo_exif.get("altitude")
                    geopoint = GeoPoint(
                        latitude=float(lat),
                        longitude=float(lon),
                        altitude=float(altitude) if altitude else None,
                    )
                    return Location(geo_point=geopoint, confidence=ConfidenceLevel.HIGH)

        # Try geoData (might be inferred)
        geo_data = sidecar.get("geoData")
        if geo_data and isinstance(geo_data, dict):
            lat = geo_data.get("latitude")
            lon = geo_data.get("longitude")

            if lat is not None and lon is not None:
                # Check for invalid 0,0 coordinates
                if lat == 0.0 and lon == 0.0:
                    return None

                altitude = geo_data.get("altitude")
                geopoint = GeoPoint(
                    latitude=float(lat),
                    longitude=float(lon),
                    altitude=float(altitude) if altitude else None,
                )
                return Location(geo_point=geopoint, confidence=ConfidenceLevel.MEDIUM)

        return None

    def _extract_people_from_sidecar(self, sidecar: Dict[str, Any]) -> List[PersonTag]:
        """Extract people tags from sidecar.

        Args:
            sidecar: Parsed sidecar JSON.

        Returns:
            List of PersonTag objects from Google's face recognition.

        Notes:
            Google's face recognition is highly reliable, so confidence is HIGH.
        """
        people_tags: List[PersonTag] = []

        people_list = sidecar.get("people", [])
        if not isinstance(people_list, list):
            return people_tags

        for person in people_list:
            if isinstance(person, dict):
                name = person.get("name")
                if name:
                    people_tags.append(PersonTag(name=name, confidence=ConfidenceLevel.HIGH))

        return people_tags

    def _extract_album_from_path(self, media_path: Path) -> Optional[str]:
        """Determine album name from file path.

        Args:
            media_path: Path to media file.

        Returns:
            Album name, or None if not in an album.

        Patterns:
            - "Album - Summer Vacation 2019/..." → "Summer Vacation 2019"
            - "Photos from 2019/..." → None (not a real album)
            - "Untitled/..." → None
        """
        # Check parent directory name
        parent_name = media_path.parent.name

        # Pattern: "Album - <album name>"
        album_match = re.match(r"Album - (.+)", parent_name)
        if album_match:
            return album_match.group(1)

        # Skip generic folders
        if parent_name.startswith("Photos from"):
            return None

        if parent_name.lower() in ["untitled", "archive", "trash"]:
            return None

        # If it's not a generic folder and not the root, consider it an album
        if parent_name != "Google Photos" and parent_name != "Takeout":
            return parent_name

        return None

    def _parse_google_timestamp(self, ts_data: Dict[str, Any]) -> Optional[datetime]:
        """Parse Google's timestamp object.

        Args:
            ts_data: Timestamp dictionary from sidecar.

        Returns:
            Timezone-aware datetime object, or None if unparseable.

        Format:
            {"timestamp": "1560610222", "formatted": "Jun 15, 2019, 2:30:22 PM UTC"}

        Prefers timestamp (Unix epoch) as it's more reliable.
        """
        if not isinstance(ts_data, dict):
            return None

        # Try timestamp field (Unix epoch)
        timestamp_str = ts_data.get("timestamp")
        if timestamp_str:
            try:
                # Handle both string and int
                if isinstance(timestamp_str, str):
                    timestamp = int(timestamp_str)
                else:
                    timestamp = timestamp_str

                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to parse timestamp {timestamp_str}: {e}")

        # Fall back to formatted string
        formatted_str = ts_data.get("formatted")
        if formatted_str:
            try:
                # Try common formats
                # "Jun 15, 2019, 2:30:22 PM UTC"
                dt = datetime.strptime(formatted_str, "%b %d, %Y, %I:%M:%S %p %Z")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

            try:
                # Try without timezone
                dt = datetime.strptime(formatted_str, "%b %d, %Y, %I:%M:%S %p")
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.warning(f"Could not parse formatted timestamp: {formatted_str}")

        return None

    def _create_memory_from_pair(
        self, media_path: Path, sidecar: Optional[Dict[str, Any]]
    ) -> Memory:
        """Create Memory from media file and optional sidecar.

        Args:
            media_path: Path to media file.
            sidecar: Parsed sidecar JSON, or None if not available.

        Returns:
            Memory object with all extractable metadata.

        Behavior:
            If sidecar available:
                - Use sidecar for timestamp, location, people
                - Use sidecar description for caption
            If no sidecar:
                - Fall back to file timestamps
                - Mark with MEDIUM/LOW confidence
        """
        # Determine media type from extension
        ext = media_path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp"}:
            media_type = MediaType.PHOTO
        elif ext in {".mp4", ".mov"}:
            media_type = MediaType.VIDEO
        else:
            media_type = MediaType.UNKNOWN

        # Extract metadata from sidecar if available
        created_at = None
        created_at_confidence = ConfidenceLevel.LOW
        location = None
        people: List[PersonTag] = []
        caption = None

        if sidecar:
            # Extract timestamp
            created_at, created_at_confidence = self._extract_datetime_from_sidecar(sidecar)

            # Extract location
            location = self._extract_location_from_sidecar(sidecar)

            # Extract people
            people = self._extract_people_from_sidecar(sidecar)

            # Extract description as caption
            caption = sidecar.get("description")

            # Check for screenshot type
            origin = sidecar.get("googlePhotosOrigin", {})
            if "screenshot" in str(origin).lower():
                media_type = MediaType.SCREENSHOT

        # Fall back to file timestamp if no sidecar timestamp
        if not created_at:
            try:
                file_mtime = media_path.stat().st_mtime
                created_at = datetime.fromtimestamp(file_mtime, tz=timezone.utc)
                created_at_confidence = ConfidenceLevel.LOW
            except Exception as e:
                logger.warning(f"Could not get file timestamp for {media_path}: {e}")

        # Extract album name from path
        album_name = self._extract_album_from_path(media_path)

        # Create Memory
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=media_type,
            created_at=created_at,
            created_at_confidence=created_at_confidence,
            source_path=str(media_path),
            location=location,
            people=people,
            caption=caption,
            album_name=album_name,
            original_metadata=sidecar or {},
        )

        return memory

    def _pair_live_photos(self, memories: List[Memory]) -> List[Memory]:
        """Detect and pair Live Photos (.jpg + .mp4 with same timestamp).

        Args:
            memories: List of Memory objects.

        Returns:
            List with Live Photos paired into single Memory objects.

        Live Photos:
            - iPhone Live Photos export as .jpg + .mp4
            - Same basename, same timestamp (within 2 seconds)
            - Create single Memory with media_type = LIVE_PHOTO
        """
        # Group by timestamp and basename
        timestamp_groups: Dict[Tuple[Optional[datetime], str], List[Memory]] = {}

        for memory in memories:
            if memory.created_at and memory.source_path:
                basename = Path(memory.source_path).stem
                # Remove common suffixes like -edited, (1), etc.
                basename = re.sub(r"[-_]edited|\(\d+\)", "", basename)

                key = (memory.created_at, basename.lower())
                if key not in timestamp_groups:
                    timestamp_groups[key] = []
                timestamp_groups[key].append(memory)

        # Find Live Photo pairs
        result: List[Memory] = []
        processed: set[int] = set()

        for group in timestamp_groups.values():
            if len(group) >= 2:
                # Check if we have a photo + video pair
                photos = [m for m in group if m.media_type == MediaType.PHOTO]
                videos = [m for m in group if m.media_type == MediaType.VIDEO]

                if photos and videos:
                    # Pair them as Live Photo
                    photo = photos[0]
                    video = videos[0]

                    # Create Live Photo memory
                    live_memory = Memory(
                        source_platform=photo.source_platform,
                        media_type=MediaType.LIVE_PHOTO,
                        created_at=photo.created_at,
                        created_at_confidence=photo.created_at_confidence,
                        source_path=photo.source_path,
                        location=photo.location,
                        people=photo.people,
                        caption=photo.caption,
                        album_name=photo.album_name,
                        original_metadata={
                            "live_photo": True,
                            "photo_path": photo.source_path,
                            "video_path": video.source_path,
                        },
                    )

                    result.append(live_memory)
                    processed.add(id(photo))
                    processed.add(id(video))

                    # Add remaining items from group
                    for mem in group[2:]:
                        if id(mem) not in processed:
                            result.append(mem)
                            processed.add(id(mem))
                    continue

            # Not a Live Photo, add all from group
            for mem in group:
                if id(mem) not in processed:
                    result.append(mem)
                    processed.add(id(mem))

        # Add memories that weren't in any timestamp group
        for memory in memories:
            if id(memory) not in processed:
                result.append(memory)

        return result

    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories based on timestamp and source path.

        Args:
            memories: List of Memory objects potentially containing duplicates.

        Returns:
            Deduplicated list of Memory objects.

        Deduplication logic:
            Same photo might appear in multiple albums. We use source_path
            as the primary key, keeping the one with the most complete metadata.
        """
        seen: Dict[str, Memory] = {}

        for memory in memories:
            if not memory.source_path:
                # Can't deduplicate without path, keep it
                continue

            key = memory.source_path

            if key not in seen:
                seen[key] = memory
            else:
                # Keep the one with more complete metadata
                existing = seen[key]

                # Prefer one with people tags
                if memory.people and not existing.people:
                    seen[key] = memory
                # Prefer one with location
                elif memory.location and not existing.location:
                    seen[key] = memory
                # Prefer one with caption
                elif memory.caption and not existing.caption:
                    seen[key] = memory

        # Add back memories without source_path
        result = list(seen.values())
        for memory in memories:
            if not memory.source_path:
                result.append(memory)

        return result
