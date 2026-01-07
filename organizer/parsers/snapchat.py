"""Snapchat export parser for Digital Life Narrative AI.

Handles Snapchat data exports from the "Download My Data" feature.
Parses memories, chat history, stories, and various JSON metadata files.

Snapchat export structure:
    snapchat_export/
    ├── memories_history.json
    ├── chat_history/
    │   └── <friend_name>/
    │       ├── messages.json
    │       └── <media files>
    ├── memories/
    │   └── <media files>
    ├── snap_history/
    ├── account.json
    └── location_history.json
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

# Snapchat-specific datetime formats
SNAPCHAT_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S UTC",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
]

# Media type mapping from Snapchat terminology
SNAPCHAT_MEDIA_TYPE_MAP = {
    "image": MediaType.PHOTO,
    "photo": MediaType.PHOTO,
    "video": MediaType.VIDEO,
    "snap": MediaType.STORY,  # Snaps are ephemeral, similar to stories
}


# =============================================================================
# Snapchat Parser
# =============================================================================


@ParserRegistry.register
class SnapchatParser(BaseParser):
    """Parser for Snapchat data exports.

    Handles both old and new Snapchat export formats, extracting:
    - Memories (saved photos/videos)
    - Chat history with shared media
    - Snap history (sent/received snaps metadata)
    - Location data when available

    Attributes:
        platform: SourcePlatform.SNAPCHAT
        supported_extensions: Set of file extensions to process

    Example:
        ```python
        parser = SnapchatParser()
        if parser.can_parse(export_path):
            result = parser.parse(export_path)
            print(f"Found {len(result.items)} media items")
        ```
    """

    platform = SourcePlatform.SNAPCHAT
    supported_extensions = {".json", ".jpg", ".jpeg", ".mp4", ".png", ".mov"}

    def __init__(self) -> None:
        """Initialize the Snapchat parser."""
        super().__init__()
        self._location_cache: dict[str, GeoLocation] = {}

    def can_parse(self, root_path: Path) -> bool:
        """Check if this parser can handle the given path.

        Looks for Snapchat-specific files and folders.

        Args:
            root_path: Path to check.

        Returns:
            True if Snapchat export detected.
        """
        if not root_path.exists() or not root_path.is_dir():
            return False

        # Check for key Snapchat files/folders
        indicators = [
            root_path / "memories_history.json",
            root_path / "memories",
            root_path / "chat_history",
            root_path / "snap_history",
            root_path / "account.json",
        ]

        return any(indicator.exists() for indicator in indicators)

    def parse(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParseResult:
        """Parse Snapchat export and extract all media items.

        Args:
            root_path: Root directory of the Snapchat export.
            progress_callback: Optional callback for progress updates.

        Returns:
            ParseResult containing all parsed items.
        """
        start_time = time.time()
        all_items: list[MediaItem] = []

        logger.info(f"Starting Snapchat parse: {root_path}")

        # Reset state
        self._errors = []
        self._stats = {"total_files": 0, "parsed": 0, "skipped": 0, "errors": 0}

        # Estimate total for progress
        estimated_total = self._estimate_total_files(root_path)

        # Load location history if available
        location_path = root_path / "location_history.json"
        if location_path.exists():
            self._load_location_history(location_path)

        # Parse memories
        memories_path = root_path / "memories_history.json"
        if memories_path.exists():
            logger.debug("Parsing memories_history.json")
            memories_items = self._parse_memories_history(
                memories_path,
                root_path / "memories",
            )
            all_items.extend(memories_items)

            if progress_callback:
                progress_callback(len(all_items), estimated_total)

        # Parse chat history
        chat_path = root_path / "chat_history"
        if chat_path.exists() and chat_path.is_dir():
            logger.debug("Parsing chat_history/")
            chat_items = self._parse_chat_history(chat_path)
            all_items.extend(chat_items)

            if progress_callback:
                progress_callback(len(all_items), estimated_total)

        # Parse snap history
        snap_history_path = root_path / "snap_history"
        if snap_history_path.exists():
            logger.debug("Parsing snap_history/")
            snap_items = self._parse_snap_history(snap_history_path)
            all_items.extend(snap_items)

            if progress_callback:
                progress_callback(len(all_items), estimated_total)

        # Parse standalone memories folder if no JSON
        if not memories_path.exists():
            memories_dir = root_path / "memories"
            if memories_dir.exists():
                logger.debug("Parsing memories/ folder directly")
                media_items = self._parse_media_folder(memories_dir)
                all_items.extend(media_items)

        duration = time.time() - start_time
        logger.info(f"Snapchat parse complete: {len(all_items)} items in {duration:.2f}s")

        return self._create_parse_result(all_items, duration)

    # =========================================================================
    # Private Parsing Methods
    # =========================================================================

    def _estimate_total_files(self, root_path: Path) -> int:
        """Estimate total number of files to parse.

        Args:
            root_path: Root export directory.

        Returns:
            Estimated file count.
        """
        count = 0
        try:
            for ext in self.supported_extensions:
                count += len(list(root_path.rglob(f"*{ext}")))
        except Exception:
            count = 100  # Fallback estimate
        return max(count, 1)

    def _load_location_history(self, location_path: Path) -> None:
        """Load location history into cache.

        Snapchat location history can be used to enrich memories
        with location data.

        Args:
            location_path: Path to location_history.json.
        """
        try:
            with open(location_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Location history may have various formats
            locations = data if isinstance(data, list) else data.get("Location History", [])

            for entry in locations:
                if isinstance(entry, dict):
                    timestamp = entry.get("Time", entry.get("timestamp", ""))
                    if timestamp:
                        location = self._normalize_location(entry)
                        if location:
                            self._location_cache[timestamp] = location

            logger.debug(f"Loaded {len(self._location_cache)} location entries")

        except Exception as e:
            logger.warning(f"Failed to load location history: {e}")

    def _parse_memories_history(
        self,
        json_path: Path,
        memories_dir: Path,
    ) -> list[MediaItem]:
        """Parse memories_history.json and match to media files.

        Snapchat memories JSON typically contains:
        - "Saved Media" array with timestamp, media type, location

        Args:
            json_path: Path to memories_history.json.
            memories_dir: Path to memories/ folder with actual files.

        Returns:
            List of parsed MediaItem objects.
        """
        items: list[MediaItem] = []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log_error(f"Failed to parse memories_history.json: {e}")
            return items

        # Handle different JSON structures
        saved_media = self._extract_saved_media(data)
        self._stats["total_files"] += len(saved_media)

        logger.debug(f"Found {len(saved_media)} entries in memories_history.json")

        # Get list of actual media files
        media_files: list[Path] = []
        if memories_dir.exists():
            media_files = [
                f
                for f in memories_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.supported_extensions
            ]

        # Process each memory entry
        for entry in saved_media:
            try:
                item = self._parse_memory_entry(entry, media_files)
                if item:
                    items.append(item)
            except Exception as e:
                self._log_error(f"Failed to parse memory entry: {e}")

        return items

    def _extract_saved_media(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract saved media array from various JSON structures.

        Args:
            data: Parsed JSON data.

        Returns:
            List of media entries.
        """
        # Try different known structures
        if isinstance(data, list):
            return data

        for key in ["Saved Media", "saved_media", "memories", "Memories"]:
            if key in data:
                result = data[key]
                if isinstance(result, list):
                    return result

        # Fallback: look for any list with media-like entries
        for value in data.values():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # Check if it looks like media entries
                    first = value[0]
                    if "Date" in first or "Media Type" in first or "timestamp" in first:
                        return value

        return []

    def _parse_memory_entry(
        self,
        entry: dict[str, Any],
        available_files: list[Path],
    ) -> MediaItem | None:
        """Parse a single memory entry into MediaItem.

        Args:
            entry: Memory entry from JSON.
            available_files: List of available media files to match.

        Returns:
            MediaItem or None if parsing fails.
        """
        # Extract timestamp
        timestamp = None
        timestamp_confidence = Confidence.LOW
        date_str = entry.get("Date", entry.get("date", entry.get("timestamp", "")))

        if date_str:
            timestamp = self._parse_snapchat_datetime(date_str)
            if timestamp:
                timestamp_confidence = Confidence.HIGH

        # Determine media type
        media_type_str = entry.get("Media Type", entry.get("media_type", "")).lower()
        media_type = SNAPCHAT_MEDIA_TYPE_MAP.get(media_type_str, MediaType.PHOTO)

        # Check for download link vs local file
        download_link = entry.get("Download Link", entry.get("download_link"))
        file_path = None
        file_hash = None

        if download_link:
            # Memory might be link-only (note in metadata)
            logger.debug("Memory has download link, no local file")
        else:
            # Try to match to a local file
            matched_file = self._match_media_file(entry, available_files, timestamp)
            if matched_file:
                file_path = matched_file
                try:
                    file_hash = self._calculate_file_hash(matched_file)
                except Exception:
                    pass

        # Extract location
        location = None
        location_confidence = Confidence.LOW
        raw_location = entry.get("Location", entry.get("location"))
        if raw_location:
            location = self._normalize_location(raw_location)
            if location:
                location_confidence = Confidence.MEDIUM

        # Check location cache by timestamp
        if not location and date_str and date_str in self._location_cache:
            location = self._location_cache[date_str]
            location_confidence = Confidence.LOW

        # Create MediaItem
        return MediaItem(
            id=self._generate_item_id(),
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=media_type,
            file_path=file_path,
            timestamp=timestamp,
            timestamp_confidence=timestamp_confidence,
            location=location,
            location_confidence=location_confidence,
            people=[],
            caption=entry.get("Caption", entry.get("caption")),
            original_metadata=entry,
            file_hash=file_hash,
        )

    def _match_media_file(
        self,
        entry: dict[str, Any],
        available_files: list[Path],
        target_timestamp: datetime | None,
    ) -> Path | None:
        """Match a memory entry to an actual media file.

        Snapchat media files often have UUID-based names.
        Matches by:
        1. Direct filename reference in entry
        2. Timestamp proximity
        3. File modification date

        Args:
            entry: Memory entry from JSON.
            available_files: List of available files.
            target_timestamp: Target timestamp to match.

        Returns:
            Matched file path or None.
        """
        if not available_files:
            return None

        # Check for explicit filename in entry
        for key in ["File", "filename", "file_name", "media_id"]:
            if key in entry:
                filename = entry[key]
                for f in available_files:
                    if f.name == filename or f.stem == filename:
                        return f

        # If we have a timestamp, try to match by file dates
        if target_timestamp:
            best_match = None
            best_diff = float("inf")

            for f in available_files:
                try:
                    file_mtime = datetime.fromtimestamp(
                        f.stat().st_mtime,
                        tz=timezone.utc,
                    )
                    diff = abs((file_mtime - target_timestamp).total_seconds())

                    if diff < best_diff and diff < 86400:  # Within 24 hours
                        best_diff = diff
                        best_match = f
                except Exception:
                    continue

            if best_match:
                return best_match

        return None

    def _parse_chat_history(self, chat_dir: Path) -> list[MediaItem]:
        """Parse chat history folder for messages and shared media.

        Args:
            chat_dir: Path to chat_history/ folder.

        Returns:
            List of MediaItem objects from chats.
        """
        items: list[MediaItem] = []

        try:
            # Each subfolder is a conversation
            for conv_folder in chat_dir.iterdir():
                if not conv_folder.is_dir():
                    continue

                # Extract friend/group name from folder name
                participant_name = self._clean_participant_name(conv_folder.name)

                # Parse messages.json if present
                messages_path = conv_folder / "messages.json"
                if messages_path.exists():
                    chat_items = self._parse_chat_messages(
                        messages_path,
                        participant_name,
                        conv_folder,
                    )
                    items.extend(chat_items)

                # Look for media files in the conversation folder
                media_items = self._parse_chat_media(conv_folder, participant_name)
                items.extend(media_items)

        except Exception as e:
            self._log_error(f"Failed to parse chat history: {e}")

        return items

    def _clean_participant_name(self, folder_name: str) -> str:
        """Clean up participant name from folder name.

        Args:
            folder_name: Raw folder name.

        Returns:
            Cleaned participant name.
        """
        # Remove common suffixes/prefixes
        name = folder_name.strip()

        # Remove UUID-like suffixes
        name = re.sub(r"_[a-f0-9]{8,}$", "", name, flags=re.IGNORECASE)

        # Replace underscores with spaces
        name = name.replace("_", " ")

        return name.title()

    def _parse_chat_messages(
        self,
        messages_path: Path,
        participant: str,
        conv_folder: Path,
    ) -> list[MediaItem]:
        """Parse messages.json from a chat conversation.

        Args:
            messages_path: Path to messages.json.
            participant: Name of chat participant.
            conv_folder: Conversation folder path.

        Returns:
            List of MediaItem objects from messages.
        """
        items: list[MediaItem] = []

        try:
            with open(messages_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log_error(f"Failed to parse {messages_path}: {e}")
            return items

        # Get messages array
        messages = data if isinstance(data, list) else data.get("messages", [])
        self._stats["total_files"] += len(messages)

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Check if message has media
            has_media = any(
                key in msg for key in ["Media Type", "media_type", "media", "attachment"]
            )

            if not has_media:
                continue

            # Extract message data
            timestamp = None
            date_str = msg.get("Date", msg.get("date", msg.get("timestamp", "")))
            if date_str:
                timestamp = self._parse_snapchat_datetime(date_str)

            media_type_str = msg.get("Media Type", msg.get("media_type", "")).lower()
            media_type = SNAPCHAT_MEDIA_TYPE_MAP.get(media_type_str, MediaType.MESSAGE)

            # Get sender
            sender = msg.get("From", msg.get("sender", participant))

            item = MediaItem(
                id=self._generate_item_id(),
                source_platform=SourcePlatform.SNAPCHAT,
                media_type=media_type,
                file_path=None,
                timestamp=timestamp,
                timestamp_confidence=Confidence.HIGH if timestamp else Confidence.LOW,
                location=None,
                location_confidence=Confidence.LOW,
                people=[sender] if sender else [participant],
                caption=msg.get("Content", msg.get("content", msg.get("text"))),
                original_metadata=msg,
                file_hash=None,
            )
            items.append(item)

        return items

    def _parse_chat_media(
        self,
        conv_folder: Path,
        participant: str,
    ) -> list[MediaItem]:
        """Parse media files directly in a chat folder.

        Args:
            conv_folder: Conversation folder path.
            participant: Participant name for tagging.

        Returns:
            List of MediaItem objects.
        """
        items: list[MediaItem] = []
        media_extensions = {".jpg", ".jpeg", ".png", ".mp4", ".mov"}

        try:
            for file_path in conv_folder.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in media_extensions:
                    continue
                if file_path.name == "messages.json":
                    continue

                self._stats["total_files"] += 1

                # Determine media type from extension
                ext = file_path.suffix.lower()
                if ext in {".jpg", ".jpeg", ".png"}:
                    media_type = MediaType.PHOTO
                elif ext in {".mp4", ".mov"}:
                    media_type = MediaType.VIDEO
                else:
                    media_type = MediaType.UNKNOWN

                # Try to get timestamp
                timestamp = self._extract_exif_datetime(file_path)
                if not timestamp:
                    timestamp = self._extract_datetime_from_filename(file_path.name)
                timestamp_confidence = Confidence.MEDIUM if timestamp else Confidence.LOW

                # Calculate hash
                file_hash = None
                try:
                    file_hash = self._calculate_file_hash(file_path)
                except Exception:
                    pass

                item = MediaItem(
                    id=self._generate_item_id(),
                    source_platform=SourcePlatform.SNAPCHAT,
                    media_type=media_type,
                    file_path=file_path,
                    timestamp=timestamp,
                    timestamp_confidence=timestamp_confidence,
                    location=None,
                    location_confidence=Confidence.LOW,
                    people=[participant],
                    caption=None,
                    original_metadata={"source": "chat_media", "participant": participant},
                    file_hash=file_hash,
                )
                items.append(item)

        except Exception as e:
            self._log_error(f"Failed to parse chat media in {conv_folder}: {e}")

        return items

    def _parse_snap_history(self, snap_history_dir: Path) -> list[MediaItem]:
        """Parse snap_history folder for sent/received snaps.

        Args:
            snap_history_dir: Path to snap_history/ folder.

        Returns:
            List of MediaItem objects.
        """
        items: list[MediaItem] = []

        # Look for snap_history.json
        json_path = snap_history_dir / "snap_history.json"
        if not json_path.exists():
            # Try alternate location
            json_path = snap_history_dir
            if json_path.suffix != ".json":
                return items

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._log_error(f"Failed to parse snap_history: {e}", json_path)
            return items

        # Get snaps array
        snaps = data if isinstance(data, list) else data.get("Received Snap History", [])
        snaps.extend(data.get("Sent Snap History", []) if isinstance(data, dict) else [])

        self._stats["total_files"] += len(snaps)

        for snap in snaps:
            if not isinstance(snap, dict):
                continue

            # Extract snap data
            timestamp = None
            date_str = snap.get("Date", snap.get("timestamp", ""))
            if date_str:
                timestamp = self._parse_snapchat_datetime(date_str)

            media_type_str = snap.get("Media Type", snap.get("type", "")).lower()
            media_type = SNAPCHAT_MEDIA_TYPE_MAP.get(media_type_str, MediaType.STORY)

            # Determine if sent or received
            recipient = snap.get("To", snap.get("recipient"))
            sender = snap.get("From", snap.get("sender"))
            people = []
            if recipient:
                people.append(recipient)
            if sender:
                people.append(sender)

            item = MediaItem(
                id=self._generate_item_id(),
                source_platform=SourcePlatform.SNAPCHAT,
                media_type=media_type,
                file_path=None,  # Snap history typically doesn't have files
                timestamp=timestamp,
                timestamp_confidence=Confidence.HIGH if timestamp else Confidence.LOW,
                location=None,
                location_confidence=Confidence.LOW,
                people=people,
                caption=None,
                original_metadata=snap,
                file_hash=None,
            )
            items.append(item)

        return items

    def _parse_media_folder(self, media_dir: Path) -> list[MediaItem]:
        """Parse a folder of media files without JSON metadata.

        Fallback for when memories_history.json is missing.

        Args:
            media_dir: Path to folder containing media files.

        Returns:
            List of MediaItem objects.
        """
        items: list[MediaItem] = []
        media_extensions = {".jpg", ".jpeg", ".png", ".mp4", ".mov"}

        try:
            for file_path in media_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in media_extensions:
                    continue

                self._stats["total_files"] += 1

                # Determine media type
                ext = file_path.suffix.lower()
                if ext in {".jpg", ".jpeg", ".png"}:
                    media_type = MediaType.PHOTO
                elif ext in {".mp4", ".mov"}:
                    media_type = MediaType.VIDEO
                else:
                    media_type = MediaType.UNKNOWN

                # Extract timestamp
                timestamp = self._extract_exif_datetime(file_path)
                timestamp_confidence = Confidence.HIGH

                if not timestamp:
                    timestamp = self._extract_datetime_from_filename(file_path.name)
                    timestamp_confidence = Confidence.MEDIUM

                if not timestamp:
                    # Use file modification time as fallback
                    try:
                        timestamp = datetime.fromtimestamp(
                            file_path.stat().st_mtime,
                            tz=timezone.utc,
                        )
                        timestamp_confidence = Confidence.LOW
                    except Exception:
                        pass

                # Calculate hash
                file_hash = None
                try:
                    file_hash = self._calculate_file_hash(file_path)
                except Exception:
                    pass

                item = MediaItem(
                    id=self._generate_item_id(),
                    source_platform=SourcePlatform.SNAPCHAT,
                    media_type=media_type,
                    file_path=file_path,
                    timestamp=timestamp,
                    timestamp_confidence=timestamp_confidence,
                    location=None,
                    location_confidence=Confidence.LOW,
                    people=[],
                    caption=None,
                    original_metadata={"source": "memories_folder"},
                    file_hash=file_hash,
                )
                items.append(item)

        except Exception as e:
            self._log_error(f"Failed to parse media folder {media_dir}: {e}")

        return items

    def _parse_snapchat_datetime(self, date_str: str) -> datetime | None:
        """Parse Snapchat-specific datetime formats.

        Args:
            date_str: Date string from Snapchat export.

        Returns:
            Parsed datetime or None.
        """
        if not date_str:
            return None

        # Clean up the string
        date_str = date_str.strip()

        # Try Snapchat-specific formats first
        for fmt in SNAPCHAT_DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue

        # Fall back to generic parser
        return self._safe_parse_datetime(date_str)
