"""Snapchat parser for Digital Life Narrative AI.

This module provides a concrete parser that converts Snapchat "Download My Data"
exports into normalized Memory objects. It handles the specific structure and quirks
of Snapchat's export format including saved memories, chat history, snap history,
and location data.

Typical usage:
    >>> from pathlib import Path
    >>> from src.parsers.snapchat import SnapchatParser
    >>> 
    >>> parser = SnapchatParser()
    >>> if parser.can_parse(Path("/exports/snapchat")):
    ...     result = parser.parse(Path("/exports/snapchat"))
    ...     print(f"Parsed {len(result.memories)} memories")
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from src.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
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


@register_parser
class SnapchatParser(BaseParser):
    """Parser for Snapchat 'Download My Data' exports.
    
    Handles the specific structure of Snapchat exports including:
    - Saved memories (photos/videos)
    - Chat history with media
    - Snap history (ephemeral snaps metadata)
    - Location history
    
    Attributes:
        platform: Source platform (SNAPCHAT).
        version: Parser version.
        supported_extensions: File extensions this parser handles.
        description: Human-readable parser description.
    """
    
    platform: ClassVar[SourcePlatform] = SourcePlatform.SNAPCHAT
    version: ClassVar[str] = "1.0.0"
    supported_extensions: ClassVar[set[str]] = {
        ".json", ".jpg", ".jpeg", ".png", ".mp4", ".mov"
    }
    description: ClassVar[str] = "Parser for Snapchat 'Download My Data' exports"
    
    def can_parse(self, root: Path) -> bool:
        """Check if this parser can handle the given directory.
        
        Args:
            root: Directory to check.
            
        Returns:
            True if directory appears to be a Snapchat export, False otherwise.
            
        Detection criteria:
            - memories_history.json exists, OR
            - Both chat_history/ and memories/ directories exist, OR
            - account.json exists with Snapchat-specific structure
        """
        # Check for memories_history.json
        if (root / "memories_history.json").exists():
            return True
        
        # Check for both chat_history and memories directories
        if (root / "chat_history").is_dir() and (root / "memories").is_dir():
            return True
        
        # Check for account.json with Snapchat structure
        account_file = root / "account.json"
        if account_file.exists():
            try:
                with open(account_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Look for Snapchat-specific fields
                    if "Username" in data or "Snapcode" in data:
                        return True
            except (json.JSONDecodeError, IOError):
                pass
        
        return False
    
    def get_signature_files(self) -> List[str]:
        """Get list of signature files/directories for Snapchat exports.
        
        Returns:
            List of file/directory names that indicate a Snapchat export.
        """
        return [
            "memories_history.json",
            "chat_history/",
            "memories/",
            "account.json",
            "snap_history/"
        ]
    
    def parse(self, root: Path, progress: ProgressCallback | None = None) -> ParseResult:
        """Parse Snapchat export into Memory objects.
        
        Args:
            root: Root directory of the Snapchat export.
            progress: Optional callback for progress reporting.
            
        Returns:
            ParseResult containing extracted memories, warnings, and errors.
            
        Processing steps:
            1. Scan export structure
            2. Load location history for enrichment
            3. Parse memories_history.json
            4. Parse chat_history/
            5. Parse snap_history.json
            6. Enrich with location data
            7. Deduplicate memories
            8. Return comprehensive result
        """
        memories: List[Memory] = []
        warnings: List[ParseWarning] = []
        errors: List[ParseError] = []
        
        def report_progress(message: str, current: int = 0, total: int = 0):
            """Helper to report progress if callback provided."""
            if progress:
                progress(ParseProgress(
                    message=message,
                    current=current,
                    total=total,
                    platform=SourcePlatform.SNAPCHAT
                ))
        
        report_progress("Scanning Snapchat export...")
        logger.info(f"Starting Snapchat parse of {root}")
        
        # Load location history for enrichment
        location_lookup: Dict[datetime, GeoPoint] = {}
        location_file = root / "location_history.json"
        if location_file.exists():
            try:
                location_lookup = self._parse_location_history(location_file)
                logger.info(f"Loaded {len(location_lookup)} location points")
            except Exception as e:
                warnings.append(ParseWarning(
                    message=f"Failed to parse location_history.json: {e}",
                    file_path=str(location_file)
                ))
        
        # Parse memories_history.json
        memories_file = root / "memories_history.json"
        if memories_file.exists():
            report_progress("Parsing memories history...")
            try:
                memory_list = self._parse_memories_history(memories_file)
                memories.extend(memory_list)
                logger.info(f"Parsed {len(memory_list)} memories from memories_history.json")
            except Exception as e:
                errors.append(ParseError(
                    message=f"Failed to parse memories_history.json: {e}",
                    file_path=str(memories_file),
                    exception=e
                ))
        
        # Parse chat history
        chat_dir = root / "chat_history"
        if chat_dir.is_dir():
            report_progress("Parsing chat history...")
            try:
                chat_memories = self._parse_chat_history(chat_dir)
                memories.extend(chat_memories)
                logger.info(f"Parsed {len(chat_memories)} chat memories")
            except Exception as e:
                errors.append(ParseError(
                    message=f"Failed to parse chat_history: {e}",
                    file_path=str(chat_dir),
                    exception=e
                ))
        
        # Parse snap history
        snap_file = root / "snap_history" / "snap_history.json"
        if snap_file.exists():
            report_progress("Parsing snap history...")
            try:
                snap_memories = self._parse_snap_history(snap_file)
                memories.extend(snap_memories)
                logger.info(f"Parsed {len(snap_memories)} snap history entries")
            except Exception as e:
                errors.append(ParseError(
                    message=f"Failed to parse snap_history.json: {e}",
                    file_path=str(snap_file),
                    exception=e
                ))
        
        # Enrich memories with location data
        if location_lookup:
            report_progress("Enriching with location data...")
            enriched_count = 0
            for memory in memories:
                if memory.created_at and not memory.location:
                    # Find closest location point within 1 hour
                    for loc_time, geopoint in location_lookup.items():
                        time_diff = abs((memory.created_at - loc_time).total_seconds())
                        if time_diff <= 3600:  # Within 1 hour
                            memory.location = Location(
                                geo_point=geopoint,
                                confidence=ConfidenceLevel.MEDIUM
                            )
                            enriched_count += 1
                            break
            logger.info(f"Enriched {enriched_count} memories with location data")
        
        # Deduplicate memories
        report_progress("Deduplicating memories...")
        unique_memories = self._deduplicate_memories(memories)
        duplicates_removed = len(memories) - len(unique_memories)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate memories")
        
        # Build result
        status = ParseStatus.SUCCESS
        if errors:
            status = ParseStatus.PARTIAL if memories else ParseStatus.FAILED
        
        result = ParseResult(
            platform=SourcePlatform.SNAPCHAT,
            status=status,
            memories=unique_memories,
            warnings=warnings,
            errors=errors,
            statistics={
                "total_memories": len(unique_memories),
                "duplicates_removed": duplicates_removed,
                "location_points": len(location_lookup),
                "warnings_count": len(warnings),
                "errors_count": len(errors)
            }
        )
        
        report_progress(f"Completed: {len(unique_memories)} memories extracted", 
                       len(unique_memories), len(unique_memories))
        logger.info(f"Parse complete: {status.value}, {len(unique_memories)} memories")
        
        return result
    
    def _parse_memories_history(self, json_path: Path) -> List[Memory]:
        """Parse memories_history.json for saved memories.
        
        Args:
            json_path: Path to memories_history.json.
            
        Returns:
            List of Memory objects extracted from the file.
            
        Handles:
            - Missing media files (creates Memory with warning)
            - Malformed dates (skips entry with warning)
            - Missing location data (location = None)
        """
        memories: List[Memory] = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        saved_media = data.get("Saved Media", [])
        memories_dir = json_path.parent / "memories"
        
        for idx, entry in enumerate(saved_media):
            try:
                # Parse datetime
                date_str = entry.get("Date")
                if not date_str:
                    logger.warning(f"Memory entry {idx} missing Date field, skipping")
                    continue
                
                created_at = self._parse_snapchat_datetime(date_str)
                if not created_at:
                    logger.warning(f"Could not parse date '{date_str}', skipping entry {idx}")
                    continue
                
                # Map media type
                media_type_str = entry.get("Media Type", "").upper()
                if media_type_str == "IMAGE":
                    media_type = MediaType.PHOTO
                elif media_type_str == "VIDEO":
                    media_type = MediaType.VIDEO
                else:
                    media_type = MediaType.UNKNOWN
                
                # Parse location
                location = self._parse_snapchat_location(entry)
                
                # Try to find matching media file
                source_path = None
                if memories_dir.exists():
                    source_path = self._match_memory_to_file(entry, memories_dir)
                
                # Create Memory object
                memory = Memory(
                    source_platform=SourcePlatform.SNAPCHAT,
                    media_type=media_type,
                    created_at=created_at,
                    created_at_confidence=ConfidenceLevel.HIGH,
                    location=location,
                    source_path=str(source_path) if source_path else None,
                    original_metadata=entry
                )
                
                memories.append(memory)
                
                if not source_path:
                    logger.debug(f"No media file found for memory {idx}")
                
            except Exception as e:
                logger.warning(f"Error parsing memory entry {idx}: {e}", exc_info=True)
                continue
        
        return memories
    
    def _parse_chat_history(self, chat_dir: Path) -> List[Memory]:
        """Parse chat_history directory for chat conversations with media.
        
        Args:
            chat_dir: Path to chat_history directory.
            
        Returns:
            List of Memory objects from chat media.
            
        Processing:
            - Iterates through each friend's conversation folder
            - Loads messages.json from each folder
            - Creates Memory objects for messages with media attachments
            - Tags memories with friend's username
        """
        memories: List[Memory] = []
        
        if not chat_dir.is_dir():
            return memories
        
        # Iterate through friend directories
        for friend_dir in chat_dir.iterdir():
            if not friend_dir.is_dir():
                continue
            
            friend_username = friend_dir.name
            messages_file = friend_dir / "messages.json"
            
            if not messages_file.exists():
                logger.debug(f"No messages.json in {friend_dir}")
                continue
            
            try:
                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                
                for msg in messages:
                    # Only process messages with media
                    media_type_str = msg.get("Media Type", "").upper()
                    if media_type_str == "TEXT":
                        continue
                    
                    # Parse datetime
                    created_str = msg.get("Created")
                    if not created_str:
                        continue
                    
                    created_at = self._parse_snapchat_datetime(created_str)
                    if not created_at:
                        continue
                    
                    # Determine media type
                    if media_type_str == "MEDIA":
                        # Try to determine from filename
                        media_filename = msg.get("Media", "")
                        if any(ext in media_filename.lower() for ext in ['.jpg', '.jpeg', '.png']):
                            media_type = MediaType.PHOTO
                        elif any(ext in media_filename.lower() for ext in ['.mp4', '.mov']):
                            media_type = MediaType.VIDEO
                        else:
                            media_type = MediaType.PHOTO  # Default assumption
                    else:
                        media_type = MediaType.UNKNOWN
                    
                    # Find media file
                    source_path = None
                    media_filename = msg.get("Media")
                    if media_filename:
                        media_path = friend_dir / media_filename
                        if media_path.exists():
                            source_path = str(media_path)
                    
                    # Get sender/recipient info
                    from_user = msg.get("From", "")
                    people = [PersonTag(name=friend_username)]
                    
                    # Create Memory
                    memory = Memory(
                        source_platform=SourcePlatform.SNAPCHAT,
                        media_type=media_type,
                        created_at=created_at,
                        created_at_confidence=ConfidenceLevel.HIGH,
                        source_path=source_path,
                        caption=msg.get("Content"),
                        people=people,
                        thread_name=friend_username,
                        original_metadata=msg
                    )
                    
                    memories.append(memory)
                
            except Exception as e:
                logger.warning(f"Error parsing messages in {friend_dir}: {e}")
                continue
        
        return memories
    
    def _parse_snap_history(self, json_path: Path) -> List[Memory]:
        """Parse snap_history.json for ephemeral snaps metadata.
        
        Args:
            json_path: Path to snap_history.json.
            
        Returns:
            List of Memory objects for snap history (without media files).
            
        Note:
            Snap history contains metadata for ephemeral content that no longer exists.
            These are "phantom" memories valuable for timeline reconstruction.
        """
        memories: List[Memory] = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse sent snaps
        sent_snaps = data.get("Sent", [])
        for snap in sent_snaps:
            try:
                created_str = snap.get("Created")
                if not created_str:
                    continue
                
                created_at = self._parse_snapchat_datetime(created_str)
                if not created_at:
                    continue
                
                # Map media type
                media_type_str = snap.get("Media Type", "").upper()
                if media_type_str == "IMAGE":
                    media_type = MediaType.PHOTO
                elif media_type_str == "VIDEO":
                    media_type = MediaType.VIDEO
                else:
                    media_type = MediaType.UNKNOWN
                
                # Tag recipient
                recipient = snap.get("To", "")
                people = [PersonTag(name=recipient)] if recipient else []
                
                # Create phantom memory
                memory = Memory(
                    source_platform=SourcePlatform.SNAPCHAT,
                    media_type=media_type,
                    created_at=created_at,
                    created_at_confidence=ConfidenceLevel.HIGH,
                    source_path=None,  # Ephemeral, no file
                    people=people,
                    original_metadata={**snap, "snap_type": "sent", "ephemeral": True}
                )
                
                memories.append(memory)
                
            except Exception as e:
                logger.warning(f"Error parsing sent snap: {e}")
                continue
        
        # Parse received snaps
        received_snaps = data.get("Received", [])
        for snap in received_snaps:
            try:
                created_str = snap.get("Created")
                if not created_str:
                    continue
                
                created_at = self._parse_snapchat_datetime(created_str)
                if not created_at:
                    continue
                
                # Map media type
                media_type_str = snap.get("Media Type", "").upper()
                if media_type_str == "IMAGE":
                    media_type = MediaType.PHOTO
                elif media_type_str == "VIDEO":
                    media_type = MediaType.VIDEO
                else:
                    media_type = MediaType.UNKNOWN
                
                # Tag sender
                sender = snap.get("From", "")
                people = [PersonTag(name=sender)] if sender else []
                
                # Create phantom memory
                memory = Memory(
                    source_platform=SourcePlatform.SNAPCHAT,
                    media_type=media_type,
                    created_at=created_at,
                    created_at_confidence=ConfidenceLevel.HIGH,
                    source_path=None,  # Ephemeral, no file
                    people=people,
                    original_metadata={**snap, "snap_type": "received", "ephemeral": True}
                )
                
                memories.append(memory)
                
            except Exception as e:
                logger.warning(f"Error parsing received snap: {e}")
                continue
        
        return memories
    
    def _parse_location_history(self, json_path: Path) -> Dict[datetime, GeoPoint]:
        """Parse location_history.json into timestamp-to-location lookup.
        
        Args:
            json_path: Path to location_history.json.
            
        Returns:
            Dictionary mapping timestamps to GeoPoint objects.
            
        Usage:
            Used to enrich other memories with location data based on temporal proximity.
        """
        location_lookup: Dict[datetime, GeoPoint] = {}
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        location_history = data.get("Location History", [])
        
        for entry in location_history:
            try:
                time_str = entry.get("Time")
                if not time_str:
                    continue
                
                timestamp = self._parse_snapchat_datetime(time_str)
                if not timestamp:
                    continue
                
                lat_str = entry.get("Latitude")
                lon_str = entry.get("Longitude")
                if not lat_str or not lon_str:
                    continue
                
                latitude = float(lat_str)
                longitude = float(lon_str)
                
                # Parse altitude if available
                altitude = None
                alt_str = entry.get("Altitude")
                if alt_str:
                    try:
                        altitude = float(alt_str)
                    except ValueError:
                        pass
                
                geopoint = GeoPoint(
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude
                )
                
                location_lookup[timestamp] = geopoint
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing location entry: {e}")
                continue
        
        return location_lookup
    
    def _match_memory_to_file(self, memory_entry: Dict[str, Any], memories_dir: Path) -> Optional[Path]:
        """Find the media file corresponding to a memory entry.
        
        Args:
            memory_entry: Memory entry from memories_history.json.
            memories_dir: Directory containing memory media files.
            
        Returns:
            Path to matching media file, or None if not found.
            
        Matching strategies:
            1. Exact filename match if provided in entry
            2. Timestamp proximity (file creation/modification time)
            3. Pattern matching on UUID segments
        """
        # Strategy 1: Check if entry has explicit filename
        if "Download Link" in memory_entry:
            filename = Path(memory_entry["Download Link"]).name
            file_path = memories_dir / filename
            if file_path.exists():
                return file_path
        
        # Strategy 2: Try timestamp proximity
        # Get memory timestamp
        date_str = memory_entry.get("Date")
        if date_str:
            memory_time = self._parse_snapchat_datetime(date_str)
            if memory_time:
                # Find files with similar timestamp (within 5 minutes)
                media_type = memory_entry.get("Media Type", "").lower()
                extensions = ['.jpg', '.jpeg', '.png'] if media_type == "image" else ['.mp4', '.mov']
                
                closest_file = None
                min_time_diff = float('inf')
                
                for file_path in memories_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in extensions:
                        try:
                            file_mtime = datetime.fromtimestamp(
                                file_path.stat().st_mtime,
                                tz=timezone.utc
                            )
                            time_diff = abs((memory_time - file_mtime).total_seconds())
                            
                            if time_diff < min_time_diff and time_diff <= 300:  # Within 5 minutes
                                min_time_diff = time_diff
                                closest_file = file_path
                        except Exception:
                            continue
                
                if closest_file:
                    return closest_file
        
        return None
    
    def _parse_snapchat_datetime(self, dt_string: str) -> Optional[datetime]:
        """Parse Snapchat's various datetime formats.
        
        Args:
            dt_string: Date/time string in Snapchat format.
            
        Returns:
            Timezone-aware datetime object, or None if unparseable.
            
        Supported formats:
            - "2019-06-15 14:30:22 UTC"
            - "2019-06-15T14:30:22-04:00"
            - ISO 8601 formats
        """
        if not dt_string:
            return None
        
        # Try format: "2019-06-15 14:30:22 UTC"
        try:
            dt = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S UTC")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        
        # Try ISO 8601 format with timezone
        try:
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        # Try format without timezone (assume UTC)
        try:
            dt = datetime.fromisoformat(dt_string)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
        
        logger.warning(f"Could not parse datetime: {dt_string}")
        return None
    
    def _parse_snapchat_location(self, entry: Dict[str, Any]) -> Optional[Location]:
        """Parse location from memory entry.
        
        Args:
            entry: Memory entry dictionary.
            
        Returns:
            Location object, or None if no location data available.
            
        Handles:
            - Structured data: {"Latitude": "40.7128", "Longitude": "-74.0060"}
            - String location: "New York, NY, USA"
        """
        # Try to parse structured lat/lon
        lat_str = entry.get("Latitude")
        lon_str = entry.get("Longitude")
        
        if lat_str and lon_str:
            try:
                latitude = float(lat_str)
                longitude = float(lon_str)
                
                geopoint = GeoPoint(latitude=latitude, longitude=longitude)
                
                # Include place name if available
                place_name = entry.get("Location")
                
                return Location(
                    geo_point=geopoint,
                    place_name=place_name,
                    confidence=ConfidenceLevel.HIGH
                )
            except (ValueError, TypeError):
                pass
        
        # Try string location only
        location_str = entry.get("Location")
        if location_str and location_str.strip():
            return Location(
                place_name=location_str,
                confidence=ConfidenceLevel.MEDIUM
            )
        
        return None
    
    def _get_username_from_account(self, root: Path) -> Optional[str]:
        """Extract user's Snapchat username from account.json.
        
        Args:
            root: Root directory of Snapchat export.
            
        Returns:
            Username string, or None if not found.
            
        Usage:
            Used to distinguish "sent" vs "received" in chat parsing.
        """
        account_file = root / "account.json"
        if not account_file.exists():
            return None
        
        try:
            with open(account_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get("Username")
        except Exception as e:
            logger.warning(f"Error reading account.json: {e}")
            return None
    
    def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Remove duplicate memories based on timestamp and type.
        
        Args:
            memories: List of Memory objects potentially containing duplicates.
            
        Returns:
            Deduplicated list of Memory objects.
            
        Deduplication logic:
            Memories with same timestamp and media type are considered duplicates.
            Keeps the one with more complete metadata.
        """
        seen: Dict[tuple, Memory] = {}
        
        for memory in memories:
            if not memory.created_at:
                # Can't deduplicate without timestamp, keep it
                continue
            
            key = (memory.created_at, memory.media_type)
            
            if key not in seen:
                seen[key] = memory
            else:
                # Keep the one with source_path if available
                if memory.source_path and not seen[key].source_path:
                    seen[key] = memory
                elif memory.location and not seen[key].location:
                    seen[key] = memory
        
        # Add back memories without timestamps
        result = list(seen.values())
        for memory in memories:
            if not memory.created_at:
                result.append(memory)
        
        return result
