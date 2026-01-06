"""Export source detection for Digital Life Narrative AI.

This module detects which service(s) an export folder is from, enabling
automatic parser selection. Supports detecting multiple sources in a
single folder and provides confidence levels for each detection.

Detection is designed to be:
- Robust to nested/partial exports
- Able to identify mixed exports
- Non-throwing (returns empty list for no detection)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from organizer.models import Confidence, SourcePlatform

if TYPE_CHECKING:
    from organizer.parsers.base import BaseParser


# =============================================================================
# Constants
# =============================================================================

# Common media file extensions for fallback detection
MEDIA_EXTENSIONS = frozenset({
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif",
    ".tiff", ".tif", ".raw", ".cr2", ".nef", ".arw", ".dng",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    ".3gp", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma",
})


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DetectionResult:
    """Result of detecting an export source platform.

    Attributes:
        platform: The detected source platform.
        confidence: Confidence level of the detection.
        root_path: The detected root path for this platform's data.
        evidence: List of files/patterns that led to this detection.
    """

    platform: SourcePlatform
    confidence: Confidence
    root_path: Path
    evidence: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.platform.value.replace('_', ' ').title()} "
            f"({self.confidence.value} confidence)"
        )


@dataclass
class DirectorySummary:
    """Summary of a directory structure for detection.

    Attributes:
        files: List of filenames at the root level.
        dirs: List of directory names at the root level.
        deep_files: Sample of files found at deeper levels.
        total_files: Approximate total file count.
    """

    files: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    deep_files: list[str] = field(default_factory=list)
    total_files: int = 0


# =============================================================================
# Directory Scanning
# =============================================================================


def scan_directory_structure(path: Path, max_depth: int = 3) -> DirectorySummary:
    """Scan directory and return a summary for detection.

    Creates a lightweight summary of the directory structure without
    loading file contents.

    Args:
        path: Directory path to scan.
        max_depth: Maximum depth to scan for deep files.

    Returns:
        DirectorySummary with files, directories, and sample deep files.
    """
    summary = DirectorySummary()

    if not path.exists() or not path.is_dir():
        return summary

    try:
        # Get root level items
        for item in path.iterdir():
            if item.is_file():
                summary.files.append(item.name)
            elif item.is_dir() and not item.name.startswith("."):
                summary.dirs.append(item.name)

        # Sample deeper files (limit to avoid slow scans)
        deep_file_limit = 100
        deep_file_count = 0

        def scan_deep(current: Path, depth: int) -> None:
            nonlocal deep_file_count
            if depth > max_depth or deep_file_count >= deep_file_limit:
                return

            try:
                for item in current.iterdir():
                    if deep_file_count >= deep_file_limit:
                        return
                    if item.is_file():
                        # Store relative path from root
                        rel_path = item.relative_to(path)
                        summary.deep_files.append(str(rel_path))
                        deep_file_count += 1
                        summary.total_files += 1
                    elif item.is_dir() and not item.name.startswith("."):
                        scan_deep(item, depth + 1)
            except PermissionError:
                pass

        # Scan subdirectories
        for dir_name in summary.dirs[:20]:  # Limit dirs to scan
            scan_deep(path / dir_name, 1)

        summary.total_files += len(summary.files)

    except PermissionError:
        pass

    return summary


def _path_contains(path: Path, target: str, max_depth: int = 3) -> Path | None:
    """Check if path contains a file or directory with the given name.

    Args:
        path: Root path to search from.
        target: File or directory name to find.
        max_depth: Maximum depth to search.

    Returns:
        Path to the found item, or None if not found.
    """
    if max_depth < 0:
        return None

    try:
        for item in path.iterdir():
            if item.name == target:
                return item
            if item.is_dir() and not item.name.startswith("."):
                result = _path_contains(item, target, max_depth - 1)
                if result:
                    return result
    except PermissionError:
        pass

    return None


def _check_json_field(file_path: Path, field_name: str) -> bool:
    """Check if a JSON file contains a specific field.

    Args:
        file_path: Path to the JSON file.
        field_name: Field name to look for.

    Returns:
        True if the field exists in the JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return field_name in data
    except (json.JSONDecodeError, PermissionError, UnicodeDecodeError):
        pass
    return False


# =============================================================================
# Platform-Specific Detectors
# =============================================================================


def _detect_snapchat(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect Snapchat export.

    HIGH confidence indicators:
    - memories_history.json
    - chat_history/ folder
    - memories/ folder with expected structure

    MEDIUM confidence indicators:
    - Files matching *_memories_*.json pattern
    - account.json present
    """
    evidence: list[str] = []
    confidence: Confidence | None = None
    root_path = path

    # HIGH: memories_history.json at root or in subdirectory
    if "memories_history.json" in summary.files:
        evidence.append("Found memories_history.json")
        confidence = Confidence.HIGH
    else:
        found = _path_contains(path, "memories_history.json", max_depth=2)
        if found:
            evidence.append(f"Found {found.relative_to(path)}")
            confidence = Confidence.HIGH
            root_path = found.parent

    # HIGH: chat_history folder
    if "chat_history" in summary.dirs:
        evidence.append("Found chat_history/ folder")
        confidence = Confidence.HIGH

    # HIGH: memories folder
    if "memories" in summary.dirs:
        evidence.append("Found memories/ folder")
        if confidence != Confidence.HIGH:
            confidence = Confidence.HIGH

    # Check for snap_history folder
    if "snap_history" in summary.dirs:
        evidence.append("Found snap_history/ folder")
        confidence = Confidence.HIGH

    # MEDIUM: account.json (Snapchat specific)
    if "account.json" in summary.files:
        evidence.append("Found account.json")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # MEDIUM: Files matching *_memories_*.json
    memory_files = [f for f in summary.files if "_memories_" in f and f.endswith(".json")]
    if memory_files:
        evidence.append(f"Found memory files: {', '.join(memory_files[:3])}")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check deep files for Snapchat patterns
    snap_patterns = ["memories/", "chat_history/", "snap_history/"]
    for deep_file in summary.deep_files:
        for pattern in snap_patterns:
            if pattern in deep_file:
                evidence.append(f"Found Snapchat pattern in: {deep_file}")
                if confidence is None:
                    confidence = Confidence.MEDIUM
                break

    if confidence:
        return DetectionResult(
            platform=SourcePlatform.SNAPCHAT,
            confidence=confidence,
            root_path=root_path,
            evidence=evidence,
        )

    return None


def _detect_google_photos(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect Google Photos/Takeout export.

    HIGH confidence indicators:
    - Google Photos/ folder with year subfolders
    - Takeout/Google Photos/ structure

    MEDIUM confidence indicators:
    - JSON sidecar files with Google Photos metadata
    - print-subscriptions.json
    - shared_album_comments/ folder
    """
    evidence: list[str] = []
    confidence: Confidence | None = None
    root_path = path

    # HIGH: "Google Photos" folder
    if "Google Photos" in summary.dirs:
        evidence.append("Found 'Google Photos/' folder")
        confidence = Confidence.HIGH
        root_path = path / "Google Photos"

    # HIGH: Takeout structure
    if "Takeout" in summary.dirs:
        takeout_path = path / "Takeout"
        if (takeout_path / "Google Photos").exists():
            evidence.append("Found Takeout/Google Photos/ structure")
            confidence = Confidence.HIGH
            root_path = takeout_path / "Google Photos"

    # Check for year-named subfolders (common in Google Photos exports)
    year_folders = [d for d in summary.dirs if d.isdigit() and 1990 <= int(d) <= 2100]
    if year_folders:
        evidence.append(f"Found year folders: {', '.join(year_folders[:5])}")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # MEDIUM: print-subscriptions.json (Google Photos specific)
    if "print-subscriptions.json" in summary.files:
        evidence.append("Found print-subscriptions.json")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check for shared_album_comments folder
    if "shared_album_comments" in summary.dirs:
        evidence.append("Found shared_album_comments/ folder")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # MEDIUM: JSON sidecars with Google Photos metadata patterns
    # Google Photos exports have .json files alongside media files
    json_files = [f for f in summary.files if f.endswith(".json")]
    for json_file in json_files[:5]:  # Check a few
        json_path = path / json_file
        if json_path.exists():
            if _check_json_field(json_path, "photoTakenTime"):
                evidence.append(f"Found Google Photos metadata in {json_file}")
                if confidence is None:
                    confidence = Confidence.MEDIUM
                break
            if _check_json_field(json_path, "geoData"):
                evidence.append(f"Found Google Photos geo metadata in {json_file}")
                if confidence is None:
                    confidence = Confidence.MEDIUM
                break

    # Check deep files for Google Photos patterns
    for deep_file in summary.deep_files:
        if "Google Photos" in deep_file:
            evidence.append(f"Found Google Photos path: {deep_file}")
            if confidence is None:
                confidence = Confidence.MEDIUM
            break

    if confidence:
        return DetectionResult(
            platform=SourcePlatform.GOOGLE_PHOTOS,
            confidence=confidence,
            root_path=root_path,
            evidence=evidence,
        )

    return None


def _detect_facebook(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect Facebook export.

    HIGH confidence indicators:
    - posts/, photos_and_videos/, or your_posts/ folders
    - profile_information/ folder

    MEDIUM confidence indicators:
    - messages/ folder with Facebook format
    - about_you/ folder
    """
    evidence: list[str] = []
    confidence: Confidence | None = None
    root_path = path

    # HIGH: Core Facebook folders
    fb_high_folders = ["posts", "photos_and_videos", "your_posts", "profile_information"]
    for folder in fb_high_folders:
        if folder in summary.dirs:
            evidence.append(f"Found {folder}/ folder")
            confidence = Confidence.HIGH

    # MEDIUM: messages folder
    if "messages" in summary.dirs:
        messages_path = path / "messages"
        # Check for Facebook-style message structure (inbox/, archived_threads/, etc.)
        if messages_path.is_dir():
            try:
                msg_subdirs = [d.name for d in messages_path.iterdir() if d.is_dir()]
                if "inbox" in msg_subdirs or "archived_threads" in msg_subdirs:
                    evidence.append("Found Facebook-style messages/ structure")
                    if confidence is None:
                        confidence = Confidence.MEDIUM
            except PermissionError:
                pass

    # MEDIUM: about_you folder
    if "about_you" in summary.dirs:
        evidence.append("Found about_you/ folder")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check for Facebook-specific files
    fb_files = ["your_posts.json", "your_photos.json", "profile_information.json"]
    for fb_file in fb_files:
        if fb_file in summary.files:
            evidence.append(f"Found {fb_file}")
            if confidence is None:
                confidence = Confidence.MEDIUM

    # Check deep files for Facebook patterns
    fb_patterns = ["photos_and_videos/", "posts/", "messages/inbox/"]
    for deep_file in summary.deep_files:
        for pattern in fb_patterns:
            if pattern in deep_file:
                evidence.append(f"Found Facebook pattern in: {deep_file}")
                if confidence is None:
                    confidence = Confidence.MEDIUM
                break

    if confidence:
        return DetectionResult(
            platform=SourcePlatform.FACEBOOK,
            confidence=confidence,
            root_path=root_path,
            evidence=evidence,
        )

    return None


def _detect_instagram(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect Instagram export.

    HIGH confidence indicators:
    - Contains specific Instagram folder structure
    - media.json or content/media.json

    MEDIUM confidence indicators:
    - Contains stories/, posts/, reels/ folders
    - account_information/ folder
    """
    evidence: list[str] = []
    confidence: Confidence | None = None
    root_path = path

    # HIGH: Instagram-specific folders
    ig_high_folders = ["your_instagram_activity", "logged_information"]
    for folder in ig_high_folders:
        if folder in summary.dirs:
            evidence.append(f"Found {folder}/ folder")
            confidence = Confidence.HIGH

    # MEDIUM: Content folders
    ig_content_folders = ["stories", "posts", "reels", "direct", "igtv"]
    found_content = [f for f in ig_content_folders if f in summary.dirs]
    if found_content:
        evidence.append(f"Found Instagram content folders: {', '.join(found_content)}")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # MEDIUM: account_information folder
    if "account_information" in summary.dirs:
        evidence.append("Found account_information/ folder")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check for media.json
    if "media.json" in summary.files:
        evidence.append("Found media.json")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check for content/media.json path
    content_path = path / "content"
    if content_path.exists() and (content_path / "media.json").exists():
        evidence.append("Found content/media.json structure")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # Check deep files for Instagram patterns
    ig_patterns = ["stories/", "posts/", "reels/", "direct/"]
    for deep_file in summary.deep_files:
        for pattern in ig_patterns:
            if pattern in deep_file:
                evidence.append(f"Found Instagram pattern in: {deep_file}")
                if confidence is None:
                    confidence = Confidence.LOW
                break

    if confidence:
        return DetectionResult(
            platform=SourcePlatform.INSTAGRAM,
            confidence=confidence,
            root_path=root_path,
            evidence=evidence,
        )

    return None


def _detect_onedrive(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect OneDrive export/sync folder.

    OneDrive exports are tricky as they're mostly just files.

    MEDIUM confidence indicators:
    - .onedrive markers or config files
    - OneDrive-specific folder names

    LOW confidence indicators:
    - Generic cloud sync patterns
    """
    evidence: list[str] = []
    confidence: Confidence | None = None
    root_path = path

    # MEDIUM: OneDrive markers
    onedrive_markers = [
        ".onedrive",
        "OneDrive",
        "OneDrive - Personal",
        "desktop.ini",  # Windows OneDrive creates these
    ]

    for marker in onedrive_markers:
        if marker in summary.files or marker in summary.dirs:
            evidence.append(f"Found OneDrive marker: {marker}")
            if confidence is None:
                confidence = Confidence.MEDIUM

    # Check for OneDrive in path name
    if "onedrive" in str(path).lower():
        evidence.append("OneDrive in path name")
        if confidence is None:
            confidence = Confidence.MEDIUM

    # LOW: Pictures/Documents/Desktop folders (common OneDrive sync)
    cloud_folders = ["Pictures", "Documents", "Desktop", "Camera Roll"]
    found_cloud = [f for f in cloud_folders if f in summary.dirs]
    if len(found_cloud) >= 2:
        evidence.append(f"Found cloud sync folders: {', '.join(found_cloud)}")
        if confidence is None:
            confidence = Confidence.LOW

    if confidence:
        return DetectionResult(
            platform=SourcePlatform.ONEDRIVE,
            confidence=confidence,
            root_path=root_path,
            evidence=evidence,
        )

    return None


def _detect_local_media(path: Path, summary: DirectorySummary) -> DetectionResult | None:
    """Detect local media folder (fallback detection).

    LOW confidence - used when no platform-specific markers found.

    Detects directories containing media files (images, videos, audio).
    """
    evidence: list[str] = []
    media_count = 0

    # Count media files at root level
    for file_name in summary.files:
        ext = Path(file_name).suffix.lower()
        if ext in MEDIA_EXTENSIONS:
            media_count += 1

    # Count media files in deep scan
    for deep_file in summary.deep_files:
        ext = Path(deep_file).suffix.lower()
        if ext in MEDIA_EXTENSIONS:
            media_count += 1

    if media_count > 0:
        evidence.append(f"Found {media_count} media files")

        # Determine confidence based on media count
        if media_count >= 10:
            confidence = Confidence.LOW
        else:
            # Too few files to be meaningful
            return None

        return DetectionResult(
            platform=SourcePlatform.LOCAL,
            confidence=confidence,
            root_path=path,
            evidence=evidence,
        )

    return None


# =============================================================================
# Main Detection Functions
# =============================================================================


def detect_export_source(path: Path) -> list[DetectionResult]:
    """Detect all export sources in a directory.

    Scans the given directory and identifies which platform(s)
    the export data belongs to. Supports mixed exports where
    multiple platforms are present.

    Args:
        path: Directory path to scan for exports.

    Returns:
        List of DetectionResult for each detected platform.
        Returns empty list if no sources detected (never raises).
    """
    results: list[DetectionResult] = []

    if not path.exists() or not path.is_dir():
        return results

    try:
        # Scan directory structure
        summary = scan_directory_structure(path, max_depth=3)

        # Run all detectors
        detectors = [
            _detect_snapchat,
            _detect_google_photos,
            _detect_facebook,
            _detect_instagram,
            _detect_onedrive,
        ]

        for detector in detectors:
            try:
                result = detector(path, summary)
                if result:
                    results.append(result)
            except Exception:
                # Individual detector failure shouldn't stop others
                continue

        # If no platform-specific detection, try local media fallback
        if not results:
            local_result = _detect_local_media(path, summary)
            if local_result:
                results.append(local_result)

        # Sort by confidence (HIGH first)
        confidence_order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        results.sort(key=lambda r: confidence_order.get(r.confidence, 99))

    except Exception:
        # Never raise for detection failures
        pass

    return results


def detect_and_suggest(path: Path) -> tuple[list[DetectionResult], list[str]]:
    """Detect sources and provide human-readable suggestions.

    Runs detection and generates helpful messages for CLI output.

    Args:
        path: Directory path to scan.

    Returns:
        Tuple of (detection results, list of suggestion strings).
    """
    results = detect_export_source(path)
    suggestions: list[str] = []

    if not results:
        suggestions.append(
            "⚠️  No recognized export format detected. "
            "Make sure you've extracted the export archive."
        )
        suggestions.append(
            "Supported formats: Snapchat, Google Photos, Facebook, Instagram, OneDrive"
        )
        return results, suggestions

    # Generate suggestions for each detection
    for result in results:
        platform_name = result.platform.value.replace("_", " ").title()
        confidence_emoji = {
            Confidence.HIGH: "✅",
            Confidence.MEDIUM: "🔶",
            Confidence.LOW: "⚠️",
        }.get(result.confidence, "❓")

        suggestions.append(
            f"{confidence_emoji} Detected {platform_name} export "
            f"({result.confidence.value} confidence)"
        )

        # Add evidence details for non-high confidence
        if result.confidence != Confidence.HIGH and result.evidence:
            suggestions.append(f"   Evidence: {result.evidence[0]}")

    # Warnings for mixed exports
    if len(results) > 1:
        suggestions.append(
            "\n📁 Multiple export sources detected in this folder. "
            "Each will be processed separately."
        )

    # Warning for low confidence only
    if all(r.confidence == Confidence.LOW for r in results):
        suggestions.append(
            "\n⚠️  Only low-confidence detections. Results may be incomplete. "
            "Consider organizing exports into separate folders."
        )

    return results, suggestions


def get_parser_for_platform(platform: SourcePlatform) -> type[BaseParser] | None:
    """Get the appropriate parser class for a detected platform.

    Uses lazy imports to avoid circular dependencies.

    Args:
        platform: The detected source platform.

    Returns:
        Parser class for the platform, or None if not supported.
    """
    # Lazy import to avoid circular dependencies
    try:
        from organizer.parsers.snapchat import SnapchatParser
        from organizer.parsers.google_photos import GooglePhotosParser
        from organizer.parsers.facebook import FacebookParser
        from organizer.parsers.instagram import InstagramParser
        from organizer.parsers.local import LocalMediaParser

        parser_map: dict[SourcePlatform, type[BaseParser]] = {
            SourcePlatform.SNAPCHAT: SnapchatParser,
            SourcePlatform.GOOGLE_PHOTOS: GooglePhotosParser,
            SourcePlatform.FACEBOOK: FacebookParser,
            SourcePlatform.INSTAGRAM: InstagramParser,
            SourcePlatform.LOCAL: LocalMediaParser,
            SourcePlatform.ONEDRIVE: LocalMediaParser,  # OneDrive uses local parser
        }

        return parser_map.get(platform)

    except ImportError:
        # Parsers not yet implemented
        return None


def auto_detect_and_parse(path: Path) -> dict[SourcePlatform, list]:
    """Convenience function to detect and get parsers for all sources.

    Args:
        path: Directory path to scan and parse.

    Returns:
        Dictionary mapping platforms to their parser instances.
    """
    results = detect_export_source(path)
    parsers: dict[SourcePlatform, list] = {}

    for result in results:
        parser_class = get_parser_for_platform(result.platform)
        if parser_class:
            parsers[result.platform] = [parser_class, result.root_path]

    return parsers
