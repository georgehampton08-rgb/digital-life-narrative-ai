"""Source Detection Module for Digital Life Narrative AI.

This module provides fast, non-fatal detection of platform-specific exports
in user-provided directories. It identifies which platforms (Snapchat, Google Photos,
Facebook, local files, etc.) are present before the parsing pipeline begins.

Detection is:
- Fast: Quick directory scan without deep file analysis
- Non-fatal: Unknown directories return empty results, not exceptions
- Transparent: Provides evidence for each detection
- Multi-source aware: One folder can contain multiple export types

Typical usage:
    >>> from pathlib import Path
    >>> from src.detection import detect_sources, summarize_detections
    >>> 
    >>> results = detect_sources(Path("/path/to/exports"))
    >>> for line in summarize_detections(results):
    ...     print(line)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.memory import ConfidenceLevel, SourcePlatform

logger = logging.getLogger(__name__)

# Media file extensions (case-insensitive)
MEDIA_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.heic', '.webp',
    '.mp4', '.mov', '.avi', '.mkv', '.mp3', '.m4a', '.wav'
}

# Metadata file extensions
METADATA_EXTENSIONS = {'.json', '.xml', '.html', '.txt', '.csv'}

# System files to skip during scanning
SKIP_FILES = {'.DS_Store', 'Thumbs.db', 'desktop.ini'}
SKIP_DIRS = {'__MACOSX', '.git', '.svn', 'node_modules'}


@dataclass
class DetectionResult:
    """Result of platform detection for a directory.
    
    Attributes:
        platform: The detected source platform.
        confidence: Confidence level of the detection (HIGH, MEDIUM, LOW).
        root_path: Root directory where this export is located.
        evidence: List of human-readable reasons for the detection.
        estimated_file_count: Rough count of media files (None if not calculated).
        date_range_hint: Optional tuple of (earliest, latest) date strings from folder names.
    """
    platform: SourcePlatform
    confidence: ConfidenceLevel
    root_path: Path
    evidence: List[str] = field(default_factory=list)
    estimated_file_count: Optional[int] = None
    date_range_hint: Optional[Tuple[str, str]] = None
    
    def to_summary(self) -> str:
        """Generate one-line human-readable summary.
        
        Returns:
            Summary string with emoji, platform name, path, and confidence.
            
        Examples:
            "✓ SNAPCHAT export detected at /exports/snap (HIGH confidence)"
            "? LOCAL photos detected at /photos (LOW confidence)"
        """
        emoji = "✓" if self.confidence == ConfidenceLevel.HIGH else "?" if self.confidence == ConfidenceLevel.LOW else "~"
        platform_name = self.platform.value.upper()
        confidence_str = self.confidence.value.upper()
        return f"{emoji} {platform_name} export detected at {self.root_path} ({confidence_str} confidence)"
    
    def __str__(self) -> str:
        """String representation (alias for to_summary)."""
        return self.to_summary()


@dataclass
class DirectorySummary:
    """Lightweight summary of directory contents for detection.
    
    Attributes:
        root: Root directory that was scanned.
        top_level_files: List of filenames at the root level.
        top_level_dirs: List of directory names at the root level.
        deep_file_sample: Sample of files from deeper directory levels.
        total_files_estimate: Estimated total number of files.
        media_file_count: Count of files with media extensions.
        json_file_count: Count of JSON files.
        has_nested_structure: Whether directory has subdirectories.
    """
    root: Path
    top_level_files: List[str] = field(default_factory=list)
    top_level_dirs: List[str] = field(default_factory=list)
    deep_file_sample: List[str] = field(default_factory=list)
    total_files_estimate: int = 0
    media_file_count: int = 0
    json_file_count: int = 0
    has_nested_structure: bool = False


def is_media_extension(filename: str) -> bool:
    """Check if filename has a media extension.
    
    Args:
        filename: Name of the file to check.
        
    Returns:
        True if the file has a recognized media extension, False otherwise.
        
    Examples:
        >>> is_media_extension("photo.jpg")
        True
        >>> is_media_extension("video.MP4")
        True
        >>> is_media_extension("data.json")
        False
    """
    ext = Path(filename).suffix.lower()
    return ext in MEDIA_EXTENSIONS


def is_metadata_extension(filename: str) -> bool:
    """Check if filename has a metadata extension.
    
    Args:
        filename: Name of the file to check.
        
    Returns:
        True if the file has a recognized metadata extension, False otherwise.
        
    Examples:
        >>> is_metadata_extension("data.json")
        True
        >>> is_metadata_extension("metadata.XML")
        True
        >>> is_metadata_extension("photo.jpg")
        False
    """
    ext = Path(filename).suffix.lower()
    return ext in METADATA_EXTENSIONS


def scan_directory_structure(
    path: Path,
    max_depth: int = 3,
    sample_size: int = 100
) -> DirectorySummary:
    """Quick scan of directory structure for detection purposes.
    
    Performs a fast scan without reading file contents. Lists top-level items,
    samples files from deeper levels, and counts media vs metadata files.
    
    Args:
        path: Directory to scan.
        max_depth: Maximum depth to scan for file samples (default: 3).
        sample_size: Maximum number of files to sample from deep levels (default: 100).
        
    Returns:
        DirectorySummary with structural information about the directory.
        
    Notes:
        - Skips hidden files/directories (starting with '.')
        - Skips system files (.DS_Store, Thumbs.db, etc.)
        - Handles permission errors gracefully
        - Should complete in <1 second for typical exports
    """
    summary = DirectorySummary(root=path)
    
    try:
        # Scan top level
        for item in path.iterdir():
            # Skip hidden and system items
            if item.name.startswith('.') or item.name in SKIP_FILES or item.name in SKIP_DIRS:
                continue
                
            if item.is_file():
                summary.top_level_files.append(item.name)
                summary.total_files_estimate += 1
                
                if is_media_extension(item.name):
                    summary.media_file_count += 1
                elif item.suffix.lower() == '.json':
                    summary.json_file_count += 1
                    
            elif item.is_dir():
                summary.top_level_dirs.append(item.name)
                summary.has_nested_structure = True
        
        # Sample files from deeper levels
        sampled_count = 0
        for root, dirs, files in os.walk(path):
            # Skip hidden and system directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
            
            # Check depth
            depth = len(Path(root).relative_to(path).parts)
            if depth > max_depth:
                continue
            
            for filename in files:
                if filename.startswith('.') or filename in SKIP_FILES:
                    continue
                    
                summary.total_files_estimate += 1
                
                if is_media_extension(filename):
                    summary.media_file_count += 1
                elif filename.endswith('.json'):
                    summary.json_file_count += 1
                
                # Add to sample if we haven't reached limit
                if sampled_count < sample_size and depth > 0:
                    rel_path = str(Path(root).relative_to(path) / filename)
                    summary.deep_file_sample.append(rel_path)
                    sampled_count += 1
                    
                if sampled_count >= sample_size:
                    break
                    
            if sampled_count >= sample_size:
                break
                
    except PermissionError as e:
        logger.warning(f"Permission denied scanning {path}: {e}")
    except Exception as e:
        logger.warning(f"Error scanning {path}: {e}")
    
    return summary


def detect_snapchat(summary: DirectorySummary, root: Path) -> Optional[DetectionResult]:
    """Detect Snapchat export in directory.
    
    Args:
        summary: Directory summary from scan_directory_structure.
        root: Root directory being analyzed.
        
    Returns:
        DetectionResult if Snapchat export detected, None otherwise.
        
    Detection signals:
        HIGH confidence:
            - memories_history.json exists at root or in subdirectory
            - Both chat_history/ AND memories/ directories exist
            - account.json + snap_history/ exist together
            
        MEDIUM confidence:
            - Files matching *_memories*.json pattern
            - Directory named "memories" with media files inside
            - snap_history.json without other signals
            
        LOW confidence:
            - Directory structure vaguely matches but missing key files
    """
    evidence = []
    confidence = None
    
    # Check for signature files
    has_memories_history = False
    has_memories_dir = False
    has_chat_history = False
    has_account = False
    has_snap_history = False
    
    # Check top-level files
    for filename in summary.top_level_files:
        if filename == 'memories_history.json':
            has_memories_history = True
            evidence.append("Found memories_history.json")
        elif filename == 'account.json':
            has_account = True
            evidence.append("Found account.json")
        elif filename == 'snap_history.json':
            has_snap_history = True
            evidence.append("Found snap_history.json")
        elif '_memories' in filename.lower() and filename.endswith('.json'):
            evidence.append(f"Found Snapchat-style file: {filename}")
    
    # Check top-level directories
    for dirname in summary.top_level_dirs:
        if dirname.lower() == 'memories':
            has_memories_dir = True
            evidence.append("Found memories/ directory")
        elif dirname.lower() == 'chat_history':
            has_chat_history = True
            evidence.append("Found chat_history/ directory")
        elif dirname.lower() == 'snap_history':
            has_snap_history = True
            evidence.append("Found snap_history/ directory")
    
    # Check deep samples for memories_history.json
    for filepath in summary.deep_file_sample:
        if 'memories_history.json' in filepath:
            has_memories_history = True
            evidence.append(f"Found memories_history.json at {filepath}")
            break
    
    # Determine confidence
    if has_memories_history:
        confidence = ConfidenceLevel.HIGH
    elif (has_chat_history and has_memories_dir) or (has_account and has_snap_history):
        confidence = ConfidenceLevel.HIGH
        evidence.append("Found chat_history/ and memories/ directories (Snapchat export structure)")
    elif has_memories_dir or has_snap_history or any('_memories' in f for f in summary.top_level_files):
        confidence = ConfidenceLevel.MEDIUM
    
    if confidence:
        return DetectionResult(
            platform=SourcePlatform.SNAPCHAT,
            confidence=confidence,
            root_path=root,
            evidence=evidence,
            estimated_file_count=summary.media_file_count
        )
    
    return None


def detect_google_photos(summary: DirectorySummary, root: Path) -> Optional[DetectionResult]:
    """Detect Google Photos Takeout export in directory.
    
    Args:
        summary: Directory summary from scan_directory_structure.
        root: Root directory being analyzed.
        
    Returns:
        DetectionResult if Google Photos export detected, None otherwise.
        
    Detection signals:
        HIGH confidence:
            - "Google Photos/" directory exists
            - "Takeout/Google Photos/" path exists
            - Directory contains "Photos from YYYY/" subdirectories
            
        MEDIUM confidence:
            - Many .json files alongside media files (sidecar pattern)
            - JSON files containing "photoTakenTime" or "geoDataExif" keys
            - "print-subscriptions.json" exists (Google Photos specific)
            
        LOW confidence:
            - Year-named folders (2019/, 2020/) with images but no sidecars
    """
    evidence = []
    confidence = None
    
    has_google_photos_dir = False
    has_takeout_structure = False
    has_year_folders = False
    has_sidecar_pattern = False
    
    # Check for Google Photos directory
    for dirname in summary.top_level_dirs:
        if dirname == 'Google Photos' or dirname == 'google photos':
            has_google_photos_dir = True
            evidence.append("Found 'Google Photos' directory")
        elif dirname == 'Takeout':
            # Check if Takeout/Google Photos exists
            takeout_path = root / 'Takeout' / 'Google Photos'
            if takeout_path.exists():
                has_takeout_structure = True
                evidence.append("Found Takeout/Google Photos structure")
        elif dirname.startswith('Photos from '):
            evidence.append(f"Found Google Photos album: {dirname}")
    
    # Check for year-based folder structure
    year_dirs = [d for d in summary.top_level_dirs if d.isdigit() and len(d) == 4]
    if len(year_dirs) >= 2:
        has_year_folders = True
        evidence.append(f"Found year-based folders: {', '.join(sorted(year_dirs))}")
    
    # Check for sidecar JSON pattern
    # If we have similar counts of media and JSON files, likely sidecar pattern
    if summary.json_file_count > 0 and summary.media_file_count > 0:
        ratio = summary.json_file_count / max(summary.media_file_count, 1)
        if 0.5 <= ratio <= 1.5:  # Similar counts
            has_sidecar_pattern = True
            evidence.append(f"Found JSON sidecar pattern ({summary.json_file_count} JSON, {summary.media_file_count} media files)")
    
    # Check for Google Photos specific files
    for filename in summary.top_level_files:
        if filename == 'print-subscriptions.json':
            evidence.append("Found print-subscriptions.json (Google Photos specific)")
            has_sidecar_pattern = True
    
    # Check deep samples for Google Photos patterns
    for filepath in summary.deep_file_sample:
        if 'Photos from' in filepath:
            evidence.append("Found 'Photos from' folder structure")
            break
    
    # Determine confidence
    if has_google_photos_dir or has_takeout_structure:
        confidence = ConfidenceLevel.HIGH
    elif has_sidecar_pattern and (has_year_folders or summary.json_file_count > 10):
        confidence = ConfidenceLevel.MEDIUM
    elif has_year_folders and summary.media_file_count > 20:
        confidence = ConfidenceLevel.LOW
    
    if confidence:
        return DetectionResult(
            platform=SourcePlatform.GOOGLE_PHOTOS,
            confidence=confidence,
            root_path=root,
            evidence=evidence,
            estimated_file_count=summary.media_file_count
        )
    
    return None


def detect_facebook(summary: DirectorySummary, root: Path) -> Optional[DetectionResult]:
    """Detect Facebook data export in directory.
    
    Args:
        summary: Directory summary from scan_directory_structure.
        root: Root directory being analyzed.
        
    Returns:
        DetectionResult if Facebook export detected, None otherwise.
        
    Detection signals:
        HIGH confidence:
            - photos_and_videos/ directory exists
            - posts/ directory with JSON files
            - your_posts/ or your_photos/ directories
            
        MEDIUM confidence:
            - messages/ directory with conversation subfolders
            - profile_information/ directory
            
    Note:
        Full Facebook parser not implemented yet, but detection should work.
    """
    evidence = []
    confidence = None
    
    has_photos_and_videos = False
    has_posts = False
    has_your_posts = False
    has_your_photos = False
    has_messages = False
    has_profile = False
    
    # Check directories
    for dirname in summary.top_level_dirs:
        dirname_lower = dirname.lower()
        if dirname_lower == 'photos_and_videos':
            has_photos_and_videos = True
            evidence.append("Found photos_and_videos/ directory")
        elif dirname_lower == 'posts':
            has_posts = True
            evidence.append("Found posts/ directory")
        elif dirname_lower == 'your_posts':
            has_your_posts = True
            evidence.append("Found your_posts/ directory")
        elif dirname_lower == 'your_photos':
            has_your_photos = True
            evidence.append("Found your_photos/ directory")
        elif dirname_lower == 'messages':
            has_messages = True
            evidence.append("Found messages/ directory")
        elif dirname_lower == 'profile_information':
            has_profile = True
            evidence.append("Found profile_information/ directory")
    
    # Determine confidence
    if has_photos_and_videos or has_posts or (has_your_posts and has_your_photos):
        confidence = ConfidenceLevel.HIGH
    elif has_messages or has_profile:
        confidence = ConfidenceLevel.MEDIUM
    
    if confidence:
        return DetectionResult(
            platform=SourcePlatform.FACEBOOK,
            confidence=confidence,
            root_path=root,
            evidence=evidence,
            estimated_file_count=summary.media_file_count
        )
    
    return None


def detect_local_files(summary: DirectorySummary, root: Path) -> Optional[DetectionResult]:
    """Fallback detection for generic local photo libraries.
    
    This is the catch-all for directories that contain media files but don't
    match any specific platform pattern.
    
    Args:
        summary: Directory summary from scan_directory_structure.
        root: Root directory being analyzed.
        
    Returns:
        DetectionResult with LOW confidence if sufficient media files found, None otherwise.
        
    Criteria:
        - At least 5 media files found
        - No other platform detected with HIGH/MEDIUM confidence
        - Confidence is always LOW (no platform-specific signals)
    """
    # Only return result if we have enough media files
    if summary.media_file_count < 5:
        return None
    
    evidence = [
        f"Found {summary.media_file_count} media files with generic structure",
        "No platform-specific export markers detected"
    ]
    
    return DetectionResult(
        platform=SourcePlatform.LOCAL,
        confidence=ConfidenceLevel.LOW,
        root_path=root,
        evidence=evidence,
        estimated_file_count=summary.media_file_count
    )


def detect_sources(path: Path) -> List[DetectionResult]:
    """Main detection function: identify all platforms present in a directory.
    
    This is the primary entry point for source detection. It runs all platform
    detectors and returns results sorted by confidence.
    
    Args:
        path: Directory to analyze for platform exports.
        
    Returns:
        List of DetectionResult objects, sorted by confidence (HIGH first).
        Returns empty list if path doesn't exist, isn't a directory, or no sources detected.
        
    Behavior:
        1. Verify path exists and is directory
        2. Scan directory structure
        3. Run all platform detectors
        4. If no platform-specific detection AND media files exist, detect as LOCAL
        5. Sort results by confidence
        
    Examples:
        >>> results = detect_sources(Path("/exports/my_snapchat"))
        >>> for result in results:
        ...     print(result.to_summary())
        ✓ SNAPCHAT export detected at /exports/my_snapchat (HIGH confidence)
    """
    # Validate path
    if not path.exists():
        logger.debug(f"Path does not exist: {path}")
        return []
    
    if not path.is_dir():
        logger.debug(f"Path is not a directory: {path}")
        return []
    
    try:
        # Scan directory structure
        summary = scan_directory_structure(path)
        
        # Run all platform detectors
        results: List[DetectionResult] = []
        
        snapchat_result = detect_snapchat(summary, path)
        if snapchat_result:
            results.append(snapchat_result)
        
        google_photos_result = detect_google_photos(summary, path)
        if google_photos_result:
            results.append(google_photos_result)
        
        facebook_result = detect_facebook(summary, path)
        if facebook_result:
            results.append(facebook_result)
        
        # If no platform-specific detection, try LOCAL fallback
        if not results:
            local_result = detect_local_files(summary, path)
            if local_result:
                results.append(local_result)
        
        # Sort by confidence (HIGH first)
        confidence_order = {
            ConfidenceLevel.HIGH: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.LOW: 2
        }
        results.sort(key=lambda r: confidence_order.get(r.confidence, 999))
        
        return results
        
    except PermissionError as e:
        logger.warning(f"Permission denied accessing {path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error detecting sources in {path}: {e}", exc_info=True)
        return []


def detect_sources_recursive(path: Path, max_depth: int = 2) -> List[DetectionResult]:
    """Recursively detect sources in subdirectories.
    
    Useful when user points at a folder containing multiple export folders.
    
    Args:
        path: Root directory to search.
        max_depth: Maximum recursion depth (default: 2).
        
    Returns:
        List of DetectionResult objects from all subdirectories, deduplicated.
        
    Example directory structure:
        my_exports/
        ├── snapchat_export/      <- Detected as SNAPCHAT
        ├── google_takeout/        <- Detected as GOOGLE_PHOTOS
        └── camera_roll/           <- Detected as LOCAL
        
    Examples:
        >>> results = detect_sources_recursive(Path("/my_exports"))
        >>> len(results)
        3
    """
    all_results: List[DetectionResult] = []
    seen_paths: Set[Path] = set()
    
    def _detect_recursive(current_path: Path, depth: int):
        """Inner recursive function."""
        if depth > max_depth or current_path in seen_paths:
            return
        
        seen_paths.add(current_path)
        
        # Try detection on current path
        results = detect_sources(current_path)
        for result in results:
            # Only add if not already detected
            if result.root_path not in seen_paths or result not in all_results:
                all_results.append(result)
        
        # If nothing found, recurse into subdirectories
        if not results and depth < max_depth:
            try:
                for item in current_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        _detect_recursive(item, depth + 1)
            except PermissionError:
                logger.warning(f"Permission denied accessing subdirectories of {current_path}")
            except Exception as e:
                logger.warning(f"Error recursing into {current_path}: {e}")
    
    _detect_recursive(path, 0)
    
    # Deduplicate by (platform, root_path)
    seen = set()
    deduplicated = []
    for result in all_results:
        key = (result.platform, result.root_path)
        if key not in seen:
            seen.add(key)
            deduplicated.append(result)
    
    # Sort by confidence
    confidence_order = {
        ConfidenceLevel.HIGH: 0,
        ConfidenceLevel.MEDIUM: 1,
        ConfidenceLevel.LOW: 2
    }
    deduplicated.sort(key=lambda r: confidence_order.get(r.confidence, 999))
    
    return deduplicated


def summarize_detections(results: List[DetectionResult]) -> List[str]:
    """Generate human-readable summary strings for detection results.
    
    Args:
        results: List of detection results.
        
    Returns:
        List of formatted summary strings with emoji for visual scanning.
        
    Format:
        - "✓ SNAPCHAT export detected at /path/to/export (HIGH confidence)"
        - "? LOCAL photos detected at /photos (LOW confidence)"
        
    Examples:
        >>> results = detect_sources(Path("/exports"))
        >>> for line in summarize_detections(results):
        ...     print(line)
    """
    if not results:
        return ["No supported exports detected."]
    
    summaries = []
    for result in results:
        summaries.append(result.to_summary())
        
        # Add evidence as sub-items if available
        if result.evidence:
            for evidence_item in result.evidence[:3]:  # Limit to first 3
                summaries.append(f"  - {evidence_item}")
    
    return summaries


def get_detection_advice(results: List[DetectionResult], path: Path) -> List[str]:
    """Generate helpful advice based on detection results.
    
    Args:
        results: List of detection results.
        path: Path that was analyzed.
        
    Returns:
        List of advice strings to help user understand results.
        
    Examples:
        - If empty: "No supported exports detected. Supported formats: ..."
        - If LOW confidence only: "Detection confidence is low. Results may be incomplete."
        - If multiple platforms: "Multiple exports detected. All will be processed."
    """
    advice = []
    
    if not results:
        advice.append(f"No supported exports detected in {path}.")
        advice.append("Supported formats: Snapchat, Google Photos, Facebook, local media folders.")
        advice.append("Try pointing to a specific export folder or use recursive detection.")
        return advice
    
    # Check confidence levels
    high_confidence = [r for r in results if r.confidence == ConfidenceLevel.HIGH]
    low_confidence = [r for r in results if r.confidence == ConfidenceLevel.LOW]
    
    if not high_confidence and low_confidence:
        advice.append("⚠ Detection confidence is low. Results may be incomplete.")
        advice.append("Consider organizing exports into platform-specific folders.")
    
    if len(results) > 1:
        advice.append(f"✓ Multiple exports detected ({len(results)} total). All will be processed.")
    
    # Estimate total files
    total_files = sum(r.estimated_file_count or 0 for r in results)
    if total_files > 0:
        advice.append(f"📊 Estimated {total_files} media files across all exports.")
    
    return advice
