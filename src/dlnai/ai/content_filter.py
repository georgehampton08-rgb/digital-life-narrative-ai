"""
Content Filter Module - Optional Content Classification Layer.

This module provides optional content classification for Memory objects,
identifying potentially sensitive content so users can control what appears
in their reports. It operates in two modes:

1. Metadata-Only: Uses filename patterns, captions, and metadata
2. Pixel-Aware (Default with Disclosure): Uses Gemini Vision for higher-confidence classification

PRIVACY MODEL: Pixel analysis is ENABLED BY DEFAULT but requires users to acknowledge
what data is sent to Gemini Vision. Users can OPT-OUT at any time for metadata-only detection.
"""

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from PIL import Image
except ImportError:
    Image = None  # Graceful degradation if PIL not available

from dlnai.ai.client import AIClient, AIClientError
from dlnai.ai.disclosure import check_and_prompt_disclosure
from dlnai.config import get_config
from dlnai.core.memory import Memory, MediaType, SourcePlatform
from dlnai.core.privacy import PrivacySettings
from dlnai.core.safety import (
    SENSITIVE_FILENAME_PATTERNS,
    DetectionMethod,
    MemorySafetyState,
    SafetyAction,
    SafetyCategory,
    SafetyFlag,
    SafetyReport,
    SafetySettings,
    SensitivityLevel,
    create_safety_state,
    merge_safety_states,
)

# Configure logging
logger = logging.getLogger(__name__)


# --- Vision Classification Prompt ---

VISION_CLASSIFICATION_PROMPT = """Analyze this image for content safety classification. This is the user's
own personal photo being organized. Respond with JSON only.

Check for these categories:
- NUDITY: Unclothed human bodies
- SEXUAL: Sexually explicit or suggestive content
- VIOLENCE: Graphic violence, injury, gore
- SELF_HARM: Content depicting self-harm
- SUBSTANCE: Drug/alcohol content
- DISTURBING: Generally disturbing imagery
- PRIVATE: Sensitive documents (IDs, medical, financial)

For each detected category, provide:
- category: The category name (must match one above)
- confidence: 0.0-1.0 (how certain you are)
- severity: "low", "medium", or "high"
- reason: Brief explanation (1 sentence, non-graphic)

If nothing sensitive is detected, return empty array.

Response format:
{
  "findings": [
    {"category": "...", "confidence": 0.0, "severity": "...", "reason": "..."}
  ]
}"""


# --- Data Models ---


@dataclass
class FilterConfig:
    """Configuration for the content filter."""

    enabled: bool = True
    safety_settings: Optional[SafetySettings] = None
    batch_size: int = 50
    max_thumbnail_size: tuple[int, int] = (256, 256)
    thumbnail_quality: int = 70
    timeout_per_image_seconds: float = 10.0
    skip_videos: bool = True
    cache_results: bool = True


@dataclass
class FilterProgress:
    """Progress information for callbacks."""

    stage: str
    current: int
    total: int
    flagged_so_far: int
    current_file: Optional[str] = None

    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def to_status_line(self) -> str:
        """Generate a human-readable status line."""
        return (
            f"{self.stage}: {self.current}/{self.total} ({self.percentage():.1f}%) - "
            f"{self.flagged_so_far} flagged"
        )


@dataclass
class FilterResult:
    """Result of filtering a batch of memories."""

    total_processed: int
    total_flagged: int
    states: Dict[str, MemorySafetyState]
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    methods_used: Set[DetectionMethod] = field(default_factory=set)
    safety_report: Optional[SafetyReport] = None


# --- Exceptions ---


class PixelAnalysisConsentError(Exception):
    """Raised when pixel analysis is attempted without disclosure acknowledgment."""

    pass


# --- Caption Patterns ---

SENSITIVE_CAPTION_PATTERNS: Dict[SafetyCategory, List[str]] = {
    SafetyCategory.NUDITY: [
        r"(?i)\bnude\b",
        r"(?i)\bnaked\b",
        r"(?i)\bnsfw\b",
        r"(?i)\bboudoir\b",
        r"(?i)\btopless\b",
    ],
    SafetyCategory.SEXUAL: [
        r"(?i)\bsex\b",
        r"(?i)\bxxx\b",
        r"(?i)\bporn\b",
        r"(?i)\badult\b.*\bcontent\b",
    ],
    SafetyCategory.VIOLENCE: [
        r"(?i)\bblood\b",
        r"(?i)\bgore\b",
        r"(?i)\binjury\b",
        r"(?i)\baccident\b",
        r"(?i)\bfight\b",
    ],
    SafetyCategory.SUBSTANCE: [
        r"(?i)\bdrunk\b",
        r"(?i)\bhigh\b",
        r"(?i)\bwasted\b",
        r"(?i)\bpartying\b",
    ],
    SafetyCategory.PRIVATE: [
        r"(?i)\bconfidential\b",
        r"(?i)\bprivate\b",
        r"(?i)\bdo not share\b",
        r"(?i)\bpersonal\b",
    ],
}


# --- Main Class ---


class ContentFilter:
    """
    Content classification engine with metadata-only and pixel-aware modes.
    """

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        safety_settings: Optional[SafetySettings] = None,
        privacy_settings: Optional[PrivacySettings] = None,
        ai_client: Optional[AIClient] = None,
    ) -> None:
        """
        Initialize the content filter.

        Args:
            config: Filter configuration (uses defaults if None)
            safety_settings: Safety settings (loads from config if None)
            privacy_settings: Privacy settings (loads from config if None)
            ai_client: AI client for pixel analysis (creates if needed and allowed)
        """
        self._config = config or FilterConfig()

        # Load settings from global config if not provided
        if safety_settings is None:
            app_config = get_config()
            safety_settings = SafetySettings()  # Use defaults for now

        # Check and prompt for disclosure if pixel analysis is enabled but not acknowledged
        # This ensures users see the disclosure before any pixel analysis happens
        safety_settings = check_and_prompt_disclosure(safety_settings, interactive=True)
        self._safety_settings = safety_settings

        if privacy_settings is None:
            app_config = get_config()
            privacy_settings = app_config.privacy
        self._privacy_settings = privacy_settings

        # Initialize AI client only if pixel analysis is enabled AND disclosure acknowledged
        if safety_settings.is_pixel_analysis_allowed() and ai_client is None:
            try:
                self._ai_client = AIClient()
            except Exception as e:
                logger.warning(f"Failed to initialize AI client for pixel analysis: {e}")
                self._ai_client = None
        else:
            self._ai_client = ai_client

        self._logger = logger
        self._cache: Dict[str, MemorySafetyState] = {}

    def filter_memories(
        self,
        memories: List[Memory],
        progress_callback: Optional[Callable[[FilterProgress], None]] = None,
    ) -> FilterResult:
        """
        Main entry point to classify all memories.

        Args:
            memories: List of memories to classify
            progress_callback: Optional callback for progress updates

        Returns:
            FilterResult with classification results
        """
        start_time = time.time()
        states: Dict[str, MemorySafetyState] = {}
        errors: List[str] = []
        methods_used: Set[DetectionMethod] = set()
        flagged_count = 0

        self._logger.info(f"Starting content filtering for {len(memories)} memories")

        for idx, memory in enumerate(memories, 1):
            try:
                # Report progress
                if progress_callback:
                    progress = FilterProgress(
                        stage="Analyzing",
                        current=idx,
                        total=len(memories),
                        flagged_so_far=flagged_count,
                        current_file=memory.source_filename if memory.source_filename else None,
                    )
                    progress_callback(progress)

                # Check cache
                if self._config.cache_results and memory.id in self._cache:
                    states[memory.id] = self._cache[memory.id]
                    if states[memory.id].is_sensitive:
                        flagged_count += 1
                    continue

                # Filter the memory
                state = self.filter_single(memory)
                states[memory.id] = state

                # Track methods used
                for flag in state.flags:
                    methods_used.add(flag.detection_method)

                # Update counts
                if state.is_sensitive:
                    flagged_count += 1

                # Cache result
                if self._config.cache_results:
                    self._cache[memory.id] = state

            except Exception as e:
                error_msg = f"Error filtering memory {memory.id}: {str(e)}"
                self._logger.error(error_msg)
                errors.append(error_msg)

        elapsed = time.time() - start_time

        # Build safety report
        by_category: Dict[SafetyCategory, int] = {}
        by_action: Dict[SafetyAction, int] = {}
        review_count = 0

        for state in states.values():
            if state.is_sensitive:
                primary = state.get_primary_category()
                if primary:
                    by_category[primary] = by_category.get(primary, 0) + 1
            by_action[state.resolved_action] = by_action.get(state.resolved_action, 0) + 1
            if state.requires_user_review:
                review_count += 1

        safety_report = SafetyReport(
            total_analyzed=len(memories),
            total_flagged=flagged_count,
            by_category=by_category,
            by_action=by_action,
            requires_review_count=review_count,
            detection_methods_used=methods_used,
        )

        self._logger.info(
            f"Filtering complete: {flagged_count}/{len(memories)} flagged in {elapsed:.2f}s"
        )

        return FilterResult(
            total_processed=len(memories),
            total_flagged=flagged_count,
            states=states,
            errors=errors,
            elapsed_seconds=elapsed,
            methods_used=methods_used,
            safety_report=safety_report,
        )

    def filter_single(self, memory: Memory) -> MemorySafetyState:
        """
        Classify a single memory.

        Args:
            memory: Memory to classify

        Returns:
            MemorySafetyState with classification results
        """
        if not self._config.enabled or not self._should_analyze_memory(memory):
            return MemorySafetyState(memory_id=memory.id)

        all_flags: List[SafetyFlag] = []

        # Run metadata-based detection
        all_flags.extend(self._check_filename(memory))
        all_flags.extend(self._check_caption(memory))
        all_flags.extend(self._check_metadata(memory))
        all_flags.extend(self._check_folder_path(memory))

        # Run pixel analysis if enabled and applicable
        if self._safety_settings.is_pixel_analysis_allowed() and self._is_pixel_analysis_applicable(
            memory
        ):
            try:
                all_flags.extend(self._analyze_with_vision(memory))
            except PixelAnalysisConsentError:
                self._logger.error("Pixel analysis attempted without disclosure acknowledgment")
                raise
            except Exception as e:
                self._logger.warning(f"Pixel analysis failed for {memory.id}: {e}")

        # Merge and create state
        merged_flags = self._merge_detection_results(all_flags)
        state = create_safety_state(memory.id, merged_flags, self._safety_settings)

        if state.is_sensitive:
            primary = state.get_primary_category()
            if primary:
                self._logger.info(f"Memory {memory.id} flagged for {primary.value}")

        return state

    # --- Metadata-Based Detection ---

    def _check_filename(self, memory: Memory) -> List[SafetyFlag]:
        """Check filename for sensitive patterns."""
        if not memory.source_filename:
            return []

        flags = []
        for category, patterns in SENSITIVE_FILENAME_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, memory.source_filename):
                    flags.append(
                        SafetyFlag(
                            category=category,
                            confidence=0.3,
                            detection_method=DetectionMethod.FILENAME_HEURISTIC,
                            severity=SensitivityLevel.MODERATE,
                            source="filename_heuristic_v1",
                            details=f"Filename pattern match",
                        )
                    )
                    break  # One flag per category

        return flags

    def _check_caption(self, memory: Memory) -> List[SafetyFlag]:
        """Check caption/description for sensitive content."""
        caption = memory.caption or ""
        if not caption:
            return []

        flags = []
        for category, patterns in SENSITIVE_CAPTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, caption):
                    flags.append(
                        SafetyFlag(
                            category=category,
                            confidence=0.5,
                            detection_method=DetectionMethod.CAPTION_ANALYSIS,
                            severity=SensitivityLevel.MODERATE,
                            source="caption_heuristic_v1",
                            details="Caption keyword match",
                        )
                    )
                    break

        return flags

    def _check_metadata(self, memory: Memory) -> List[SafetyFlag]:
        """Check EXIF and other metadata for hints."""
        flags = []
        metadata = memory.metadata or {}

        # Check for album names
        album = metadata.get("album", "")
        if album:
            sensitive_album_patterns = [
                r"(?i)private",
                r"(?i)hidden",
                r"(?i)vault",
                r"(?i)secret",
                r"(?i)personal",
            ]
            for pattern in sensitive_album_patterns:
                if re.search(pattern, album):
                    flags.append(
                        SafetyFlag(
                            category=SafetyCategory.PRIVATE,
                            confidence=0.4,
                            detection_method=DetectionMethod.METADATA_HEURISTIC,
                            severity=SensitivityLevel.MODERATE,
                            source="metadata_heuristic_v1",
                            details="Album name suggests private content",
                        )
                    )
                    break

        return flags

    def _check_folder_path(self, memory: Memory) -> List[SafetyFlag]:
        """Check folder/path for hints."""
        if not memory.source_path:
            return []

        path_str = str(memory.source_path).lower()
        flags = []

        # Check for sensitive folder names
        sensitive_paths = {
            SafetyCategory.PRIVATE: [r"private", r"hidden", r"vault", r"confidential"],
            SafetyCategory.NUDITY: [r"nsfw", r"intimate", r"boudoir"],
        }

        for category, patterns in sensitive_paths.items():
            for pattern in patterns:
                if pattern in path_str:
                    flags.append(
                        SafetyFlag(
                            category=category,
                            confidence=0.3,
                            detection_method=DetectionMethod.METADATA_HEURISTIC,
                            severity=SensitivityLevel.MODERATE,
                            source="folder_heuristic_v1",
                            details="Folder path suggests sensitive content",
                        )
                    )
                    break

        return flags

    # --- Pixel-Based Detection ---

    def _analyze_with_vision(self, memory: Memory) -> List[SafetyFlag]:
        """
        Use Gemini Vision to analyze image content.

        REQUIRES: safety_settings.is_pixel_analysis_allowed() == True
        """
        # Verify disclosure acknowledged
        if not verify_pixel_consent(self._safety_settings):
            raise PixelAnalysisConsentError(
                "Pixel analysis requires disclosure acknowledgment. "
                "User must review what data is sent to Gemini Vision."
            )

        if not self._ai_client:
            self._logger.warning("AI client not available for pixel analysis")
            return []

        if not memory.source_path or not memory.source_path.exists():
            return []

        try:
            # Create thumbnail
            thumbnail_bytes = self._create_thumbnail(memory.source_path)

            # Encode as base64
            thumbnail_b64 = base64.b64encode(thumbnail_bytes).decode("utf-8")

            # Build prompt
            prompt = self._build_vision_prompt()

            # Call Gemini Vision
            response = self._ai_client.analyze_image(
                image_data=thumbnail_b64,
                prompt=prompt,
            )

            # Parse response
            flags = self._parse_vision_response(response)

            return flags

        except Exception as e:
            self._logger.warning(f"Vision analysis failed: {e}")
            return []

    def _create_thumbnail(self, image_path: Path) -> bytes:
        """Create a small thumbnail for vision analysis."""
        if Image is None:
            raise ImportError("PIL/Pillow is required for pixel analysis")

        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize
        img.thumbnail(self._config.max_thumbnail_size, Image.Resampling.LANCZOS)

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self._config.thumbnail_quality)
        return buffer.getvalue()

    def _build_vision_prompt(self) -> str:
        """Build the prompt for Gemini Vision classification."""
        return VISION_CLASSIFICATION_PROMPT

    def _parse_vision_response(self, response: Any) -> List[SafetyFlag]:
        """Parse Gemini Vision response into SafetyFlags."""
        flags = []

        try:
            # Expect JSON response
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response

            findings = data.get("findings", [])

            severity_map = {
                "low": SensitivityLevel.PERMISSIVE,
                "medium": SensitivityLevel.MODERATE,
                "high": SensitivityLevel.STRICT,
            }

            for finding in findings:
                category_str = finding.get("category", "").upper()
                try:
                    category = SafetyCategory[category_str]
                except KeyError:
                    self._logger.warning(f"Unknown category from vision: {category_str}")
                    continue

                confidence = float(finding.get("confidence", 0.8))
                severity_str = finding.get("severity", "medium").lower()
                severity = severity_map.get(severity_str, SensitivityLevel.MODERATE)
                reason = finding.get("reason", "")

                flags.append(
                    SafetyFlag(
                        category=category,
                        confidence=confidence,
                        detection_method=DetectionMethod.AI_VISION_CLOUD,
                        severity=severity,
                        source="gemini_vision_v1",
                        details=reason[:100],  # Truncate to avoid logging sensitive content
                    )
                )

        except Exception as e:
            self._logger.warning(f"Failed to parse vision response: {e}")

        return flags

    # --- Helper Methods ---

    def _should_analyze_memory(self, memory: Memory) -> bool:
        """Check if this memory should be analyzed."""
        # Skip if not an image (for now)
        if self._config.skip_videos and memory.media_type == MediaType.VIDEO:
            return False

        # Skip if no filename (can't check anything)
        if not memory.source_filename:
            return False

        return True

    def _is_pixel_analysis_applicable(self, memory: Memory) -> bool:
        """Check if pixel analysis is possible for this memory."""
        # Must be an image
        if memory.media_type != MediaType.IMAGE:
            return False

        # Must have accessible file
        if not memory.source_path or not memory.source_path.exists():
            return False

        # Check file size (skip very large files)
        try:
            size_mb = memory.source_path.stat().st_size / (1024 * 1024)
            if size_mb > 50:  # Skip files larger than 50MB
                return False
        except Exception:
            return False

        return True

    def _merge_detection_results(self, all_flags: List[SafetyFlag]) -> List[SafetyFlag]:
        """
        Merge flags from different detection methods.

        If same category detected by multiple methods, keep the highest confidence.
        """
        if not all_flags:
            return []

        # Group by category
        by_category: Dict[SafetyCategory, List[SafetyFlag]] = {}
        for flag in all_flags:
            if flag.category not in by_category:
                by_category[flag.category] = []
            by_category[flag.category].append(flag)

        # Keep highest confidence per category
        merged = []
        for category, flags in by_category.items():
            best_flag = max(flags, key=lambda f: f.confidence)

            # Note other methods in details if multiple
            if len(flags) > 1:
                methods = ", ".join(f.detection_method.value for f in flags)
                best_flag.details = f"Detected by: {methods}"

            merged.append(best_flag)

        return merged


# --- Module-Level Functions ---


def verify_pixel_consent(safety_settings: SafetySettings) -> bool:
    """
    Verify that pixel analysis disclosure has been acknowledged.

    This uses an OPT-OUT model: pixel analysis is enabled by default,
    but users must acknowledge the disclosure about what's being sent
    to Gemini Vision. Users can opt-out if they prefer.

    Args:
        safety_settings: Safety settings to check

    Returns:
        True if disclosure acknowledged and not opted out, False otherwise
    """
    if not safety_settings.use_pixel_analysis:
        # User has opted out
        return False

    if not safety_settings.pixel_analysis_disclosure_acknowledged:
        # User hasn't acknowledged the disclosure yet
        return False

    if not safety_settings.pixel_analysis_disclosure_timestamp:
        # No timestamp recorded
        return False

    # Could add additional checks (e.g., need to re-acknowledge after X days)
    return True


def filter_memories_quick(
    memories: List[Memory],
    pixel_analysis: bool = False,
) -> FilterResult:
    """
    Convenience function with defaults.

    Args:
        memories: List of memories to filter
        pixel_analysis: Whether to enable pixel analysis (requires disclosure acknowledgment)

    Returns:
        FilterResult with classification results
    """
    safety_settings = SafetySettings()
    if pixel_analysis:
        # Note: This would still require disclosure acknowledgment to actually work
        safety_settings.use_pixel_analysis = True

    config = FilterConfig(safety_settings=safety_settings)
    filter_engine = ContentFilter(config=config, safety_settings=safety_settings)

    return filter_engine.filter_memories(memories)


def check_single_memory(memory: Memory) -> MemorySafetyState:
    """
    Quick check for a single memory using defaults.

    Args:
        memory: Memory to check

    Returns:
        MemorySafetyState with classification results
    """
    filter_engine = ContentFilter()
    return filter_engine.filter_single(memory)
