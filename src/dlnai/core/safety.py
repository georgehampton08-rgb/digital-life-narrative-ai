"""
Content Safety and Sensitive Media Policy Module.

This module defines how the application identifies, classifies, and handles potentially
sensitive content (nudity, violence, etc.) in users' personal media. It prioritizes
user control, transparency, and privacy, ensuring no pixel analysis occurs without
explicit opt-in consent.
"""

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


# --- Enums ---


class SafetyCategory(str, Enum):
    """
    Content categories for safety classification.

    These are descriptive labels, not moral judgments. Users' personal media
    may contain any of these for legitimate reasons.
    """

    NUDITY = "NUDITY"  # Unclothed human bodies (artistic, personal, etc.)
    SEXUAL = "SEXUAL"  # Sexually explicit or suggestive content
    VIOLENCE = "VIOLENCE"  # Graphic violence, injury, gore
    SELF_HARM = "SELF_HARM"  # Content depicting or suggesting self-harm
    SUBSTANCE = "SUBSTANCE"  # Drug/alcohol-related content
    HATE = "HATE"  # Hate symbols, slurs, discriminatory content
    DISTURBING = "DISTURBING"  # Generally disturbing imagery (not fitting other categories)
    PRIVATE = "PRIVATE"  # Sensitive personal documents (IDs, medical, financial)
    UNKNOWN_SENSITIVE = "UNKNOWN_SENSITIVE"  # Flagged as sensitive but category unclear


class SafetyAction(str, Enum):
    """
    What to do with flagged content.
    """

    ALLOW = "ALLOW"  # Treat normally, no special handling
    FLAG_ONLY = "FLAG_ONLY"  # Include in report but mark as sensitive (badge/warning)
    BLUR_IN_REPORT = "BLUR_IN_REPORT"  # Show blurred/obscured thumbnail in HTML report
    HIDE_FROM_REPORT = "HIDE_FROM_REPORT"  # Exclude from visual report but count in statistics
    EXCLUDE_FROM_AI = "EXCLUDE_FROM_AI"  # Don't send metadata to AI analysis
    REQUIRE_REVIEW = "REQUIRE_REVIEW"  # Hold for user review before including
    QUARANTINE = "QUARANTINE"  # Move to separate "review" section, don't process further

    def __lt__(self, other: "SafetyAction") -> bool:
        """Define strictness ordering."""
        order = [
            SafetyAction.ALLOW,
            SafetyAction.FLAG_ONLY,
            SafetyAction.BLUR_IN_REPORT,
            SafetyAction.HIDE_FROM_REPORT,
            SafetyAction.EXCLUDE_FROM_AI,
            SafetyAction.REQUIRE_REVIEW,
            SafetyAction.QUARANTINE,
        ]
        return order.index(self) < order.index(other)


class SensitivityLevel(str, Enum):
    """
    How strict the detection/filtering should be.
    """

    PERMISSIVE = "PERMISSIVE"  # Only flag clearly explicit content
    MODERATE = "MODERATE"  # Flag explicit and suggestive content (default)
    STRICT = "STRICT"  # Flag anything potentially sensitive
    PARANOID = "PARANOID"  # Flag aggressively, err on side of caution

    def __lt__(self, other: "SensitivityLevel") -> bool:
        """Define severity ordering."""
        order = [
            SensitivityLevel.PERMISSIVE,
            SensitivityLevel.MODERATE,
            SensitivityLevel.STRICT,
            SensitivityLevel.PARANOID,
        ]
        return order.index(self) < order.index(other)


class DetectionMethod(str, Enum):
    """
    How content was classified and the relative reliability.
    """

    FILENAME_HEURISTIC = "FILENAME_HEURISTIC"  # Based on filename patterns (Low confidence)
    METADATA_HEURISTIC = "METADATA_HEURISTIC"  # Based on EXIF/metadata signals (Medium confidence)
    CAPTION_ANALYSIS = "CAPTION_ANALYSIS"  # Based on caption/description text (Medium confidence)
    AI_VISION_LOCAL = "AI_VISION_LOCAL"  # Local model analysis (High confidence - Future)
    AI_VISION_CLOUD = "AI_VISION_CLOUD"  # Cloud AI analysis (Gemini Vision) (High confidence)
    USER_TAGGED = "USER_TAGGED"  # Manually flagged by user (Highest confidence)
    INHERITED = "INHERITED"  # Inherited from album/folder classification


# --- Models ---


class SafetyFlag(BaseModel):
    """
    A single safety flag on a piece of content.
    """

    category: SafetyCategory
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    detection_method: DetectionMethod
    severity: SensitivityLevel = SensitivityLevel.MODERATE
    source: str  # e.g., "content_filter_v1", "user_manual"
    details: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.now)

    def is_high_confidence(self) -> bool:
        """Checks if the flag has high confidence."""
        return self.confidence >= 0.8


class SafetySettings(BaseModel):
    """
    User-configurable safety settings.
    """

    enabled: bool = True
    sensitivity: SensitivityLevel = SensitivityLevel.MODERATE
    default_action: SafetyAction = SafetyAction.FLAG_ONLY

    # Per-category action overrides
    nudity_action: SafetyAction = SafetyAction.BLUR_IN_REPORT
    sexual_action: SafetyAction = SafetyAction.HIDE_FROM_REPORT
    violence_action: SafetyAction = SafetyAction.FLAG_ONLY
    self_harm_action: SafetyAction = SafetyAction.REQUIRE_REVIEW
    substance_action: SafetyAction = SafetyAction.FLAG_ONLY
    hate_action: SafetyAction = SafetyAction.HIDE_FROM_REPORT
    private_action: SafetyAction = SafetyAction.HIDE_FROM_REPORT

    # Detection settings (OPT-OUT model with disclosure)
    use_pixel_analysis: bool = True  # Enabled by default
    pixel_analysis_disclosure_acknowledged: bool = False  # Must acknowledge what's being sent
    pixel_analysis_disclosure_timestamp: Optional[datetime] = None
    detection_methods_enabled: Set[DetectionMethod] = Field(
        default_factory=lambda: {
            DetectionMethod.FILENAME_HEURISTIC,
            DetectionMethod.METADATA_HEURISTIC,
        }
    )

    # Logging
    log_detections: bool = True
    log_actions_taken: bool = True

    # UI preferences
    show_sensitivity_badges: bool = True
    blur_sensitive_thumbnails: bool = True
    require_click_to_reveal: bool = True

    def get_action_for_category(self, category: SafetyCategory) -> SafetyAction:
        """Gets the configured action for a specific safety category."""
        mapping = {
            SafetyCategory.NUDITY: self.nudity_action,
            SafetyCategory.SEXUAL: self.sexual_action,
            SafetyCategory.VIOLENCE: self.violence_action,
            SafetyCategory.SELF_HARM: self.self_harm_action,
            SafetyCategory.SUBSTANCE: self.substance_action,
            SafetyCategory.HATE: self.hate_action,
            SafetyCategory.PRIVATE: self.private_action,
        }
        return mapping.get(category, self.default_action)

    def is_pixel_analysis_allowed(self) -> bool:
        """Check both enabled AND disclosure acknowledged for pixel analysis."""
        return self.use_pixel_analysis and self.pixel_analysis_disclosure_acknowledged

    def to_summary(self) -> str:
        """Human-readable summary of settings."""
        if self.use_pixel_analysis:
            pixel = (
                "Enabled (disclosure acknowledged)"
                if self.is_pixel_analysis_allowed()
                else "Enabled (pending disclosure)"
            )
        else:
            pixel = "Disabled (user opted out)"
        return f"Safety: {'Active' if self.enabled else 'Inactive'} (Sensitivity: {self.sensitivity.value}, Pixels: {pixel})"


class MemorySafetyState(BaseModel):
    """
    Safety classification state for a single Memory.
    """

    memory_id: str
    flags: List[SafetyFlag] = Field(default_factory=list)
    resolved_action: SafetyAction = SafetyAction.ALLOW
    is_sensitive: bool = False
    requires_user_review: bool = False
    user_reviewed: bool = False
    user_override_action: Optional[SafetyAction] = None
    classified_at: Optional[datetime] = None
    classification_version: str = "1.0"

    def add_flag(self, flag: SafetyFlag, settings: Optional[SafetySettings] = None) -> None:
        """Adds a new flag to the state and optionally re-resolves action."""
        self.flags.append(flag)
        self.is_sensitive = True
        if settings:
            self.resolve_action(settings)

    def resolve_action(self, settings: SafetySettings) -> None:
        """Resolves the final action based on flags and settings."""
        if self.user_override_action:
            self.resolved_action = self.user_override_action
            return

        if not self.flags or not settings.enabled:
            self.resolved_action = SafetyAction.ALLOW
            self.is_sensitive = bool(self.flags)
            return

        actions = [settings.get_action_for_category(f.category) for f in self.flags]
        self.resolved_action = max(actions) if actions else SafetyAction.ALLOW
        self.is_sensitive = True
        self.requires_user_review = self.resolved_action == SafetyAction.REQUIRE_REVIEW

    def has_category(self, category: SafetyCategory) -> bool:
        """Checks if the memory has been flagged for a specific category."""
        return any(f.category == category for f in self.flags)

    def get_highest_severity(self) -> Optional[SensitivityLevel]:
        """Gets the highest sensitivity level among all flags."""
        if not self.flags:
            return None
        return max(f.severity for f in self.flags)

    def get_primary_category(self) -> Optional[SafetyCategory]:
        """Returns the most severe/confident category."""
        if not self.flags:
            return None
        # Sort by confidence then severity (order)
        sorted_flags = sorted(self.flags, key=lambda x: (x.confidence, x.severity), reverse=True)
        return sorted_flags[0].category

    def is_visually_safe(self) -> bool:
        """Can show in report without blur/hide."""
        safe_actions = {SafetyAction.ALLOW, SafetyAction.FLAG_ONLY, SafetyAction.EXCLUDE_FROM_AI}
        return self.resolved_action in safe_actions


class SafetyReport(BaseModel):
    """
    Aggregated safety findings for a batch of memories.
    """

    total_analyzed: int
    total_flagged: int
    by_category: Dict[SafetyCategory, int] = Field(default_factory=dict)
    by_action: Dict[SafetyAction, int] = Field(default_factory=dict)
    requires_review_count: int = 0
    detection_methods_used: Set[DetectionMethod] = Field(default_factory=set)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    def flagged_percentage(self) -> float:
        """Returns the percentage of flagged memories."""
        if self.total_analyzed == 0:
            return 0.0
        return (self.total_flagged / self.total_analyzed) * 100

    def to_summary(self) -> str:
        """Human-readable summary of the safety report."""
        return (
            f"Safety Analysis: {self.total_flagged}/{self.total_analyzed} flagged "
            f"({self.flagged_percentage():.1f}%). {self.requires_review_count} pending review."
        )


# --- Filename Heuristics ---

SENSITIVE_FILENAME_PATTERNS: Dict[SafetyCategory, List[str]] = {
    SafetyCategory.NUDITY: [
        r"(?i)nude",
        r"(?i)naked",
        r"(?i)nsfw",
        r"(?i)boudoir",
        r"(?i)intimate",
    ],
    SafetyCategory.PRIVATE: [
        r"(?i)passport",
        r"(?i)license",
        r"(?i)ssn",
        r"(?i)medical",
        r"(?i)bank.?statement",
        r"(?i)confidential",
    ],
    SafetyCategory.VIOLENCE: [
        r"(?i)gore",
        r"(?i)blood",
        r"(?i)injury",
        r"(?i)accident",
    ],
    SafetyCategory.SUBSTANCE: [
        r"(?i)drugs",
        r"(?i)pills",
        r"(?i)weed",
        r"(?i)cocaine",
    ],
}


# --- Helper Functions ---


def resolve_action_for_flags(flags: List[SafetyFlag], settings: SafetySettings) -> SafetyAction:
    """
    Determine final action when multiple flags exist.
    """
    if not flags or not settings.enabled:
        return SafetyAction.ALLOW

    actions = [settings.get_action_for_category(f.category) for f in flags]
    return max(actions)


def is_visually_safe_for_report(state: MemorySafetyState, settings: SafetySettings) -> bool:
    """
    Check if memory can be shown normally in HTML report.
    """
    if not settings.enabled:
        return True

    # These actions prevent normal display
    unsafe_actions = {
        SafetyAction.BLUR_IN_REPORT,
        SafetyAction.HIDE_FROM_REPORT,
        SafetyAction.REQUIRE_REVIEW,
        SafetyAction.QUARANTINE,
    }
    return state.resolved_action not in unsafe_actions


def should_include_in_ai_analysis(state: MemorySafetyState) -> bool:
    """
    Check if memory metadata should be sent to AI.
    """
    exclude_actions = {SafetyAction.EXCLUDE_FROM_AI, SafetyAction.QUARANTINE}
    return state.resolved_action not in exclude_actions


def get_display_treatment(state: MemorySafetyState, settings: SafetySettings) -> Dict[str, Any]:
    """
    Get UI display instructions for a memory.
    """
    resolved = state.resolved_action

    return {
        "show_thumbnail": resolved != SafetyAction.HIDE_FROM_REPORT
        and resolved != SafetyAction.QUARANTINE,
        "blur_thumbnail": resolved == SafetyAction.BLUR_IN_REPORT
        and settings.blur_sensitive_thumbnails,
        "show_badge": state.is_sensitive and settings.show_sensitivity_badges,
        "badge_text": (
            state.get_primary_category().value
            if state.is_sensitive and state.get_primary_category()
            else None
        ),
        "require_click_to_view": resolved
        in {SafetyAction.BLUR_IN_REPORT, SafetyAction.REQUIRE_REVIEW}
        and settings.require_click_to_reveal,
        "warning_text": (
            f"Content flagged as {state.get_primary_category().value}"
            if state.is_sensitive and state.get_primary_category()
            else None
        ),
    }


def create_safety_state(
    memory_id: str, flags: List[SafetyFlag], settings: SafetySettings
) -> MemorySafetyState:
    """
    Factory function to create a fully resolved safety state.
    """
    state = MemorySafetyState(memory_id=memory_id, flags=flags, classified_at=datetime.now())
    state.resolve_action(settings)

    if settings.log_detections and state.is_sensitive:
        logger.info(f"Memory {memory_id} flagged for {state.get_primary_category()}")

    return state


def merge_safety_states(
    existing: MemorySafetyState, new_flags: List[SafetyFlag], settings: SafetySettings
) -> MemorySafetyState:
    """
    Merge new flags into existing state, re-resolve action.
    """
    # Filter out duplicate flags (same category and method)
    for nf in new_flags:
        if not any(
            f.category == nf.category and f.detection_method == nf.detection_method
            for f in existing.flags
        ):
            existing.flags.append(nf)

    existing.resolve_action(settings)
    existing.classified_at = datetime.now()
    return existing


def check_filename_safety(filename: str) -> List[SafetyFlag]:
    """
    Check filename against heuristic patterns.
    Returns list of flags (empty if nothing detected).

    Note: This is LOW confidence detection.
    """
    flags = []
    for category, patterns in SENSITIVE_FILENAME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, filename):
                flags.append(
                    SafetyFlag(
                        category=category,
                        confidence=0.3,  # Low confidence for filename heuristics
                        detection_method=DetectionMethod.FILENAME_HEURISTIC,
                        severity=SensitivityLevel.MODERATE,
                        source="filename_heuristic_v1",
                        details=f"Filename matched pattern: {pattern}",
                    )
                )
                break  # Only one flag per category from filename

    return flags
