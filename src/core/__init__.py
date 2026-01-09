"""Core data models for Digital Life Narrative AI.

This package contains the foundational data structures that all other components
depend on:

- **Memory**: Universal schema that parsers normalize into
- **PrivacyGate**: Security boundary controlling all data sent to AI services
- **Timeline**: Aggregation and analysis layer for memories

Example:
    >>> from src.core import Memory, MediaType, SourcePlatform
    >>> from src.core import PrivacyGate, PrivacyLevel
    >>> from src.core import Timeline, TimelineQuality
    >>>
    >>> # Create memories
    >>> memory = Memory(
    ...     source_platform=SourcePlatform.GOOGLE_PHOTOS,
    ...     media_type=MediaType.PHOTO
    ... )
    >>>
    >>> # Build timeline
    >>> timeline = Timeline([memory])
    >>> stats = timeline.compute_statistics()
    >>>
    >>> # Prepare for AI with privacy
    >>> gate = PrivacyGate()
    >>> gate.settings.privacy_level = PrivacyLevel.STANDARD
    >>> gate.grant_consent("User acknowledged")
    >>> safe_data, warnings = gate.prepare_memories_for_ai([memory])
"""

from src.core.models import (
    AnalysisConfig,
    ConfidenceLevel,
    DepthMode,
    GeoPoint,
    LifeChapter,
    LifeStoryReport,
    Location,
    MediaType,
    Memory,
    PersonTag,
    PlatformBehaviorInsight,
    DataGap,
    SourcePlatform,
    VisualAnalysisStats,
)
from src.core.privacy import (
    ConsentReceipt,
    ConsentRequiredError,
    ConsentStatus,
    DataCategory,
    LocalOnlyModeError,
    PrivacyError,
    PrivacyGate,
    PrivacyLevel,
    PrivacySettings,
    PrivacyViolationError,
    TransmissionRecord,
    get_default_gate,
    require_consent,
)
from src.core.timeline import (
    DateRange,
    PlatformActivity,
    TimeGranularity,
    Timeline,
    TimelineGap,
    TimePeriod,
    TimelineQuality,
    TimelineStatistics,
)

__all__ = [
    # Memory models
    "Memory",
    "MediaType",
    "SourcePlatform",
    "ConfidenceLevel",
    "DepthMode",
    "AnalysisConfig",
    "GeoPoint",
    "Location",
    "PersonTag",
    "LifeChapter",
    "LifeStoryReport",
    "VisualAnalysisStats",
    # Privacy models
    "PrivacyGate",
    "PrivacyLevel",
    "PrivacySettings",
    "ConsentStatus",
    "DataCategory",
    "ConsentReceipt",
    "TransmissionRecord",
    # Privacy exceptions
    "PrivacyError",
    "ConsentRequiredError",
    "LocalOnlyModeError",
    "PrivacyViolationError",
    # Privacy functions
    "get_default_gate",
    "require_consent",
    # Timeline models
    "Timeline",
    "TimelineQuality",
    "TimeGranularity",
    "TimePeriod",
    "TimelineGap",
    "TimelineStatistics",
    "PlatformActivity",
    "DateRange",
]
