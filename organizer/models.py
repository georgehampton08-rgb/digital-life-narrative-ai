"""Core data models for Digital Life Narrative AI.

This module defines the common schema that all parsers normalize data into,
and the structures that the AI analyzer produces. All models use Pydantic v2
for validation and serialization.
"""

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class MediaType(str, Enum):
    """Types of media that can be processed.

    Attributes:
        PHOTO: Static image files (jpg, png, heic, etc.)
        VIDEO: Video files (mp4, mov, etc.)
        AUDIO: Audio recordings or voice messages
        TEXT: Text-based content (notes, posts without media)
        STORY: Ephemeral story content (Snapchat stories, Instagram stories)
        MESSAGE: Chat messages or DMs
        CHECK_IN: Location check-ins or place tags
        UNKNOWN: Unrecognized or unsupported media type
    """

    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    STORY = "story"
    MESSAGE = "message"
    CHECK_IN = "check_in"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


class SourcePlatform(str, Enum):
    """Supported source platforms for data exports.

    Attributes:
        SNAPCHAT: Snapchat data export
        GOOGLE_PHOTOS: Google Photos/Takeout export
        FACEBOOK: Facebook data download
        INSTAGRAM: Instagram data download
        ONEDRIVE: Microsoft OneDrive export
        LOCAL: Local filesystem (camera roll, downloads, etc.)
        UNKNOWN: Unrecognized platform
    """

    SNAPCHAT = "snapchat"
    GOOGLE_PHOTOS = "google_photos"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    ONEDRIVE = "onedrive"
    LOCAL = "local"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    """Confidence level for AI-inferred or extracted data.

    Attributes:
        HIGH: Strong confidence (explicit metadata, verified data)
        MEDIUM: Moderate confidence (inferred from context, partial data)
        LOW: Low confidence (best guess, fallback values)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Core Data Models
# =============================================================================


class GeoLocation(BaseModel):
    """Geographic location information.

    Represents a physical location with optional coordinates and place names.
    All fields are optional since location data may be partial or unavailable.

    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        place_name: Human-readable place name (e.g., "Central Park")
        country: Country name or code
        raw_location_string: Original unparsed location string from source
    """

    latitude: float | None = None
    longitude: float | None = None
    place_name: str | None = None
    country: str | None = None
    raw_location_string: str | None = None

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float | None) -> float | None:
        """Validate latitude is within valid range."""
        if v is not None and (v < -90 or v > 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float | None) -> float | None:
        """Validate longitude is within valid range."""
        if v is not None and (v < -180 or v > 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v

    @property
    def has_coordinates(self) -> bool:
        """Check if valid coordinates are present."""
        return self.latitude is not None and self.longitude is not None

    def to_display_string(self) -> str:
        """Return a human-readable location string.

        Returns:
            Display string prioritizing place_name, falling back to coordinates.
        """
        if self.place_name and self.country:
            return f"{self.place_name}, {self.country}"
        if self.place_name:
            return self.place_name
        if self.has_coordinates:
            return f"{self.latitude:.4f}, {self.longitude:.4f}"
        if self.raw_location_string:
            return self.raw_location_string
        return "Unknown location"


class MediaItem(BaseModel):
    """Universal normalized schema for media items.

    This is the core data structure that all platform-specific parsers normalize
    into. It provides a common representation regardless of the source platform.

    Attributes:
        id: Unique identifier (UUID)
        source_platform: Platform this item was exported from
        media_type: Type of media (photo, video, etc.)
        file_path: Original file path (if available)
        timestamp: When the media was created/captured
        timestamp_confidence: Confidence level of the timestamp
        location: Geographic location information
        location_confidence: Confidence level of the location
        people: List of people names/tags detected in the media
        caption: Text caption or description
        original_metadata: Raw metadata from the source platform
        file_hash: Hash for deduplication (MD5 or SHA256)
    """

    id: str | UUID = Field(default_factory=lambda: str(uuid4()))
    source_platform: SourcePlatform
    media_type: MediaType
    file_path: Path | None = None
    timestamp: datetime | None = None
    timestamp_confidence: Confidence = Confidence.LOW
    location: GeoLocation | None = None
    location_confidence: Confidence = Confidence.LOW
    people: list[str] = Field(default_factory=list)
    caption: str | None = None
    original_metadata: dict[str, Any] = Field(default_factory=dict)
    file_hash: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    def to_ai_summary(self, privacy_mode: bool = False) -> dict[str, Any]:
        """Generate a privacy-respecting summary for AI analysis.

        Creates a dictionary suitable for sending to Gemini or other LLMs,
        with sensitive information redacted or truncated.

        Args:
            privacy_mode: If True, apply stricter privacy filtering.

        Returns:
            Dictionary with AI-safe summary of the media item.
        """
        summary: dict[str, Any] = {
            "id": self.id,
            "platform": self.source_platform.value,
            "type": self.media_type.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_confidence": self.timestamp_confidence.value,
            "has_location": self.location is not None and self.location.has_coordinates,
            "people_count": len(self.people),
        }

        # Location: only include place-level info, not exact coordinates
        if self.location:
            if privacy_mode:
                summary["location_country"] = self.location.country
            else:
                summary["location"] = self.location.to_display_string()
                summary["location_country"] = self.location.country

        # Caption: truncate to reasonable length
        if self.caption:
            max_length = 100 if privacy_mode else 500
            truncated = self.caption[:max_length]
            if len(self.caption) > max_length:
                truncated += "..."
            summary["caption"] = truncated

        # People: include count only in privacy mode, names otherwise
        if not privacy_mode and self.people:
            summary["people"] = self.people

        # File info: include extension, name, and parent folder for context
        if self.file_path:
            summary["file_extension"] = self.file_path.suffix.lower()
            if not privacy_mode:
                summary["file_name"] = self.file_path.name
                # Include parent folder name as it often contains thematic clues (e.g. "Wedding", "Paris")
                summary["context_folder"] = self.file_path.parent.name

        return summary


class ParseResult(BaseModel):
    """Result of parsing a platform export.

    Contains all parsed media items along with statistics and any errors
    encountered during parsing.

    Attributes:
        source_platform: The platform that was parsed
        items: List of successfully parsed media items
        parse_errors: List of error messages for failed items
        stats: Parsing statistics (total, parsed, skipped, errors)
        parse_duration_seconds: Time taken to parse the export
    """

    source_platform: SourcePlatform
    items: list[MediaItem] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)
    stats: dict[str, int] = Field(
        default_factory=lambda: {
            "total_files": 0,
            "parsed": 0,
            "skipped": 0,
            "errors": 0,
        }
    )
    parse_duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate the parsing success rate.

        Returns:
            Success rate as a percentage (0-100).
        """
        total = self.stats.get("total_files", 0)
        if total == 0:
            return 0.0
        parsed = self.stats.get("parsed", 0)
        return (parsed / total) * 100


# =============================================================================
# AI-Generated Models
# =============================================================================


class LifeChapter(BaseModel):
    """AI-generated chapter of the user's life story.

    Represents a coherent period in the user's life with common themes,
    locations, or experiences.

    Attributes:
        id: Unique identifier for the chapter
        title: AI-generated descriptive title (e.g., "The Chicago Years")
        start_date: Beginning of the chapter period
        end_date: End of the chapter period
        themes: Key themes identified (e.g., ["career change", "travel"])
        narrative: AI-written paragraph describing this chapter
        key_events: List of significant events identified
        location_summary: Primary location(s) during this period
        media_count: Number of media items in this chapter
        representative_media_ids: Sample MediaItem IDs representing the chapter
        confidence: AI confidence in this chapter analysis
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    start_date: date
    end_date: date
    themes: list[str] = Field(default_factory=list)
    narrative: str
    key_events: list[str] = Field(default_factory=list)
    location_summary: str | None = None
    media_count: int = 0
    representative_media_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM

    @model_validator(mode="after")
    def validate_date_range(self) -> "LifeChapter":
        """Validate that start_date is before or equal to end_date."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        return self

    @property
    def duration_days(self) -> int:
        """Calculate the duration of the chapter in days."""
        return (self.end_date - self.start_date).days


class PlatformBehaviorInsight(BaseModel):
    """AI-generated analysis of user behavior on a specific platform.

    Provides insights into how the user utilizes different platforms
    for documenting their life.

    Attributes:
        platform: The source platform being analyzed
        usage_pattern: AI description of usage patterns
        peak_years: Years with highest activity
        common_content_types: Most frequent media types used
        unique_aspects: Platform-specific behaviors or content
    """

    platform: SourcePlatform
    usage_pattern: str
    peak_years: list[int] = Field(default_factory=list)
    common_content_types: list[MediaType] = Field(default_factory=list)
    unique_aspects: list[str] = Field(default_factory=list)


class DataGap(BaseModel):
    """Detected gap in the user's data timeline.

    Represents a period with little or no documented media,
    which may indicate significant life events or simply periods
    of lower digital activity.

    Attributes:
        start_date: Beginning of the gap
        end_date: End of the gap
        gap_days: Duration of the gap in days
        possible_reasons: AI-inferred explanations for the gap
    """

    start_date: date
    end_date: date
    gap_days: int
    possible_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_and_calculate_gap(self) -> "DataGap":
        """Validate dates and ensure gap_days is consistent."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        expected_days = (self.end_date - self.start_date).days
        if self.gap_days != expected_days:
            # Auto-correct gap_days to match the date range
            object.__setattr__(self, "gap_days", expected_days)
        return self


class LifeStoryReport(BaseModel):
    """Complete AI-generated life story report.

    The main output of the AI analysis, containing the reconstructed
    narrative, chapters, insights, and supporting data.

    Attributes:
        generated_at: When this report was generated
        ai_model_used: The AI model used for analysis (e.g., "gemini-1.5-pro")
        total_media_analyzed: Number of media items processed
        date_range: Overall date range of the analyzed media
        executive_summary: AI-written overall life narrative
        chapters: List of life chapters
        platform_insights: Per-platform behavior analysis
        detected_patterns: Cross-cutting patterns identified
        data_gaps: Significant gaps in the timeline
        data_quality_notes: Notes about data quality issues
        raw_ai_response: Raw AI response for debugging
        is_fallback_mode: True if AI was unavailable
    """

    generated_at: datetime = Field(default_factory=datetime.now)
    ai_model_used: str
    total_media_analyzed: int
    date_range: tuple[date, date] | None = None
    executive_summary: str
    chapters: list[LifeChapter] = Field(default_factory=list)
    platform_insights: list[PlatformBehaviorInsight] = Field(default_factory=list)
    detected_patterns: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    raw_ai_response: str | None = None
    is_fallback_mode: bool = False

    @property
    def chapter_count(self) -> int:
        """Get the number of chapters in the report."""
        return len(self.chapters)

    @property
    def years_covered(self) -> int | None:
        """Calculate the number of years covered by the data.

        Returns:
            Number of years, or None if date_range is not available.
        """
        if not self.date_range:
            return None
        start, end = self.date_range
        return end.year - start.year + 1


# =============================================================================
# Configuration Models
# =============================================================================


class AnalysisConfig(BaseModel):
    """Configuration settings for AI analysis.

    Controls various aspects of how the AI analyzes media and
    generates the life story report.

    Attributes:
        min_chapter_duration_days: Minimum days for a chapter
        max_chapters: Maximum number of chapters to generate
        include_platform_analysis: Include per-platform insights
        detect_gaps_threshold_days: Minimum gap size to report
        privacy_mode: If True, send minimal data to AI
    """

    min_chapter_duration_days: int = Field(default=30, ge=1)
    max_chapters: int = Field(default=20, ge=1, le=100)
    include_platform_analysis: bool = True
    detect_gaps_threshold_days: int = Field(default=60, ge=1)
    privacy_mode: bool = False

    @field_validator("min_chapter_duration_days")
    @classmethod
    def validate_min_chapter_duration(cls, v: int) -> int:
        """Validate minimum chapter duration is reasonable."""
        if v > 365:
            raise ValueError("min_chapter_duration_days should not exceed 365 days")
        return v
