"""Core data models for Digital Life Narrative AI.

This module consolidates all foundational data structures into a single location, 
supporting visual enrichment, privacy controls, and AI-driven analysis.

Models follow a tiered flow:
1. RAW DATA (MediaType, SourcePlatform)
2. NORMALIZED MEMORY (Memory, Location, GeoPoint)
3. AI-DERIVED INSIGHTS (LifeChapter, VisualAnalysisStats)
4. FINAL REPORT (LifeStoryReport)
"""

import hashlib
import uuid as uuid_module
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field, AliasChoices, ConfigDict


# =============================================================================
# Enums
# =============================================================================


class MediaType(str, Enum):
    """Types of media that can be processed.

    Different media types carry different behavioral signals for AI analysis.
    """

    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    STORY = "story"
    MESSAGE = "message"
    CHECK_IN = "check_in"
    SCREENSHOT = "screenshot"
    LIVE_PHOTO = "live_photo"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class SourcePlatform(str, Enum):
    """Supported source platforms for data exports."""

    SNAPCHAT = "snapchat"
    GOOGLE_PHOTOS = "google_photos"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    WHATSAPP = "whatsapp"
    IMESSAGE = "imessage"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    LOCAL = "local"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level for AI-inferred or extracted data."""

    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFERRED = "inferred"


class DepthMode(str, Enum):
    """Depth of visual analysis, balancing cost and narrative richness.
    
    Attributes:
        QUICK: Minimal visual sampling (1-2 images/chapter), cheapest.
        STANDARD: Balanced sampling (3-5 images/chapter), default.
        DEEP: Extensive sampling (8-12 images/chapter), most expensive.
    """
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class AnalysisConfig(BaseModel):
    """Configuration for a specific analysis run, controlling cost and quality."""
    
    # 1. Visual Intelligence (Depth & Models)
    depth: DepthMode = DepthMode.DEEP
    vision_model: str = "gemini-2.0-flash-exp"
    narrative_model: str = "gemini-1.5-pro"
    max_images: int | None = None
    """Global cap on images analyzed across the entire report"""
    
    force_refresh: bool = False
    """If True, bypasses AI results cache"""

    # 2. Analytical Control (Legacy Parity)
    min_chapters: int = 3
    max_chapters: int = 15
    min_chapter_duration_days: int = 30
    max_memories_for_chapter_detection: int = 500
    max_memories_per_chapter_narrative: int = 200
    
    include_platform_analysis: bool = True
    """Whether to generate PlatformBehaviorInsight records"""
    
    include_gap_analysis: bool = True
    """Whether to generate DataGap records for timeline silences"""
    
    detect_patterns: bool = True
    """Whether to generate high-level cross-cutting insights"""

    narrative_style: str = "warm"
    """Tone of the chapter summaries (warm, neutral, analytical)"""
    
    privacy_level: str = "standard"
    """Privacy filtering depth (strict, standard, detailed)"""
    
    fail_on_partial: bool = False
    """If True, fails if any sub-analysis step errors out"""


# =============================================================================
# Analysis Results
# =============================================================================


class PlatformBehaviorInsight(BaseModel):
    """Analysis of platform-specific usage patterns.

    Attributes:
        platform: The source platform.
        usage_pattern: How they use this platform.
        peak_period: When they used it most.
        unique_characteristics: What's unique about their usage.
        memory_count: Memories from this platform.
        percentage_of_total: Percentage of all memories.
    """

    platform: SourcePlatform
    usage_pattern: str = ""
    peak_period: str | None = None
    unique_characteristics: list[str] = Field(default_factory=list)
    memory_count: int = 0
    percentage_of_total: float = 0.0


class DataGap(BaseModel):
    """Information about a gap in the timeline.

    Attributes:
        start_date: When the gap starts.
        end_date: When the gap ends.
        duration_days: Gap duration in days.
        possible_explanations: AI-suggested reasons.
        severity: Gap severity level.
        impacts_narrative: Whether gap affects story.
    """

    start_date: date
    end_date: date
    gap_days: int = Field(default=0, validation_alias=AliasChoices("gap_days", "duration_days"))

    @property
    def duration_days(self) -> int:
        """Alias for gap_days used in newer code."""
        return self.gap_days
    possible_explanations: list[str] = Field(default_factory=list)
    severity: str = "minor"
    impacts_narrative: bool = False


# =============================================================================
# Final Report
# =============================================================================


class GeoPoint(BaseModel):
    """Geographic coordinates with optional altitude and accuracy."""

    latitude: float
    longitude: float
    altitude_meters: float | None = None
    accuracy_meters: float | None = None

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if v < -90 or v > 90:
            raise ValueError(f"Latitude must be between -90 and 90 degrees, got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if v < -180 or v > 180:
            raise ValueError(f"Longitude must be between -180 and 180 degrees, got {v}")
        return v

    def distance_km(self, other: "GeoPoint") -> float:
        import math
        R = 6371.0
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def to_approximate(self, precision: int = 2) -> "GeoPoint":
        return GeoPoint(
            latitude=round(self.latitude, precision),
            longitude=round(self.longitude, precision),
            altitude_meters=None,
            accuracy_meters=None,
        )


class Location(BaseModel):
    """Complete location information combining coordinates and place names."""

    coordinates: GeoPoint | None = None
    place_name: str | None = None
    locality: str | None = None
    region: str | None = None
    country: str | None = None
    country_code: str | None = None
    raw_string: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    def is_empty(self) -> bool:
        return all([
            self.coordinates is None,
            self.place_name is None,
            self.locality is None,
            self.region is None,
            self.country is None,
            self.country_code is None,
        ])

    def to_display_string(self) -> str:
        if self.is_empty():
            return "Unknown location"
        parts = []
        if self.place_name:
            parts.append(self.place_name)
        if self.locality and self.locality != self.place_name:
            parts.append(self.locality)
        if self.region and self.region not in parts:
            parts.append(self.region)
        if self.country and self.country not in parts:
            parts.append(self.country)
        return ", ".join(parts) if parts else "Unknown location"

    def to_ai_summary(self) -> str:
        if self.is_empty():
            return "Unknown"
        parts = []
        if self.region:
            parts.append(self.region)
        elif self.locality:
            parts.append(self.locality)
        elif self.place_name:
            parts.append(self.place_name)
        if self.country_code:
            parts.append(self.country_code)
        elif self.country:
            parts.append(self.country)
        return ", ".join(parts) if parts else "Unknown"


class PersonTag(BaseModel):
    """Person tag/mention in media."""

    name: str
    normalized_name: str | None = Field(default=None, validate_default=True)
    platform_id: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    @field_validator("normalized_name", mode="before")
    @classmethod
    def auto_normalize(cls, v: str | None, info) -> str | None:
        if v is None and "name" in info.data:
            return info.data["name"].lower().strip()
        return v.lower().strip() if v else None

    def to_anonymous(self) -> "PersonTag":
        name_hash = hashlib.sha256(self.name.encode()).hexdigest()[:8]
        return PersonTag(
            name=f"person_{name_hash}",
            normalized_name=f"person_{name_hash}",
            platform_id=None,
            confidence=self.confidence,
        )


# =============================================================================
# Main Memory Model
# =============================================================================


class Memory(BaseModel):
    """Universal memory object representing a single moment in time.
    
    Now supports visual enrichment fields populated by vision AI.
    """

    # Identity fields
    id: str = Field(default_factory=lambda: str(uuid_module.uuid4()))
    content_hash: str | None = None
    metadata_hash: str | None = None

    # Source fields
    source_platform: SourcePlatform
    source_path: Path | None = None
    source_filename: str | None = None
    source_export_path: str | None = None

    # Temporal fields
    created_at: datetime | None = None
    created_at_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    timezone_name: str | None = None

    # Content fields
    media_type: MediaType = MediaType.UNKNOWN
    caption: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None

    @field_validator("id", mode="before")
    @classmethod
    def generate_id_if_empty(cls, v: str | None) -> str:
        """Generate UUID if empty or None."""
        if not v:
            return str(uuid_module.uuid4())
        return v

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_and_validate_datetime(cls, v: Any) -> datetime | None:
        """Parse various datetime formats and ensure timezone-aware."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                pass
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v

    @field_validator("caption", mode="before")
    @classmethod
    def validate_caption(cls, v: Any) -> str | None:
        """Normalize empty captions to None and strip whitespace."""
        if v == "":
            return None
        if isinstance(v, str):
            return v.strip() or None
        return v

    @model_validator(mode="after")
    def validate_created_at_and_metadata(self) -> Self:
        """Post-validation logic for created_at and warnings."""
        if self.created_at is not None:
            # Timezone awareness
            if self.created_at.tzinfo is None:
                object.__setattr__(self, "created_at", self.created_at.replace(tzinfo=timezone.utc))
                if "Naive datetime converted to UTC" not in self.parse_warnings:
                    self.parse_warnings.append("Naive datetime converted to UTC")
            
            # Old date warning - specific to test expectations
            if self.created_at.year < 1990:
                if "Timestamp before 1990" not in self.parse_warnings:
                    self.parse_warnings.append("Timestamp before 1990")
        
        # Check for near-zero coordinates warning
        if self.location and self.location.coordinates:
            if abs(self.location.coordinates.latitude) < 0.0001 and abs(self.location.coordinates.longitude) < 0.0001:
                if "Coordinates near (0,0)" not in self.parse_warnings:
                    self.parse_warnings.append("Coordinates near (0,0)")
        return self

    @field_validator("people", mode="before")
    @classmethod
    def validate_people(cls, v: Any) -> list[PersonTag]:
        """Convert list of strings to list of PersonTag objects."""
        if isinstance(v, list):
            return [PersonTag(name=p) if isinstance(p, str) else p for p in v]
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Compatibility helper for model validation from dict."""
        return cls.model_validate(data)

    def compute_content_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for exact deduplication."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5_hash.update(chunk)
        hash_value = md5_hash.hexdigest()
        object.__setattr__(self, "content_hash", hash_value)
        return hash_value

    def compute_metadata_hash(self) -> str:
        """Create hash from key metadata for fuzzy deduplication."""
        components = []
        if self.created_at:
            components.append(self.created_at.strftime("%Y%m%d%H%M"))
        components.append(self.source_platform.value)
        components.append(self.media_type.value)
        if self.location and self.location.coordinates:
            approx = self.location.coordinates.to_approximate(precision=2)
            components.append(f"{approx.latitude},{approx.longitude}")
        combined = "|".join(components)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        object.__setattr__(self, "metadata_hash", hash_value)
        return hash_value

    def get_year(self) -> int | None:
        """Get year of memory."""
        return self.created_at.year if self.created_at else None

    def get_month(self) -> int | None:
        """Get month of memory (1-12)."""
        return self.created_at.month if self.created_at else None

    def get_year_month(self) -> str | None:
        """Get year and month as YYYY-MM string."""
        return self.created_at.strftime("%Y-%m") if self.created_at else None

    def days_since(self, reference_date: date) -> int:
        """Calculate days between reference_date and memory date."""
        if not self.created_at:
            return 0
        delta = reference_date - self.created_at.date()
        return delta.days

    # Visual Enrichment (AI-derived from vision model)
    scene_tags: list[str] = Field(default_factory=list)
    """AI-detected scene classifications (indoor, outdoor, beach, etc.)"""
    
    vibe_tags: list[str] = Field(default_factory=list)
    """AI-detected ambience/mood (cozy, energetic, contemplative, etc.)"""
    
    visual_motifs: list[str] = Field(default_factory=list)
    """Detected objects/themes of significance (camera, pet, food, etc.)"""
    
    visual_confidence: float | None = None
    """Confidence score from the vision model (0.0-1.0)"""
    
    visually_analyzed: bool = False
    """Flag indicating whether this item went through visual AI analysis"""

    # Context fields
    location: Location | None = None
    people: list[PersonTag] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    album_name: str | None = None

    # Metadata fields
    original_metadata: dict[str, Any] = Field(default_factory=dict)
    parse_warnings: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def to_ai_payload(self, privacy_level: str = "standard") -> dict:
        """Convert Memory to dict safe for AI consumption, including visual context.
        
        Privacy levels:
        - "strict": Only timestamp (month/year), media_type, source_platform
        - "standard": Above + location (country), people count, has_caption, visual_context
        - "detailed": Above + location (city), caption (truncated), anonymized names
        """
        if privacy_level not in ["strict", "standard", "detailed"]:
            raise ValueError(f"Invalid privacy_level: {privacy_level}")

        payload: dict[str, Any] = {
            "id": self.id,
            "platform": self.source_platform.value,
            "media_type": self.media_type.value,
        }

        if self.created_at:
            if privacy_level == "strict":
                payload["created_at"] = self.created_at.strftime("%Y-%m")
            else:
                payload["created_at"] = self.created_at.strftime("%Y-%m-%d")

        if privacy_level in ["standard", "detailed"]:
            payload["has_location"] = self.location is not None and not self.location.is_empty()
            if self.location and self.location.country:
                payload["location_country"] = self.location.country_code or self.location.country
            payload["people_count"] = len(self.people)
            payload["has_caption"] = self.caption is not None
            
            # Include visual context (Scene/Vibe/Motif) if analyzed
            if self.visually_analyzed:
                visual_context = {}
                if self.scene_tags: visual_context["scenes"] = self.scene_tags
                if self.vibe_tags: visual_context["vibes"] = self.vibe_tags
                if self.visual_motifs: visual_context["motifs"] = self.visual_motifs
                if visual_context:
                    payload["visual_context"] = visual_context

        if privacy_level == "detailed":
            if self.location:
                payload["location"] = self.location.to_ai_summary()
            if self.caption:
                payload["caption"] = self.caption[:100] + ("..." if len(self.caption) > 100 else "")
            if self.people:
                payload["people"] = [p.to_anonymous().name for p in self.people]
            if self.width and self.height:
                payload["dimensions"] = f"{self.width}x{self.height}"

        return payload

    def get_year_month(self) -> str | None:
        """Return YYYY-MM string for grouping."""
        if not self.created_at:
            return None
        return self.created_at.strftime("%Y-%m")

    def days_since(self, reference: date | datetime) -> int:
        """Calculate days between this memory and a reference date."""
        if not self.created_at:
            return 0
        if isinstance(reference, datetime):
            ref_date = reference.date()
        else:
            ref_date = reference
        return (ref_date - self.created_at.date()).days

    def is_same_moment(self, other: "Memory", tolerance_seconds: int = 60) -> bool:
        """Check if two memories are from the same moment (deduplication)."""
        if not self.created_at or not other.created_at:
            return False
        time_diff = abs((self.created_at - other.created_at).total_seconds())
        if time_diff > tolerance_seconds or self.media_type != other.media_type:
            return False
        if (self.location and self.location.coordinates and 
            other.location and other.location.coordinates):
            if self.location.coordinates.distance_km(other.location.coordinates) > 1.0:
                return False
        return True

    def merge_with(self, other: "Memory") -> "Memory":
        """Merge another Memory into this one, preferring higher-confidence data."""
        conf_rank = {
            ConfidenceLevel.VERIFIED: 5, ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3, ConfidenceLevel.LOW: 2,
            ConfidenceLevel.INFERRED: 1,
        }

        def prefer(v1, c1, v2, c2):
            if v1 is None: return v2, c2
            if v2 is None: return v1, c1
            return (v1, c1) if conf_rank[c1] >= conf_rank[c2] else (v2, c2)

        merged_data = self.model_dump()
        
        # Merge key fields
        created_at, conf = prefer(self.created_at, self.created_at_confidence,
                                  other.created_at, other.created_at_confidence)
        merged_data["created_at"] = created_at
        merged_data["created_at_confidence"] = conf

        # Merge location
        loc1 = self.location
        loc2 = other.location
        if loc1 is None:
            merged_data["location"] = loc2
        elif loc2 is None:
            merged_data["location"] = loc1
        else:
            if conf_rank[loc2.confidence] > conf_rank[loc1.confidence]:
                merged_data["location"] = loc2
            else:
                merged_data["location"] = loc1

        # Merge caption
        if not self.caption and other.caption:
            merged_data["caption"] = other.caption

        # People and tags (deduplicated)
        people_dict = {p.normalized_name or p.name: p for p in self.people}
        for p in other.people:
            key = p.normalized_name or p.name
            if key not in people_dict or conf_rank[p.confidence] > conf_rank[people_dict[key].confidence]:
                people_dict[key] = p
        merged_data["people"] = list(people_dict.values())
        merged_data["tags"] = list(set(self.tags + other.tags))
        
        # Merge visual enrichment
        if other.visually_analyzed and not self.visually_analyzed:
            merged_data["scene_tags"] = other.scene_tags
            merged_data["vibe_tags"] = other.vibe_tags
            merged_data["visual_motifs"] = other.visual_motifs
            merged_data["visual_confidence"] = other.visual_confidence
            merged_data["visually_analyzed"] = True

        return Memory(**merged_data)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Factory method handling legacy and nested formats."""
        if "location" in data and isinstance(data["location"], dict):
            loc = data["location"]
            if "coordinates" in loc and isinstance(loc["coordinates"], dict):
                loc["coordinates"] = GeoPoint(**loc["coordinates"])
            data["location"] = Location(**loc)
        if "people" in data and isinstance(data["people"], list):
            data["people"] = [PersonTag(**p) if isinstance(p, dict) else p for p in data["people"]]
        return cls(**data)


# =============================================================================
# AI-Generated Models
# =============================================================================


class LifeChapter(BaseModel):
    """A distinct chapter or phase in someone's life, enriched with visual patterns."""

    title: str
    start_date: date
    end_date: date
    themes: list[str] = Field(default_factory=list)
    narrative: str = ""
    opening_line: str = ""
    key_events: list[str] = Field(default_factory=list)
    location_summary: str | None = None
    memory_count: int = 0
    confidence: float = 0.5
    memory_ids: list[str] = Field(default_factory=list)
    """List of memory IDs belonging to this chapter"""

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: Any) -> float:
        """Handle both numeric and categorical confidence."""
        if isinstance(v, (int, float)):
            if v < 0 or v > 1:
                raise ValueError(f"Confidence score must be between 0 and 1, got {v}")
            return float(v)
        if isinstance(v, str):
            mapping = {
                "verified": 1.0,
                "high": 0.8,
                "medium": 0.5,
                "low": 0.2,
                "inferred": 0.1,
            }
            return mapping.get(v.lower(), 0.5)
        return 0.5

    @property
    def duration_days(self) -> int:
        """Calculate duration of the chapter."""
        return (self.end_date - self.start_date).days

    def merge_with(self, other: "LifeChapter") -> "LifeChapter":
        """Merge another chapter into this one."""
        new_start = min(self.start_date, other.start_date)
        new_end = max(self.end_date, other.end_date)
        new_themes = sorted(list(set(self.themes + other.themes)))
        new_memory_ids = sorted(list(set(self.memory_ids + other.memory_ids)))
        
        return LifeChapter(
            title=self.title, # Keep own title
            start_date=new_start,
            end_date=new_end,
            themes=new_themes,
            narrative=self.narrative + "\n" + other.narrative,
            memory_ids=new_memory_ids,
            memory_count=len(new_memory_ids)
        )

    def overlaps_with(self, other: "LifeChapter") -> bool:
        """Check if this chapter overlaps in time with another."""
        return self.start_date <= other.end_date and self.end_date >= other.start_date

    def to_timeline_entry(self) -> dict[str, Any]:
        """Convert to format suitable for timeline visualization."""
        return {
            "title": self.title,
            "start": self.start_date.isoformat(),
            "end": self.end_date.isoformat(),
            "memory_count": self.memory_count,
            "narrative": self.narrative
        }

    # Visual Summary Fields
    dominant_scenes: list[str] = Field(default_factory=list)
    """Aggregated most common scene tags across chapter"""
    
    dominant_vibes: list[str] = Field(default_factory=list)
    """Aggregated mood/vibe across chapter"""
    
    recurring_motifs: list[str] = Field(default_factory=list)
    """Objects/themes that appear repeatedly"""
    
    representative_images: list[str] = Field(default_factory=list)
    """List of memory IDs selected as representative"""
    
    thumbnail_paths: list[str] = Field(default_factory=list)
    """Local paths for thumbnail display in report (privacy-filtered)"""


class VisualAnalysisStats(BaseModel):
    """Deep metrics about the visual analysis process."""
    total_images_available: int = 0
    images_sampled: int = 0
    images_analyzed: int = 0
    chapters_with_visual_context: int = 0


class LifeStoryReport(BaseModel):
    """Complete life story analysis report with deep metadata."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: uuid_module.UUID = Field(default_factory=uuid_module.uuid4)
    generated_at: datetime
    ai_model: str
    total_memories: int = Field(validation_alias=AliasChoices("total_memories", "total_memories_analyzed", "total_media_analyzed"))
    date_range: tuple[date, date] | None = None
    executive_summary: str
    chapters: list[LifeChapter]

    @property
    def total_memories_analyzed(self) -> int:
        """Alias for total_memories used in some tests."""
        return self.total_memories

    @computed_field
    @property
    def total_media_analyzed(self) -> int:
        """Alias for total_memories used in legacy tests."""
        return self.total_memories

    @computed_field
    @property
    def is_fallback_mode(self) -> bool:
        """Alias for is_fallback used in legacy tests."""
        return self.is_fallback

    @computed_field
    @property
    def ai_model_used(self) -> str:
        """Alias for ai_model used in legacy tests."""
        return self.ai_model

    @computed_field
    @property
    def years_covered(self) -> int:
        """Calculate number of years covered by the report."""
        if self.date_range:
            return self.date_range[1].year - self.date_range[0].year + 1
        
        if not self.chapters:
            return 0
            
        start_years = [c.start_date.year for c in self.chapters if c.start_date]
        end_years = [c.end_date.year for c in self.chapters if c.end_date]
        
        if not start_years or not end_years:
            return 0
            
        return max(end_years) - min(start_years) + 1
    timeline_stats: dict[str, Any] = Field(default_factory=dict)
    data_quality_notes: list[str] = Field(default_factory=list)
    is_fallback: bool = False

    # Analysis Metadata
    analysis_depth: DepthMode | None = None
    """The depth mode used ("quick", "standard", "deep")"""
    
    images_analyzed_count: int | None = None
    """Actual number of images sent to vision AI"""
    
    estimated_vision_tokens: int | None = None
    """Approximate token usage for vision calls"""
    
    estimated_narrative_tokens: int | None = None
    """Approximate token usage for narrative calls"""
    
    total_estimated_cost_usd: float | None = None
    """Rough cost estimate for transparency"""
    
    sampling_reduction_applied: bool = False
    """True if global cap forced per-chapter reductions"""
    
    vision_model_used: str | None = None
    """Model used for visual intelligence (e.g., gemini-1.5-flash)"""
    
    narrative_model_used: str | None = None
    """Model used for narrative generation (e.g., gemini-1.5-pro)"""
    
    visual_stats: VisualAnalysisStats | None = None
    """Aggregated visual analysis statistics"""

    platform_insights: list[PlatformBehaviorInsight] = Field(default_factory=list)
    """Analysis of platform-specific usage patterns"""

    data_gaps: list[DataGap] = Field(default_factory=list)
    """Identified significant gaps in the timeline"""

    detected_patterns: list[str] = Field(default_factory=list)
    """Identified behavioral patterns across the timeline"""

    def to_dict(self) -> dict[str, Any]:
        """Compatibility helper for dict conversion."""
        return self.model_dump()
