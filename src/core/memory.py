"""Core Memory Data Model for Digital Life Narrative AI.

This module defines the universal "Memory" object that represents ANY piece of media
from ANY platform (Snapchat, Google Photos, Facebook, local folders, etc.). This is
the common language between parsers and the AI analyzer.

A Memory is a single moment captured in time — a photo, video, message, check-in, or
story. The Memory model is platform-agnostic; after parsing, you shouldn't be able to
tell what platform a Memory came from except by checking its source field.

The design prioritizes:
- Privacy by design (to_ai_payload controls what goes to Gemini)
- Deduplication support (content_hash, metadata_hash, is_same_moment)
- Timezone correctness (all datetimes must be timezone-aware)
- Confidence tracking (VERIFIED > HIGH > MEDIUM > LOW > INFERRED)
- Merging capabilities (combine memories from different sources)

Example:
    >>> from pathlib import Path
    >>> from datetime import datetime, timezone
    >>> 
    >>> memory = Memory(
    ...     source_platform=SourcePlatform.GOOGLE_PHOTOS,
    ...     media_type=MediaType.PHOTO,
    ...     created_at=datetime.now(timezone.utc),
    ...     caption="Sunset at the beach"
    ... )
    >>> memory.compute_content_hash(Path("photo.jpg"))
    >>> ai_payload = memory.to_ai_payload(privacy_level="standard")
"""

import hashlib
import uuid as uuid_module
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class MediaType(str, Enum):
    """Types of media that can be processed.
    
    Different media types carry different behavioral signals for AI analysis.
    Screenshots, for example, often indicate problem-solving or information capture,
    while live photos suggest iPhone users capturing candid moments.
    
    Attributes:
        PHOTO: Static image files (jpg, png, heic, etc.)
        VIDEO: Video files (mp4, mov, etc.)
        AUDIO: Audio recordings or voice messages
        STORY: Ephemeral story content (Snapchat stories, Instagram stories)
        MESSAGE: Chat messages or DMs
        CHECK_IN: Location check-ins or place tags
        SCREENSHOT: Screenshots (distinct behavioral signal from photos)
        LIVE_PHOTO: iOS live photos (photo + video pair)
        DOCUMENT: Documents, PDFs, text files
        UNKNOWN: Unrecognized or unsupported media type
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
    """Supported source platforms for data exports.
    
    LOCAL represents photos not from any cloud export (DSLR, manual phone transfers).
    UNKNOWN is a fallback, but parsers should try to avoid it.
    
    Attributes:
        SNAPCHAT: Snapchat data export
        GOOGLE_PHOTOS: Google Photos/Takeout export
        FACEBOOK: Facebook data download
        INSTAGRAM: Instagram data download
        WHATSAPP: WhatsApp chat export
        IMESSAGE: iMessage export (macOS)
        ONEDRIVE: Microsoft OneDrive export
        DROPBOX: Dropbox export
        LOCAL: Local filesystem (camera roll, downloads, DSLR)
        UNKNOWN: Unrecognized platform
    """
    
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
    """Confidence level for AI-inferred or extracted data.
    
    Confidence cascade: When merging memories, higher confidence wins.
    VERIFIED > HIGH > MEDIUM > LOW > INFERRED
    
    Attributes:
        VERIFIED: From explicit platform metadata (e.g., GPS coordinates in EXIF)
        HIGH: From reliable source (e.g., EXIF with GPS)
        MEDIUM: From parsing (e.g., filename patterns like IMG_20200715_123456.jpg)
        LOW: From heuristics (e.g., folder names like "Summer 2020")
        INFERRED: AI-generated or guessed
    """
    
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFERRED = "inferred"


# =============================================================================
# Supporting Models
# =============================================================================


class GeoPoint(BaseModel):
    """Geographic coordinates with optional altitude and accuracy.
    
    Represents a specific point on Earth. Coordinates must be within valid bounds:
    latitude between -90 and 90, longitude between -180 and 180.
    
    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        altitude_meters: Optional altitude above sea level in meters
        accuracy_meters: Optional accuracy radius in meters
    
    Example:
        >>> central_park = GeoPoint(latitude=40.7829, longitude=-73.9654)
        >>> distance = central_park.distance_km(times_square)
        >>> approximate = central_park.to_approximate(precision=2)
    """
    
    latitude: float
    longitude: float
    altitude_meters: float | None = None
    accuracy_meters: float | None = None
    
    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within valid range (-90 to 90)."""
        if v < -90 or v > 90:
            raise ValueError(f"Latitude must be between -90 and 90 degrees, got {v}")
        return v
    
    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within valid range (-180 to 180)."""
        if v < -180 or v > 180:
            raise ValueError(f"Longitude must be between -180 and 180 degrees, got {v}")
        return v
    
    def distance_km(self, other: "GeoPoint") -> float:
        """Calculate distance to another point using Haversine formula.
        
        Args:
            other: Another GeoPoint to calculate distance to
            
        Returns:
            Distance in kilometers
            
        Example:
            >>> point1 = GeoPoint(latitude=40.7829, longitude=-73.9654)
            >>> point2 = GeoPoint(latitude=34.0522, longitude=-118.2437)
            >>> distance = point1.distance_km(point2)  # NYC to LA
        """
        import math
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def to_approximate(self, precision: int = 2) -> "GeoPoint":
        """Round coordinates for privacy protection.
        
        Reduces coordinate precision to avoid exact location tracking while
        preserving general area information.
        
        Args:
            precision: Number of decimal places to keep (default: 2)
                      2 = ~1.1km accuracy, 3 = ~110m, 4 = ~11m
            
        Returns:
            New GeoPoint with rounded coordinates
            
        Example:
            >>> exact = GeoPoint(latitude=40.7829123, longitude=-73.9654456)
            >>> approx = exact.to_approximate(precision=2)
            >>> # approx.latitude = 40.78, approx.longitude = -73.97
        """
        return GeoPoint(
            latitude=round(self.latitude, precision),
            longitude=round(self.longitude, precision),
            altitude_meters=None,  # Remove altitude for privacy
            accuracy_meters=None,  # Remove accuracy for privacy
        )


class Location(BaseModel):
    """Complete location information combining coordinates and place names.
    
    At minimum, either coordinates OR place_name should be set for a Location
    to be meaningful. Supports hierarchical place information (place → locality
    → region → country).
    
    Attributes:
        coordinates: Optional GPS coordinates
        place_name: Place name (e.g., "Central Park")
        locality: Locality/city (e.g., "Manhattan")
        region: Region/state (e.g., "New York")
        country: Country name (e.g., "United States")
        country_code: ISO country code (e.g., "US")
        raw_string: Original location string before parsing
        confidence: Confidence level of this location data
        
    Example:
        >>> location = Location(
        ...     coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
        ...     place_name="Central Park",
        ...     locality="Manhattan",
        ...     region="New York",
        ...     country="United States",
        ...     country_code="US",
        ...     confidence=ConfidenceLevel.VERIFIED
        ... )
        >>> print(location.to_display_string())
        >>> # "Central Park, Manhattan, New York"
    """
    
    coordinates: GeoPoint | None = None
    place_name: str | None = None
    locality: str | None = None
    region: str | None = None
    country: str | None = None
    country_code: str | None = None
    raw_string: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    def is_empty(self) -> bool:
        """Check if all location fields are None.
        
        Returns:
            True if no location information is available
        """
        return all(
            [
                self.coordinates is None,
                self.place_name is None,
                self.locality is None,
                self.region is None,
                self.country is None,
                self.country_code is None,
            ]
        )
    
    def to_display_string(self) -> str:
        """Generate human-readable location string.
        
        Prioritizes place names over coordinates, builds hierarchical display.
        
        Returns:
            Human-readable location string
            
        Example:
            >>> location.to_display_string()
            >>> # "Central Park, Manhattan, New York"
        """
        if self.is_empty():
            return "Unknown location"
        
        # Build hierarchical display from most specific to least
        parts = []
        
        if self.place_name:
            parts.append(self.place_name)
        
        if self.locality and self.locality != self.place_name:
            parts.append(self.locality)
        
        if self.region and self.region not in parts:
            parts.append(self.region)
        
        if parts:
            return ", ".join(parts)
        
        # Fall back to country only
        if self.country:
            return self.country
        
        # Fall back to coordinates
        if self.coordinates:
            return f"{self.coordinates.latitude:.4f}, {self.coordinates.longitude:.4f}"
        
        # Fall back to raw string
        if self.raw_string:
            return self.raw_string
        
        return "Unknown location"
    
    def to_ai_summary(self) -> str:
        """Generate condensed location for AI consumption.
        
        Provides minimal, privacy-conscious location information for AI analysis.
        
        Returns:
            Condensed location string (e.g., "New York, US")
            
        Example:
            >>> location.to_ai_summary()
            >>> # "New York, US" instead of "Central Park, Manhattan, New York"
        """
        if self.is_empty():
            return "Unknown"
        
        parts = []
        
        # Prefer region (state/province) over locality for privacy
        if self.region:
            parts.append(self.region)
        elif self.locality:
            parts.append(self.locality)
        elif self.place_name:
            parts.append(self.place_name)
        
        # Add country code if available
        if self.country_code:
            parts.append(self.country_code)
        elif self.country:
            parts.append(self.country)
        
        if parts:
            return ", ".join(parts)
        
        return "Unknown"


class PersonTag(BaseModel):
    """Person tag/mention in media.
    
    Names might be usernames, display names, or real names depending on platform.
    Supports normalization and anonymization for privacy.
    
    Attributes:
        name: The tag/name as it appears in source
        normalized_name: Cleaned version (lowercase, trimmed)
        platform_id: Platform's internal ID if available (e.g., @username, user_id)
        confidence: Confidence level of the tag
        
    Example:
        >>> person = PersonTag(
        ...     name="Alice Smith",
        ...     normalized_name="alice smith",
        ...     platform_id="alice_s_123",
        ...     confidence=ConfidenceLevel.VERIFIED
        ... )
        >>> anonymous = person.to_anonymous()
    """
    
    name: str
    normalized_name: str | None = None
    platform_id: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    @field_validator("normalized_name", mode="before")
    @classmethod
    def auto_normalize(cls, v: str | None, info) -> str | None:
        """Auto-generate normalized_name from name if not provided."""
        if v is None and "name" in info.data:
            # Lowercase and strip whitespace
            return info.data["name"].lower().strip()
        return v.lower().strip() if v else None
    
    def to_anonymous(self) -> "PersonTag":
        """Return copy with name replaced by hash for privacy.
        
        Returns:
            New PersonTag with anonymized name
            
        Example:
            >>> person = PersonTag(name="Alice Smith")
            >>> anon = person.to_anonymous()
            >>> # anon.name = "person_a1b2c3d4"
        """
        # Create consistent hash for this name
        name_hash = hashlib.sha256(self.name.encode()).hexdigest()[:8]
        
        return PersonTag(
            name=f"person_{name_hash}",
            normalized_name=f"person_{name_hash}",
            platform_id=None,  # Remove platform ID for privacy
            confidence=self.confidence,
        )


# =============================================================================
# Main Memory Model
# =============================================================================


class Memory(BaseModel):
    """Universal memory object representing a single moment in time.
    
    This is the bridge between parsers and AI. All parsed media normalizes into
    this schema. The Memory model is platform-agnostic and privacy-conscious.
    
    Identity Fields:
        id: Unique identifier (UUID, auto-generated)
        content_hash: Hash of file content for exact deduplication
        metadata_hash: Hash of key metadata for fuzzy deduplication
    
    Source Fields:
        source_platform: Platform this memory came from
        source_path: Original file path (internal use only, never sent to AI)
        source_filename: Just the filename
        source_export_path: Relative path within export
    
    Temporal Fields:
        created_at: When the memory was created/captured (timezone-aware)
        created_at_confidence: Confidence level of the timestamp
        timezone_name: Timezone name (e.g., "America/New_York")
        modified_at: If different from created_at
    
    Content Fields:
        media_type: Type of media (photo, video, etc.)
        caption: Any text associated (post text, message, etc.)
        duration_seconds: For video/audio
        width: Image/video width in pixels
        height: Image/video height in pixels
    
    Context Fields:
        location: Geographic location information
        people: List of person tags
        tags: Any tags/labels from source
        album_name: If from an album
        thread_name: If from a message thread
    
    Metadata Fields:
        camera_make: Camera manufacturer (from EXIF)
        camera_model: Camera model (from EXIF)
        original_metadata: Preserve everything from source
    
    Processing Fields:
        is_duplicate: Flag for duplicate detection
        duplicate_of_id: Reference to original if duplicate
        parse_warnings: Warnings encountered during parsing
        
    Example:
        >>> from datetime import datetime, timezone
        >>> memory = Memory(
        ...     source_platform=SourcePlatform.GOOGLE_PHOTOS,
        ...     media_type=MediaType.PHOTO,
        ...     created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
        ...     location=Location(
        ...         place_name="Central Park",
        ...         locality="Manhattan",
        ...         region="New York"
        ...     ),
        ...     caption="Summer afternoon"
        ... )
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
    
    # Temporal fields (must be timezone-aware)
    created_at: datetime | None = None
    created_at_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    timezone_name: str | None = None
    modified_at: datetime | None = None
    
    # Content fields
    media_type: MediaType = MediaType.UNKNOWN
    caption: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    
    # Context fields
    location: Location | None = None
    people: list[PersonTag] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    album_name: str | None = None
    thread_name: str | None = None
    
    # Metadata fields
    camera_make: str | None = None
    camera_model: str | None = None
    original_metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Processing fields
    is_duplicate: bool = False
    duplicate_of_id: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    
    model_config = {"arbitrary_types_allowed": True}
    
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
        """Parse various datetime formats and ensure timezone-aware.
        
        Handles:
        - datetime objects (naive or aware)
        - ISO format strings
        - Unix timestamps (int/float)
        - None (valid for memories without timestamps)
        """
        if v is None:
            return None
        
        # Already a datetime
        if isinstance(v, datetime):
            return v
        
        # Parse ISO string
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v)
                return dt
            except ValueError:
                # Try other formats if needed
                pass
        
        # Parse Unix timestamp
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        
        return v
    
    @model_validator(mode="after")
    def ensure_timezone_aware(self) -> Self:
        """Ensure created_at is timezone-aware, assume UTC if naive."""
        if self.created_at is not None and self.created_at.tzinfo is None:
            # Naive datetime: assume UTC and warn
            object.__setattr__(
                self, "created_at", self.created_at.replace(tzinfo=timezone.utc)
            )
            if "Naive datetime converted to UTC" not in self.parse_warnings:
                self.parse_warnings.append("Naive datetime converted to UTC")
        
        # Warn for timestamps before 1990 (likely parse error)
        if self.created_at is not None and self.created_at.year < 1990:
            if "Timestamp before 1990, may be parse error" not in self.parse_warnings:
                self.parse_warnings.append("Timestamp before 1990, may be parse error")
        
        # Flag coordinates at (0, 0) as suspicious
        if (
            self.location
            and self.location.coordinates
            and abs(self.location.coordinates.latitude) < 0.1
            and abs(self.location.coordinates.longitude) < 0.1
        ):
            if "Coordinates near (0,0) - likely error" not in self.parse_warnings:
                self.parse_warnings.append("Coordinates near (0,0) - likely error")
        
        return self
    
    @field_validator("caption", mode="before")
    @classmethod
    def normalize_caption(cls, v: str | None) -> str | None:
        """Strip whitespace and convert empty strings to None."""
        if v is None:
            return None
        
        stripped = v.strip()
        return stripped if stripped else None
    
    def compute_content_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for exact deduplication.
        
        Reads file in 64KB chunks to handle large files efficiently.
        Stores result in self.content_hash.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            The computed hash string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            
        Example:
            >>> memory = Memory(source_platform=SourcePlatform.LOCAL)
            >>> hash_value = memory.compute_content_hash(Path("photo.jpg"))
        """
        md5_hash = hashlib.md5()
        
        with open(file_path, "rb") as f:
            # Read in 64KB chunks
            for chunk in iter(lambda: f.read(65536), b""):
                md5_hash.update(chunk)
        
        hash_value = md5_hash.hexdigest()
        object.__setattr__(self, "content_hash", hash_value)
        
        return hash_value
    
    def compute_metadata_hash(self) -> str:
        """Create hash from key metadata for fuzzy deduplication.
        
        Hashes: created_at (to minute), source_platform, media_type, approximate location.
        This allows fuzzy deduplication of same moment from different sources
        (e.g., same moment captured on phone and DSLR).
        
        Returns:
            The computed metadata hash
            
        Example:
            >>> memory.compute_metadata_hash()
            >>> # Can detect: same photo auto-backed up to multiple platforms
        """
        components = []
        
        # Timestamp to minute precision
        if self.created_at:
            timestamp_str = self.created_at.strftime("%Y%m%d%H%M")
            components.append(timestamp_str)
        
        # Platform and media type
        components.append(self.source_platform.value)
        components.append(self.media_type.value)
        
        # Approximate location (rounded to 2 decimal places)
        if self.location and self.location.coordinates:
            approx_coords = self.location.coordinates.to_approximate(precision=2)
            components.append(f"{approx_coords.latitude},{approx_coords.longitude}")
        
        # Combine and hash
        combined = "|".join(components)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        object.__setattr__(self, "metadata_hash", hash_value)
        
        return hash_value
    
    def to_ai_payload(self, privacy_level: str = "standard") -> dict:
        """Convert Memory to dict safe for sending to Gemini.
        
        **CRITICAL METHOD** — This is the ONLY way memories should be prepared
        for AI consumption. Ensures sensitive data never leaves the system.
        
        Privacy levels:
        - "strict": Only timestamp (month/year), media_type, source_platform
        - "standard": Above + location (country only), people count, has_caption bool
        - "detailed": Above + location (city), caption (truncated 100 chars), 
                     anonymized people names
        
        NEVER includes: source_path, full coordinates, full captions,
                       original_metadata, content_hash
        
        Args:
            privacy_level: One of "strict", "standard", "detailed"
            
        Returns:
            Dictionary safe for AI consumption
            
        Raises:
            ValueError: If privacy_level is invalid
            
        Example:
            >>> memory.to_ai_payload(privacy_level="standard")
            >>> # {"id": "...", "created_at": "2020-07", "platform": "google_photos", ...}
        """
        if privacy_level not in ["strict", "standard", "detailed"]:
            raise ValueError(
                f"Invalid privacy_level: {privacy_level}. "
                f"Must be 'strict', 'standard', or 'detailed'"
            )
        
        payload: dict[str, Any] = {
            "id": self.id,
            "platform": self.source_platform.value,
            "media_type": self.media_type.value,
        }
        
        # === STRICT: Minimal data ===
        if self.created_at:
            if privacy_level == "strict":
                # Month and year only
                payload["created_at"] = self.created_at.strftime("%Y-%m")
            else:
                # Full date (but not time)
                payload["created_at"] = self.created_at.strftime("%Y-%m-%d")
        
        # === STANDARD: Add aggregates ===
        if privacy_level in ["standard", "detailed"]:
            # Location: country only
            if self.location and not self.location.is_empty():
                payload["location_country"] = (
                    self.location.country_code or self.location.country
                )
                payload["has_location"] = True
            else:
                payload["has_location"] = False
            
            # People: count only
            payload["people_count"] = len(self.people)
            
            # Caption: just whether it exists
            payload["has_caption"] = self.caption is not None
        
        # === DETAILED: Add more context ===
        if privacy_level == "detailed":
            # Location: city/region
            if self.location and not self.location.is_empty():
                payload["location"] = self.location.to_ai_summary()
            
            # Caption: truncated
            if self.caption:
                max_length = 100
                truncated = self.caption[:max_length]
                if len(self.caption) > max_length:
                    truncated += "..."
                payload["caption"] = truncated
            
            # People: anonymized names
            if self.people:
                payload["people"] = [p.to_anonymous().name for p in self.people]
            
            # Media dimensions
            if self.width and self.height:
                payload["dimensions"] = f"{self.width}x{self.height}"
            
            # Duration for video/audio
            if self.duration_seconds:
                payload["duration_seconds"] = round(self.duration_seconds, 1)
        
        return payload
    
    def to_timeline_point(self) -> dict:
        """Return minimal dict for timeline aggregation.
        
        Returns:
            Lightweight dict for timeline visualization
            
        Example:
            >>> memory.to_timeline_point()
            >>> # {"id": "...", "created_at": "2020-07-15T12:30:00Z", ...}
        """
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_platform": self.source_platform.value,
            "media_type": self.media_type.value,
            "has_location": (
                self.location is not None and not self.location.is_empty()
            ),
            "has_people": len(self.people) > 0,
        }
    
    def get_year(self) -> int | None:
        """Extract year from created_at.
        
        Returns:
            Year as integer, or None if no timestamp
        """
        return self.created_at.year if self.created_at else None
    
    def get_month(self) -> int | None:
        """Extract month from created_at.
        
        Returns:
            Month as integer (1-12), or None if no timestamp
        """
        return self.created_at.month if self.created_at else None
    
    def get_year_month(self) -> str | None:
        """Return YYYY-MM string from created_at.
        
        Returns:
            Year-month string, or None if no timestamp
            
        Example:
            >>> memory.get_year_month()
            >>> # "2020-07"
        """
        return self.created_at.strftime("%Y-%m") if self.created_at else None
    
    def is_same_moment(self, other: "Memory", tolerance_seconds: int = 60) -> bool:
        """Check if two memories are from the same moment.
        
        Used for cross-platform deduplication. For example, a Snapchat save
        and Google Photos backup of the same photo taken within 60 seconds.
        
        Args:
            other: Another Memory to compare
            tolerance_seconds: Time window to consider "same moment" (default: 60)
            
        Returns:
            True if memories are from the same moment
            
        Example:
            >>> snapchat_memory = Memory(created_at=datetime(...))
            >>> google_memory = Memory(created_at=datetime(...))
            >>> snapchat_memory.is_same_moment(google_memory, tolerance_seconds=60)
        """
        # Both must have timestamps
        if not self.created_at or not other.created_at:
            return False
        
        # Calculate time difference
        time_diff = abs((self.created_at - other.created_at).total_seconds())
        
        # Within tolerance?
        if time_diff > tolerance_seconds:
            return False
        
        # Same media type?
        if self.media_type != other.media_type:
            return False
        
        # If both have coordinates, check proximity (within ~1km)
        if (
            self.location
            and self.location.coordinates
            and other.location
            and other.location.coordinates
        ):
            distance = self.location.coordinates.distance_km(
                other.location.coordinates
            )
            if distance > 1.0:  # More than 1km apart
                return False
        
        return True
    
    def merge_with(self, other: "Memory") -> "Memory":
        """Merge another Memory into this one, preferring higher-confidence data.
        
        Business rules:
        - Prefer VERIFIED > HIGH > MEDIUM > LOW > INFERRED
        - Prefer non-None over None
        - Keep both people lists merged and deduplicated
        - Preserve both original_metadata dicts
        
        Args:
            other: Another Memory to merge with
            
        Returns:
            New Memory with merged data
            
        Example:
            >>> phone_memory = Memory(...)  # Has timestamp, no location
            >>> dslr_memory = Memory(...)   # Has location, different timestamp
            >>> merged = phone_memory.merge_with(dslr_memory)
        """
        # Confidence ranking
        confidence_rank = {
            ConfidenceLevel.VERIFIED: 5,
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.INFERRED: 1,
        }
        
        def prefer(
            val1: Any, conf1: ConfidenceLevel, val2: Any, conf2: ConfidenceLevel
        ) -> tuple[Any, ConfidenceLevel]:
            """Return value with higher confidence, or non-None value."""
            if val1 is None:
                return val2, conf2
            if val2 is None:
                return val1, conf1
            
            # Both non-None: prefer higher confidence
            if confidence_rank[conf1] >= confidence_rank[conf2]:
                return val1, conf1
            else:
                return val2, conf2
        
        # Start with a copy of self
        merged_data = self.model_dump()
        
        # Merge timestamp
        timestamp, conf = prefer(
            self.created_at,
            self.created_at_confidence,
            other.created_at,
            other.created_at_confidence,
        )
        merged_data["created_at"] = timestamp
        merged_data["created_at_confidence"] = conf
        
        # Merge location (prefer entire location with higher confidence)
        if self.location and other.location:
            loc_self_conf = self.location.confidence
            loc_other_conf = other.location.confidence
            merged_data["location"] = (
                self.location
                if confidence_rank[loc_self_conf] >= confidence_rank[loc_other_conf]
                else other.location
            )
        elif other.location:
            merged_data["location"] = other.location
        
        # Merge people (deduplicate by normalized_name)
        people_dict = {}
        for person in self.people + other.people:
            key = person.normalized_name or person.name
            if (
                key not in people_dict
                or confidence_rank[person.confidence]
                > confidence_rank[people_dict[key].confidence]
            ):
                people_dict[key] = person
        merged_data["people"] = list(people_dict.values())
        
        # Merge tags (deduplicate)
        merged_data["tags"] = list(set(self.tags + other.tags))
        
        # Merge original_metadata (keep both)
        merged_metadata = {**self.original_metadata, **other.original_metadata}
        merged_data["original_metadata"] = merged_metadata
        
        # Merge parse_warnings
        merged_data["parse_warnings"] = list(
            set(self.parse_warnings + other.parse_warnings)
        )
        
        # Prefer non-None for simple fields
        for field in ["caption", "camera_make", "camera_model", "album_name"]:
            if merged_data.get(field) is None and getattr(other, field) is not None:
                merged_data[field] = getattr(other, field)
        
        return Memory(**merged_data)
    
    def days_since(self, reference: date) -> int | None:
        """Calculate days between this memory and a reference date.
        
        Args:
            reference: Reference date to calculate from
            
        Returns:
            Number of days (can be negative if memory is after reference),
            or None if no timestamp
            
        Example:
            >>> from datetime import date
            >>> memory.days_since(date(2020, 1, 1))
            >>> # 196  (if memory is from 2020-07-15)
        """
        if not self.created_at:
            return None
        
        memory_date = self.created_at.date()
        return (reference - memory_date).days
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Factory method that handles various input formats gracefully.
        
        Handles:
        - Missing optional fields
        - String timestamps that need parsing
        - Nested objects (Location, PersonTag)
        - Legacy field names
        
        Args:
            data: Dictionary of memory data
            
        Returns:
            New Memory instance
            
        Example:
            >>> data = {"source_platform": "google_photos", "media_type": "photo"}
            >>> memory = Memory.from_dict(data)
        """
        # Handle nested Location
        if "location" in data and isinstance(data["location"], dict):
            # Handle nested GeoPoint within Location
            if "coordinates" in data["location"] and isinstance(
                data["location"]["coordinates"], dict
            ):
                data["location"]["coordinates"] = GeoPoint(
                    **data["location"]["coordinates"]
                )
            data["location"] = Location(**data["location"])
        
        # Handle nested PersonTag list
        if "people" in data and isinstance(data["people"], list):
            people = []
            for person_data in data["people"]:
                if isinstance(person_data, dict):
                    people.append(PersonTag(**person_data))
                elif isinstance(person_data, str):
                    # Legacy: just a name string
                    people.append(PersonTag(name=person_data))
                else:
                    people.append(person_data)
            data["people"] = people
        
        return cls(**data)
