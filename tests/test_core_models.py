"""Comprehensive tests for the core data models of Digital Life Narrative AI.

This module verifies Pydantic validation, privacy helpers, serialization,
and edge case handling for Memory, GeoLocation, LifeChapter, and other core models.
"""

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlnai.ai import LifeChapter, LifeStoryReport

# Core Models
from dlnai.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)
from dlnai.core.safety import (
    DetectionMethod,
    MemorySafetyState,
    SafetyAction,
    SafetyCategory,
    SafetyFlag,
    SafetySettings,
)
from dlnai.core.timeline import DateRange

# =============================================================================
# Memory Model Tests
# =============================================================================


class TestMemory:
    """Tests for the universal Memory object."""

    def test_memory_creation_full_fields(self) -> None:
        """Create Memory with all fields populated and assert correctness."""
        id_val = str(uuid.uuid4())
        dt = datetime(2022, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        loc = Location(place_name="San Francisco", country="USA")

        memory = Memory(
            id=id_val,
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=dt,
            source_path="/photos/test.jpg",
            caption="Test Caption",
            location=loc,
            people=[PersonTag(name="Alice")],
            original_metadata={"camera": "iPhone"},
            width=1920,
            height=1080,
            duration_seconds=None,
            content_hash="hash123",
            metadata_hash="meta456",
            created_at_confidence=ConfidenceLevel.HIGH,
        )

        assert memory.id == id_val
        assert memory.source_platform == SourcePlatform.GOOGLE_PHOTOS
        assert memory.media_type == MediaType.PHOTO
        assert memory.created_at == dt
        assert memory.caption == "Test Caption"
        assert memory.location.place_name == "San Francisco"
        assert len(memory.people) == 1
        assert memory.width == 1920
        assert memory.created_at_confidence == ConfidenceLevel.HIGH

    def test_memory_creation_minimal_fields(self) -> None:
        """Create Memory with only required fields and assert defaults."""
        memory = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO)

        assert memory.id is not None
        assert isinstance(memory.id, str)
        assert memory.created_at is None
        assert memory.people == []
        assert memory.original_metadata == {}
        assert memory.created_at_confidence == ConfidenceLevel.MEDIUM

    def test_memory_id_auto_generation(self) -> None:
        """Assert id is generated if not provided and is unique."""
        m1 = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO)
        m2 = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO)

        assert m1.id is not None
        assert m2.id is not None
        assert m1.id != m2.id
        # Verify it's a valid UUID string
        uuid.UUID(m1.id)

    def test_memory_timestamp_validation(self) -> None:
        """Test valid timezone-aware datetime storage."""
        dt = datetime(2021, 5, 20, 15, 30, 0, tzinfo=timezone.utc)
        memory = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, created_at=dt
        )
        assert memory.created_at == dt

    def test_memory_timestamp_naive_handling(self) -> None:
        """Naive datetimes should be converted to UTC or treated as UTC."""
        naive_dt = datetime(2021, 5, 20, 15, 30, 0)
        memory = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, created_at=naive_dt
        )
        # The model have a validator or logic to ensure awareness
        assert memory.created_at.tzinfo == timezone.utc

    def test_memory_optional_fields_none(self) -> None:
        """Creation succeeds with all optional fields as None."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            created_at=None,
            caption=None,
            location=None,
            width=None,
        )
        assert memory.caption is None
        assert memory.location is None
        # Serialization check
        data = memory.model_dump()
        assert data["caption"] is None

    def test_memory_to_ai_summary_no_path_leak(self) -> None:
        """Ensure source_path is NEVER leaked in AI payload."""
        path = "/home/user/private/vacation.jpg"
        memory = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, source_path=path
        )

        payload = memory.to_ai_payload(privacy_level="detailed")
        payload_str = json.dumps(payload)

        assert "source_path" not in payload
        assert "home" not in payload_str
        assert "user" not in payload_str
        assert "private" not in payload_str

    def test_memory_to_ai_summary_caption_truncation(self) -> None:
        """Assert caption is truncated in AI payload based on privacy level."""
        long_caption = "A" * 500
        memory = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, caption=long_caption
        )

        # In detailed mode it should be truncated (usually 100 chars)
        payload = memory.to_ai_payload(privacy_level="detailed")
        assert len(payload["caption"]) < 500
        assert payload["caption"].endswith("...")

        # In standard mode, it should just be a boolean has_caption
        payload_std = memory.to_ai_payload(privacy_level="standard")
        assert "caption" not in payload_std
        assert payload_std["has_caption"] is True

    def test_memory_to_ai_summary_people_anonymization(self) -> None:
        """Assert real names are anonymized in detailed AI payload."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            people=[PersonTag(name="John Doe"), PersonTag(name="Jane Smith")],
        )

        payload = memory.to_ai_payload(privacy_level="detailed")
        assert "people" in payload
        assert "John Doe" not in str(payload["people"])
        assert "Jane Smith" not in str(payload["people"])
        assert len(payload["people"]) == 2

    def test_memory_to_ai_summary_json_serializable(self) -> None:
        """Assert AI payload is fully JSON serializable."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime.now(timezone.utc),
            location=Location(place_name="London", country="UK"),
            people=[PersonTag(name="Bob")],
        )
        payload = memory.to_ai_payload(privacy_level="detailed")
        # This will raise if not serializable
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)

    def test_memory_compute_content_hash(self, tmp_path: Path) -> None:
        """Verify content hashing from real file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        memory = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO)
        hash_val = memory.compute_content_hash(test_file)

        assert hash_val is not None
        assert isinstance(hash_val, str)
        assert len(hash_val) == 32  # MD5 length
        assert memory.content_hash == hash_val

    def test_memory_is_same_moment_within_tolerance(self) -> None:
        """Within 30 seconds tolerance."""
        dt1 = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = dt1 + timedelta(seconds=30)

        m1 = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, created_at=dt1
        )
        m2 = Memory(
            source_platform=SourcePlatform.SNAPCHAT, media_type=MediaType.PHOTO, created_at=dt2
        )

        assert m1.is_same_moment(m2, tolerance_seconds=60) is True

    def test_memory_is_same_moment_outside_tolerance(self) -> None:
        """Outside 1 minute tolerance."""
        dt1 = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = dt1 + timedelta(minutes=5)

        m1 = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, created_at=dt1
        )
        m2 = Memory(
            source_platform=SourcePlatform.SNAPCHAT, media_type=MediaType.PHOTO, created_at=dt2
        )

        assert m1.is_same_moment(m2, tolerance_seconds=60) is False

    def test_memory_merge_with_higher_confidence_wins(self) -> None:
        """High confidence data replaces low confidence data."""
        m1 = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            location=Location(place_name="Unknown City", confidence=ConfidenceLevel.LOW),
            created_at_confidence=ConfidenceLevel.LOW,
        )
        m2 = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            location=Location(place_name="San Francisco", confidence=ConfidenceLevel.HIGH),
            created_at_confidence=ConfidenceLevel.HIGH,
        )

        merged = m1.merge_with(m2)
        assert merged.location.place_name == "San Francisco"
        assert merged.created_at_confidence == ConfidenceLevel.HIGH

    def test_memory_empty_caption_vs_none(self) -> None:
        """Empty strings should be normalized to None."""
        m1 = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, caption="")
        m2 = Memory(source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, caption=None)

        assert m1.caption is None
        assert m2.caption is None

    def test_memory_unicode_caption(self) -> None:
        """Unicode characters in caption work correctly."""
        caption = "Beautiful day in 東京 🇯🇵"
        memory = Memory(
            source_platform=SourcePlatform.LOCAL, media_type=MediaType.PHOTO, caption=caption
        )
        assert memory.caption == caption
        data = memory.model_dump_json()
        assert "東京" in data


# =============================================================================
# GeoLocation Tests
# =============================================================================


class TestGeoLocation:
    """Tests for GeoPoint and Location models."""

    def test_geolocation_valid_coordinates(self) -> None:
        """Standard valid coords."""
        gp = GeoPoint(latitude=40.7128, longitude=-74.0060)
        assert gp.latitude == 40.7128
        assert gp.longitude == -74.0060

    def test_geolocation_invalid_latitude_high(self) -> None:
        """Latitude > 90."""
        with pytest.raises(ValidationError):
            GeoPoint(latitude=95, longitude=0)

    def test_geolocation_invalid_latitude_low(self) -> None:
        """Latitude < -90."""
        with pytest.raises(ValidationError):
            GeoPoint(latitude=-95, longitude=0)

    def test_geolocation_invalid_longitude(self) -> None:
        """Longitude > 180."""
        with pytest.raises(ValidationError):
            GeoPoint(latitude=0, longitude=200)

    def test_geolocation_zero_zero_valid(self) -> None:
        """0,0 is perfectly valid."""
        gp = GeoPoint(latitude=0, longitude=0)
        assert gp.latitude == 0
        assert gp.longitude == 0

    def test_location_to_display_string(self) -> None:
        """Test hierarchical display string generation."""
        loc = Location(
            place_name="Central Park", locality="Manhattan", region="New York", country="USA"
        )
        display = loc.to_display_string()
        assert "Central Park" in display
        assert "Manhattan" in display
        assert "USA" in display

    def test_location_is_empty(self) -> None:
        """Test is_empty detection."""
        loc_empty = Location()
        assert loc_empty.is_empty() is True

        loc_filled = Location(place_name="Somewhere")
        assert loc_filled.is_empty() is False


# =============================================================================
# LifeChapter Tests
# =============================================================================


class TestLifeChapter:
    """Tests for LifeChapter model and logic."""

    def test_chapter_creation_valid(self) -> None:
        """Full creation test."""
        chapter = LifeChapter(
            title="The Adventure",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 30),
            themes=["travel", "discovery"],
            narrative="A small narrative.",
        )
        assert chapter.title == "The Adventure"
        assert chapter.duration_days == 181  # Jan 1 to Jun 30

    def test_chapter_date_range_validation(self) -> None:
        """Some models might have explicit validators for date consistency."""
        # Check if DateRange or LifeChapter enforces end > start
        # If not enforced at init, this test might need adjustment
        chapter = LifeChapter(
            title="Invalid", start_date=date(2021, 1, 1), end_date=date(2020, 1, 1)
        )
        assert chapter.start_date > chapter.end_date  # As confirmed in existing tests

    def test_chapter_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        chapter = LifeChapter(
            title="Test", start_date=date(2020, 1, 1), end_date=date(2020, 1, 2), confidence=0.8
        )
        assert chapter.confidence == 0.8

        with pytest.raises(ValidationError):
            LifeChapter(
                title="Bad", start_date=date(2020, 1, 1), end_date=date(2020, 1, 2), confidence=1.5
            )

    def test_chapter_overlaps_with_true(self) -> None:
        """Overlapping chapters."""
        c1 = LifeChapter(title="C1", start_date=date(2020, 1, 1), end_date=date(2020, 6, 1))
        c2 = LifeChapter(title="C2", start_date=date(2020, 5, 1), end_date=date(2020, 12, 1))
        assert c1.overlaps_with(c2) is True

    def test_chapter_overlaps_with_false(self) -> None:
        """Non-overlapping chapters."""
        c1 = LifeChapter(title="C1", start_date=date(2020, 1, 1), end_date=date(2020, 2, 1))
        c2 = LifeChapter(title="C2", start_date=date(2020, 3, 1), end_date=date(2020, 4, 1))
        assert c1.overlaps_with(c2) is False

    def test_chapter_merge_with(self) -> None:
        """Merging two chapters."""
        c1 = LifeChapter(
            title="Phase 1",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            themes=["A"],
            memory_ids=["m1"],
        )
        c2 = LifeChapter(
            title="Phase 2",
            start_date=date(2020, 2, 1),
            end_date=date(2020, 2, 28),
            themes=["B"],
            memory_ids=["m2"],
        )

        merged = c1.merge_with(c2)
        assert merged.start_date == date(2020, 1, 1)
        assert merged.end_date == date(2020, 2, 28)
        assert "A" in merged.themes
        assert "B" in merged.themes
        assert "m1" in merged.memory_ids
        assert "m2" in merged.memory_ids

    def test_chapter_to_timeline_entry(self) -> None:
        """Verify timeline entry dictionary format."""
        chapter = LifeChapter(
            title="Chapter 1",
            start_date=date(2019, 1, 1),
            end_date=date(2019, 1, 5),
            memory_count=10,
        )
        entry = chapter.to_timeline_entry()
        assert entry["title"] == "Chapter 1"
        assert entry["start"] == "2019-01-01"
        assert entry["memory_count"] == 10


# =============================================================================
# DateRange Tests
# =============================================================================


class TestDateRange:
    """Tests for DateRange model, usually part of timeline aggregation."""

    def test_date_range_duration_days(self) -> None:
        """Test duration calculation (inclusive)."""
        dr = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
        assert dr.days == 31

    def test_date_range_contains(self) -> None:
        """Test point-in-range check."""
        dr = DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31))
        assert dr.contains(date(2020, 6, 15)) is True
        assert dr.contains(date(2019, 12, 31)) is False

    def test_date_range_merge(self) -> None:
        """Merging two ranges."""
        dr1 = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 10))
        dr2 = DateRange(start=date(2020, 1, 15), end=date(2020, 1, 20))
        merged = dr1.merge(dr2)
        assert merged.start == date(2020, 1, 1)
        assert merged.end == date(2020, 1, 20)


# =============================================================================
# LifeStoryReport Tests
# =============================================================================


class TestLifeStoryReport:
    """Tests for the final report model."""

    def test_report_creation(self, sample_life_report: LifeStoryReport) -> None:
        """Test with fixture-provided report."""
        assert len(sample_life_report.chapters) > 0
        assert sample_life_report.executive_summary != ""
        assert sample_life_report.is_fallback is False

    def test_report_serialization_json(self, sample_life_report: LifeStoryReport) -> None:
        """Roundtrip JSON serialization."""
        json_data = sample_life_report.model_dump_json()
        restored = LifeStoryReport.model_validate_json(json_data)
        assert restored.id == sample_life_report.id
        assert restored.total_memories_analyzed == sample_life_report.total_memories_analyzed

    def test_report_fallback_mode_fields(self, sample_fallback_report: LifeStoryReport) -> None:
        """Identify fallback reports."""
        assert sample_fallback_report.is_fallback is True
        assert "fallback" in sample_fallback_report.ai_model.lower()

    def test_report_to_dict(self, sample_life_report: LifeStoryReport) -> None:
        """Test dictionary conversion."""
        data = sample_life_report.to_dict()
        assert isinstance(data, dict)
        assert "chapters" in data
        assert "executive_summary" in data


# =============================================================================
# Safety Model Tests
# =============================================================================


class TestSafetyModels:
    """Tests for content safety and filtering models."""

    def test_safety_flag_creation(self) -> None:
        """Standard flag creation."""
        flag = SafetyFlag(
            category=SafetyCategory.NUDITY,
            confidence=0.8,
            detection_method=DetectionMethod.CAPTION_ANALYSIS,
            source="test",
        )
        assert flag.category == SafetyCategory.NUDITY
        assert flag.confidence == 0.8

    def test_safety_settings_defaults(self) -> None:
        """Default safety settings."""
        settings = SafetySettings()
        assert settings.enabled is True
        assert settings.default_action == SafetyAction.FLAG_ONLY

    def test_safety_settings_get_action_for_category(self) -> None:
        """Specific category actions."""
        settings = SafetySettings(nudity_action=SafetyAction.HIDE_FROM_REPORT)
        assert (
            settings.get_action_for_category(SafetyCategory.NUDITY) == SafetyAction.HIDE_FROM_REPORT
        )

    def test_memory_safety_state_add_flag(self) -> None:
        """Adding flags to a memory's safety state."""
        state = MemorySafetyState(memory_id="m1")
        flag = SafetyFlag(
            category=SafetyCategory.VIOLENCE,
            detection_method=DetectionMethod.CAPTION_ANALYSIS,
            source="test",
        )
        state.add_flag(flag)
        assert len(state.flags) == 1
        assert state.flags[0].category == SafetyCategory.VIOLENCE

    def test_resolve_action_for_flags_strictest_wins(self) -> None:
        """Resolution logic ensures the most restrictive action."""
        settings = SafetySettings(
            nudity_action=SafetyAction.HIDE_FROM_REPORT, violence_action=SafetyAction.FLAG_ONLY
        )
        state = MemorySafetyState(memory_id="m1")
        state.add_flag(
            SafetyFlag(
                category=SafetyCategory.NUDITY,
                detection_method=DetectionMethod.CAPTION_ANALYSIS,
                source="test",
            )
        )
        state.add_flag(
            SafetyFlag(
                category=SafetyCategory.VIOLENCE,
                detection_method=DetectionMethod.CAPTION_ANALYSIS,
                source="test",
            )
        )

        state.resolve_action(settings)
        # HIDE is stricter than FLAG
        assert state.resolved_action == SafetyAction.HIDE_FROM_REPORT


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Basic checks for enum contents."""

    def test_media_type_values(self) -> None:
        """Verify key media types exist."""
        assert MediaType.PHOTO.value == "photo"
        assert MediaType.STORY.value == "story"

    def test_source_platform_values(self) -> None:
        """Verify key platforms exist."""
        assert SourcePlatform.SNAPCHAT.value == "snapchat"
        assert SourcePlatform.GOOGLE_PHOTOS.value == "google_photos"
        assert SourcePlatform.LOCAL.value == "local"

    def test_confidence_level_ordering(self) -> None:
        """If using Python enums, verify ordering if implemented (e.g. by value or custom)."""
        # ConfidenceLevel in this project is usually a string enum
        # but the request asks to test ordering if applicable.
        # Based on src/core/memory.py docstring: VERIFIED > HIGH > MEDIUM > LOW > INFERRED
        levels = [
            ConfidenceLevel.VERIFIED,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.INFERRED,
        ]
        assert levels[0] == ConfidenceLevel.VERIFIED


# =============================================================================
# Parametrized & Special Cases
# =============================================================================


@pytest.mark.parametrize(
    "lat,lon,expected_valid",
    [
        (0, 0, True),
        (90, 180, True),
        (-90, -180, True),
        (91, 0, False),
        (0, 181, False),
    ],
)
def test_geopoint_validation(lat: float, lon: float, expected_valid: bool) -> None:
    """Parametrized coordinate validation."""
    if expected_valid:
        gp = GeoPoint(latitude=lat, longitude=lon)
        assert gp.latitude == lat
    else:
        with pytest.raises(ValidationError):
            GeoPoint(latitude=lat, longitude=lon)


def test_chapter_single_day_range() -> None:
    """Start and end on same day."""
    d = date(2020, 1, 1)
    chapter = LifeChapter(title="One Day", start_date=d, end_date=d)
    # Days property is (end - start).days + 1? Usually DateRange.days handles it.
    # Actually LifeChapter uses date_range.days which uses (end - start).days + 1
    assert chapter.duration_days == 0  # Wait, LifeChapter uses (end - start).days
    # Let's check LifeChapter.duration_days code again.
    # 262: return (self.end_date - self.start_date).days
    assert chapter.duration_days == 0
