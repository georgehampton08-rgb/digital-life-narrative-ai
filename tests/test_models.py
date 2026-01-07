"""Unit tests for the data models in organizer/models.py."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from organizer.models import (
    AnalysisConfig,
    Confidence,
    DataGap,
    GeoLocation,
    LifeChapter,
    LifeStoryReport,
    MediaItem,
    MediaType,
    ParseResult,
    SourcePlatform,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_media_type_values(self) -> None:
        """Test MediaType enum has expected values."""
        assert MediaType.PHOTO.value == "photo"
        assert MediaType.VIDEO.value == "video"
        assert MediaType.AUDIO.value == "audio"
        assert MediaType.UNKNOWN.value == "unknown"

    def test_source_platform_values(self) -> None:
        """Test SourcePlatform enum has expected values."""
        assert SourcePlatform.SNAPCHAT.value == "snapchat"
        assert SourcePlatform.GOOGLE_PHOTOS.value == "google_photos"
        assert SourcePlatform.FACEBOOK.value == "facebook"
        assert SourcePlatform.INSTAGRAM.value == "instagram"
        assert SourcePlatform.LOCAL.value == "local"

    def test_confidence_values(self) -> None:
        """Test Confidence enum has expected values."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"


# =============================================================================
# GeoLocation Tests
# =============================================================================


class TestGeoLocation:
    """Tests for GeoLocation model."""

    def test_valid_geo_location(self) -> None:
        """Test creating valid GeoLocation."""
        loc = GeoLocation(
            latitude=41.8781,
            longitude=-87.6298,
            place_name="Chicago",
            country="USA",
        )
        assert loc.latitude == 41.8781
        assert loc.longitude == -87.6298
        assert loc.place_name == "Chicago"

    def test_geo_location_minimal(self) -> None:
        """Test GeoLocation with only lat/lon."""
        loc = GeoLocation(latitude=0.0, longitude=0.0)
        assert loc.latitude == 0.0
        assert loc.longitude == 0.0
        assert loc.place_name is None

    @pytest.mark.parametrize(
        "latitude,longitude,valid",
        [
            (0.0, 0.0, True),
            (90.0, 180.0, True),
            (-90.0, -180.0, True),
            (41.8781, -87.6298, True),  # Chicago
            (91.0, 0.0, False),  # Invalid latitude
            (-91.0, 0.0, False),  # Invalid latitude
            (0.0, 181.0, False),  # Invalid longitude
            (0.0, -181.0, False),  # Invalid longitude
        ],
    )
    def test_geo_location_bounds(
        self,
        latitude: float,
        longitude: float,
        valid: bool,
    ) -> None:
        """Test latitude/longitude validation bounds."""
        if valid:
            loc = GeoLocation(latitude=latitude, longitude=longitude)
            assert loc.latitude == latitude
            assert loc.longitude == longitude
        else:
            with pytest.raises(ValidationError):
                GeoLocation(latitude=latitude, longitude=longitude)


# =============================================================================
# MediaItem Tests
# =============================================================================


class TestMediaItem:
    """Tests for MediaItem model."""

    def test_media_item_full(self) -> None:
        """Test MediaItem creation with all fields."""
        item = MediaItem(
            id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            file_path=Path("/photos/test.jpg"),
            timestamp=datetime(2020, 5, 15, 10, 30, 0, tzinfo=timezone.utc),
            location=GeoLocation(latitude=40.0, longitude=-74.0, place_name="NYC"),
            people=["Alice", "Bob"],
            caption="Test photo",
            original_metadata={"album": "Summer"},
            file_hash="abc123",
            file_size_bytes=1024000,
            timestamp_confidence=Confidence.HIGH,
        )

        assert item.source_platform == SourcePlatform.GOOGLE_PHOTOS
        assert item.media_type == MediaType.PHOTO
        assert item.file_path == Path("/photos/test.jpg")
        assert len(item.people) == 2
        assert item.file_hash == "abc123"

    def test_media_item_minimal(self) -> None:
        """Test MediaItem creation with minimal fields (test defaults)."""
        item = MediaItem(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=Path("/photo.jpg"),
        )

        assert item.id is not None  # Auto-generated UUID
        assert item.timestamp is None
        assert item.location is None
        assert item.people == []
        assert item.caption is None
        assert item.original_metadata == {}
        assert item.file_hash is None
        assert item.timestamp_confidence == Confidence.MEDIUM

    def test_media_item_auto_uuid(self) -> None:
        """Test that MediaItem generates unique UUIDs."""
        item1 = MediaItem(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=Path("/photo1.jpg"),
        )
        item2 = MediaItem(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=Path("/photo2.jpg"),
        )

        assert item1.id != item2.id

    def test_media_item_to_ai_summary(self) -> None:
        """Test MediaItem.to_ai_summary() returns correct format."""
        item = MediaItem(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.VIDEO,
            file_path=Path("/path/to/snap.mp4"),
            timestamp=datetime(2020, 6, 15, tzinfo=timezone.utc),
            location=GeoLocation(latitude=41.0, longitude=-87.0, place_name="Chicago"),
            people=["Alice"],
            caption="Fun day!",
        )

        summary = item.to_ai_summary()

        assert summary["platform"] == "snapchat"
        assert summary["type"] == "video"
        assert "2020-06-15" in summary["date"]
        assert summary["location"] == "Chicago"
        assert summary["people"] == ["Alice"]
        assert summary["caption"] == "Fun day!"

    def test_media_item_to_ai_summary_privacy_mode(self) -> None:
        """Test MediaItem.to_ai_summary() privacy sanitization."""
        item = MediaItem(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=Path("/users/john/secret/photo.jpg"),
            timestamp=datetime(2020, 6, 15, tzinfo=timezone.utc),
            caption="With family at home",
        )

        summary = item.to_ai_summary(privacy_mode=True)

        # File path should be anonymized
        assert "file_path" not in summary or "john" not in str(summary.get("file_path", ""))

    def test_media_item_to_ai_summary_minimal(self) -> None:
        """Test to_ai_summary with minimal data."""
        item = MediaItem(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            file_path=Path("/photo.jpg"),
        )

        summary = item.to_ai_summary()

        assert summary["platform"] == "local"
        assert summary["type"] == "photo"
        assert summary.get("date") is None
        assert summary.get("location") is None


# =============================================================================
# LifeChapter Tests
# =============================================================================


class TestLifeChapter:
    """Tests for LifeChapter model."""

    def test_life_chapter_creation(self) -> None:
        """Test LifeChapter creation."""
        chapter = LifeChapter(
            title="College Years",
            start_date=date(2018, 9, 1),
            end_date=date(2022, 5, 31),
            themes=["education", "growth", "friendships"],
            narrative="Four transformative years...",
            key_events=["Graduation"],
            location_summary="State University",
            media_count=500,
            representative_media_ids=[uuid.uuid4()],
            confidence=Confidence.HIGH,
        )

        assert chapter.title == "College Years"
        assert (chapter.end_date - chapter.start_date).days > 1000
        assert len(chapter.themes) == 3

    def test_life_chapter_minimal(self) -> None:
        """Test LifeChapter with minimal fields."""
        chapter = LifeChapter(
            title="Year 2020",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            themes=[],
            narrative="",
            key_events=[],
            media_count=0,
            representative_media_ids=[],
            confidence=Confidence.LOW,
        )

        assert chapter.location_summary is None


# =============================================================================
# LifeStoryReport Tests
# =============================================================================


class TestLifeStoryReport:
    """Tests for LifeStoryReport model."""

    def test_life_story_report_creation(self, sample_life_report: LifeStoryReport) -> None:
        """Test LifeStoryReport creation."""
        assert sample_life_report.total_media_analyzed == 33
        assert len(sample_life_report.chapters) == 3
        assert sample_life_report.is_fallback_mode is False

    def test_life_story_report_serialization(self, sample_life_report: LifeStoryReport) -> None:
        """Test LifeStoryReport serialization to JSON."""
        json_str = sample_life_report.model_dump_json()
        data = json.loads(json_str)

        assert data["total_media_analyzed"] == 33
        assert len(data["chapters"]) == 3
        assert data["ai_model_used"] == "gemini-1.5-pro"
        assert data["is_fallback_mode"] is False

    def test_life_story_report_roundtrip(self, sample_life_report: LifeStoryReport) -> None:
        """Test JSON roundtrip serialization."""
        json_str = sample_life_report.model_dump_json()
        restored = LifeStoryReport.model_validate_json(json_str)

        assert restored.total_media_analyzed == sample_life_report.total_media_analyzed
        assert len(restored.chapters) == len(sample_life_report.chapters)
        assert restored.executive_summary == sample_life_report.executive_summary

    def test_years_covered_property(self, sample_life_report: LifeStoryReport) -> None:
        """Test years_covered computed property."""
        years = sample_life_report.years_covered
        assert years == 3  # 2019, 2020, 2021


# =============================================================================
# ParseResult Tests
# =============================================================================


class TestParseResult:
    """Tests for ParseResult model."""

    def test_parse_result_creation(self, sample_parse_result: ParseResult) -> None:
        """Test ParseResult creation."""
        assert len(sample_parse_result.items) > 0
        assert len(sample_parse_result.source_paths) == 3
        assert sample_parse_result.duration_seconds == 2.5

    def test_parse_result_stats(self, sample_parse_result: ParseResult) -> None:
        """Test ParseResult statistics calculation."""
        assert "total" in sample_parse_result.stats
        assert sample_parse_result.stats["total"] == len(sample_parse_result.items)


# =============================================================================
# DataGap Tests
# =============================================================================


class TestDataGap:
    """Tests for DataGap model."""

    def test_data_gap_creation(self) -> None:
        """Test DataGap creation."""
        gap = DataGap(
            start_date=date(2020, 3, 1),
            end_date=date(2020, 6, 1),
            gap_days=92,
            possible_reasons=["Pandemic", "Lost phone"],
        )

        assert gap.gap_days == 92
        assert len(gap.possible_reasons) == 2

    def test_data_gap_minimal(self) -> None:
        """Test DataGap with minimal fields."""
        gap = DataGap(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 2, 1),
            gap_days=31,
        )

        assert gap.possible_reasons == []


# =============================================================================
# AnalysisConfig Tests
# =============================================================================


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_analysis_config_defaults(self) -> None:
        """Test AnalysisConfig default values."""
        config = AnalysisConfig()

        assert config.min_chapter_duration_days == 30
        assert config.max_chapters == 20
        assert config.privacy_mode is False

    def test_analysis_config_custom(self) -> None:
        """Test AnalysisConfig custom values."""
        config = AnalysisConfig(
            min_chapter_duration_days=60,
            max_chapters=5,
            privacy_mode=True,
        )

        assert config.min_chapter_duration_days == 60
        assert config.max_chapters == 5
        assert config.privacy_mode is True


# =============================================================================
# Validation Error Tests
# =============================================================================


class TestValidationErrors:
    """Tests for Pydantic validation errors."""

    def test_invalid_media_type(self) -> None:
        """Test validation error for invalid media type."""
        with pytest.raises(ValidationError):
            MediaItem(
                source_platform=SourcePlatform.LOCAL,
                media_type="invalid_type",  # type: ignore
                file_path=Path("/photo.jpg"),
            )

    def test_invalid_platform(self) -> None:
        """Test validation error for invalid platform."""
        with pytest.raises(ValidationError):
            MediaItem(
                source_platform="invalid_platform",  # type: ignore
                media_type=MediaType.PHOTO,
                file_path=Path("/photo.jpg"),
            )

    def test_missing_required_fields(self) -> None:
        """Test validation error for missing required fields."""
        with pytest.raises(ValidationError):
            MediaItem()  # type: ignore

    def test_invalid_date_range(self) -> None:
        """Test that invalid date range can be created (no built-in validation)."""
        # Note: Model allows this; validation should be done at usage
        chapter = LifeChapter(
            title="Invalid",
            start_date=date(2021, 1, 1),
            end_date=date(2020, 1, 1),  # End before start
            themes=[],
            narrative="",
            key_events=[],
            media_count=0,
            representative_media_ids=[],
            confidence=Confidence.LOW,
        )
        # Model allows it, but end is before start
        assert chapter.start_date > chapter.end_date
