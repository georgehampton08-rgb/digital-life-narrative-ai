"""Pytest fixtures for Digital Life Narrative AI tests.

Shared fixtures for mock data, temp directories, and AI client mocking.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

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
    PlatformBehaviorInsight,
    SourcePlatform,
)


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_geo_location() -> GeoLocation:
    """Sample geo location for testing."""
    return GeoLocation(
        latitude=41.8781,
        longitude=-87.6298,
        place_name="Chicago",
        country="USA",
    )


@pytest.fixture
def sample_media_item(sample_geo_location: GeoLocation) -> MediaItem:
    """Single sample media item."""
    return MediaItem(
        id=uuid.uuid4(),
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        file_path=Path("/photos/2020/vacation.jpg"),
        timestamp=datetime(2020, 7, 15, 14, 30, 0, tzinfo=timezone.utc),
        location=sample_geo_location,
        people=["Alice", "Bob"],
        caption="Summer vacation in Chicago",
        original_metadata={"album": "Vacation 2020"},
        file_hash="abc123def456",
        file_size_bytes=2048000,
        timestamp_confidence=Confidence.HIGH,
    )


@pytest.fixture
def sample_media_items() -> list[MediaItem]:
    """List of diverse MediaItem objects spanning 3 years, multiple platforms."""
    items = []

    # Year 1 - 2019 (Snapchat heavy)
    for i in range(10):
        items.append(MediaItem(
            id=uuid.uuid4(),
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.PHOTO if i % 2 == 0 else MediaType.VIDEO,
            file_path=Path(f"/snapchat/memories/snap_{i}.jpg"),
            timestamp=datetime(2019, 3 + i % 10, 10 + i, 12, 0, 0, tzinfo=timezone.utc),
            caption=f"Snap moment {i}",
        ))

    # Year 2 - 2020 (Google Photos)
    for i in range(15):
        items.append(MediaItem(
            id=uuid.uuid4(),
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            file_path=Path(f"/google_photos/takeout/IMG_{i:04d}.jpg"),
            timestamp=datetime(2020, 1 + i % 12, 15, 10, 30, 0, tzinfo=timezone.utc),
            location=GeoLocation(
                latitude=40.7128 + i * 0.1,
                longitude=-74.0060 + i * 0.1,
                place_name=f"Location {i}",
            ),
            people=["Alice"] if i % 3 == 0 else [],
        ))

    # Year 3 - 2021 (Mixed)
    for i in range(8):
        items.append(MediaItem(
            id=uuid.uuid4(),
            source_platform=SourcePlatform.LOCAL if i % 2 == 0 else SourcePlatform.INSTAGRAM,
            media_type=MediaType.PHOTO,
            file_path=Path(f"/local/photos/photo_{i}.jpg"),
            timestamp=datetime(2021, 6 + i % 6, 20, 15, 0, 0, tzinfo=timezone.utc),
        ))

    # Some items without timestamps
    items.append(MediaItem(
        id=uuid.uuid4(),
        source_platform=SourcePlatform.LOCAL,
        media_type=MediaType.PHOTO,
        file_path=Path("/local/unknown/photo.jpg"),
        timestamp=None,
    ))

    return items


@pytest.fixture
def sample_life_chapters() -> list[LifeChapter]:
    """Sample life chapters for testing."""
    return [
        LifeChapter(
            title="The Social Media Era",
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            themes=["social", "friends", "spontaneous"],
            narrative="A year of casual moments captured on Snapchat...",
            key_events=["Started new job", "Road trip with friends"],
            location_summary="Chicago area",
            media_count=10,
            representative_media_ids=[],
            confidence=Confidence.HIGH,
        ),
        LifeChapter(
            title="The Pandemic Year",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            themes=["home", "reflection", "adaptation"],
            narrative="A year of significant change and reflection...",
            key_events=["Started working from home", "Took up photography"],
            location_summary="Home",
            media_count=15,
            representative_media_ids=[],
            confidence=Confidence.MEDIUM,
        ),
        LifeChapter(
            title="New Beginnings",
            start_date=date(2021, 1, 1),
            end_date=date(2021, 12, 31),
            themes=["growth", "exploration"],
            narrative="Emerging from lockdown with new perspectives...",
            key_events=["Moved to new apartment"],
            location_summary="New York",
            media_count=8,
            representative_media_ids=[],
            confidence=Confidence.MEDIUM,
        ),
    ]


@pytest.fixture
def sample_platform_insights() -> list[PlatformBehaviorInsight]:
    """Sample platform insights for testing."""
    return [
        PlatformBehaviorInsight(
            platform=SourcePlatform.SNAPCHAT,
            usage_pattern="Quick, spontaneous moments with friends",
            peak_years=[2019],
            common_content_types=[MediaType.PHOTO, MediaType.VIDEO],
            unique_aspects=["Ephemeral content", "Friend-focused"],
        ),
        PlatformBehaviorInsight(
            platform=SourcePlatform.GOOGLE_PHOTOS,
            usage_pattern="Archival storage for important memories",
            peak_years=[2020, 2021],
            common_content_types=[MediaType.PHOTO],
            unique_aspects=["Auto-backup", "Event organization"],
        ),
    ]


@pytest.fixture
def sample_life_report(
    sample_life_chapters: list[LifeChapter],
    sample_platform_insights: list[PlatformBehaviorInsight],
) -> LifeStoryReport:
    """Pre-built LifeStoryReport for testing report generation."""
    return LifeStoryReport(
        generated_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        ai_model_used="gemini-1.5-pro",
        total_media_analyzed=33,
        date_range=(date(2019, 1, 1), date(2021, 12, 31)),
        executive_summary=(
            "This is a journey through three transformative years. "
            "From the spontaneous social moments of 2019, through the "
            "reflective pandemic year of 2020, to the new beginnings of 2021."
        ),
        chapters=sample_life_chapters,
        platform_insights=sample_platform_insights,
        detected_patterns=[
            "Transition from social to documentary photography",
            "Decreasing Snapchat usage over time",
        ],
        data_gaps=[
            DataGap(
                start_date=date(2020, 3, 15),
                end_date=date(2020, 5, 1),
                gap_days=47,
                possible_reasons=["Pandemic lockdown adjustment"],
            ),
        ],
        data_quality_notes=["Good timestamp coverage: 95%"],
        is_fallback_mode=False,
    )


@pytest.fixture
def sample_parse_result(sample_media_items: list[MediaItem]) -> ParseResult:
    """Sample parse result for testing."""
    return ParseResult(
        items=sample_media_items,
        source_paths=[Path("/snapchat"), Path("/google_photos"), Path("/local")],
        parse_errors=["Failed to parse one file"],
        stats={
            "total": len(sample_media_items),
            "photos": 25,
            "videos": 5,
        },
        duration_seconds=2.5,
    )


# =============================================================================
# Mock Export Directory Fixtures
# =============================================================================


@pytest.fixture
def snapchat_export_dir(tmp_path: Path) -> Path:
    """Temp directory with mock Snapchat export structure."""
    export_dir = tmp_path / "snapchat_export"
    export_dir.mkdir()

    # Create memories directory
    memories_dir = export_dir / "memories"
    memories_dir.mkdir()

    # Create memories_history.json
    memories_history = [
        {
            "Date": "2019-05-15 14:30:00 UTC",
            "Media Type": "PHOTO",
            "Location": "Chicago, IL",
        },
        {
            "Date": "2019-06-20 10:15:00 UTC",
            "Media Type": "VIDEO",
            "Location": "New York, NY",
        },
    ]
    (export_dir / "memories_history.json").write_text(
        json.dumps(memories_history), encoding="utf-8"
    )

    # Create sample media files (empty files for testing)
    (memories_dir / "snap_001.jpg").write_bytes(b"fake image data")
    (memories_dir / "snap_002.mp4").write_bytes(b"fake video data")

    # Create chat_history directory
    chat_dir = export_dir / "chat_history"
    chat_dir.mkdir()

    return export_dir


@pytest.fixture
def google_photos_export_dir(tmp_path: Path) -> Path:
    """Temp directory with mock Google Photos/Takeout structure."""
    export_dir = tmp_path / "google_takeout"
    export_dir.mkdir()

    photos_dir = export_dir / "Takeout" / "Google Photos"
    photos_dir.mkdir(parents=True)

    # Create album directory
    album_dir = photos_dir / "Vacation 2020"
    album_dir.mkdir()

    # Create photo with JSON sidecar
    photo_path = album_dir / "IMG_001.jpg"
    photo_path.write_bytes(b"fake image data")

    # JSON sidecar
    sidecar = {
        "title": "IMG_001.jpg",
        "photoTakenTime": {"timestamp": "1594828800"},  # 2020-07-16
        "geoData": {
            "latitude": 41.8781,
            "longitude": -87.6298,
        },
        "people": [{"name": "Alice"}],
    }
    sidecar_path = album_dir / "IMG_001.jpg.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    # Another photo
    (album_dir / "IMG_002.jpg").write_bytes(b"fake image 2")
    (album_dir / "IMG_002.jpg.json").write_text(json.dumps({
        "title": "IMG_002.jpg",
        "creationTime": {"timestamp": "1594915200"},
    }), encoding="utf-8")

    return export_dir


@pytest.fixture
def local_photos_dir(tmp_path: Path) -> Path:
    """Temp directory with sample local photos."""
    photos_dir = tmp_path / "local_photos"
    photos_dir.mkdir()

    # Create year folders
    year_2020 = photos_dir / "2020"
    year_2020.mkdir()

    year_2021 = photos_dir / "2021"
    year_2021.mkdir()

    # Create sample files with date patterns in names
    (year_2020 / "IMG_20200715_143000.jpg").write_bytes(b"fake photo 1")
    (year_2020 / "Screenshot_2020-08-10.png").write_bytes(b"fake screenshot")
    (year_2021 / "PXL_20210301_120000.jpg").write_bytes(b"fake pixel photo")

    return photos_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Clean temp directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Mock AI Client Fixtures
# =============================================================================


@pytest.fixture
def mock_ai_response() -> dict[str, Any]:
    """Predetermined AI response for testing."""
    return {
        "chapters": [
            {
                "title": "Test Chapter",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "themes": ["test", "mock"],
                "confidence": "high",
            }
        ],
        "narrative": "This is a test narrative generated by the mock AI.",
        "key_events": ["Test event 1", "Test event 2"],
    }


@pytest.fixture
def mock_ai_client(mock_ai_response: dict[str, Any]) -> MagicMock:
    """Mocked AIClient that returns predetermined responses."""
    mock_client = MagicMock()

    # Mock the model name
    mock_client.model_name = "gemini-mock"

    # Mock generate method
    mock_generate_response = MagicMock()
    mock_generate_response.text = json.dumps(mock_ai_response)
    mock_generate_response.usage_metadata = MagicMock()
    mock_generate_response.usage_metadata.prompt_token_count = 100
    mock_generate_response.usage_metadata.candidates_token_count = 50
    mock_client.generate.return_value = mock_generate_response

    # Mock generate_json method
    mock_client.generate_json.return_value = mock_ai_response

    # Mock count_tokens
    mock_client.count_tokens.return_value = 100

    # Mock is_available
    mock_client.is_available.return_value = True

    return mock_client


@pytest.fixture
def patch_ai_client(mock_ai_client: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch the AI client globally for a test."""
    with patch("organizer.ai.client.get_client", return_value=mock_ai_client):
        yield mock_ai_client


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def analysis_config() -> AnalysisConfig:
    """Default analysis configuration for tests."""
    return AnalysisConfig(
        min_chapter_duration_days=30,
        detect_gaps_threshold_days=60,
        max_chapters=10,
        include_platform_analysis=True,
        privacy_mode=False,
    )


@pytest.fixture
def privacy_config() -> AnalysisConfig:
    """Privacy-focused configuration for tests."""
    return AnalysisConfig(
        min_chapter_duration_days=30,
        detect_gaps_threshold_days=60,
        max_chapters=10,
        include_platform_analysis=True,
        privacy_mode=True,
    )
