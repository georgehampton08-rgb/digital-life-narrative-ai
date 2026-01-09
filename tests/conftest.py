"""Central Pytest Fixtures for Digital Life Narrative AI.

This module provides reusable test data, mock objects, and temporary directories
across all test modules. It ensures consistent test data and keeps tests DRY.

Fixtures included:
- Core data: sample_memories, sample_life_report, sample_fallback_report
- Directory structures: snapchat_export_dir, google_photos_export_dir, local_photos_dir
- AI Mocks: mock_ai_client, mock_ai_client_unavailable, mock_ai_client_rate_limited
- Safety: sample_safety_settings, memories_with_safety_flags
- Utilities: temp_output_dir
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.ai.client import AIRateLimitError, AIResponse, AIUnavailableError, StructuredAIResponse

# AI Models
from src.ai import (
    AIResponse,
    LifeChapter,
    LifeStoryReport,
    StructuredResponse,
)

# Core Models
from src.core.memory import (
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)

# Safety Models
from src.core.safety import (
    DetectionMethod,
    MemorySafetyState,
    SafetyAction,
    SafetyCategory,
    SafetyFlag,
    SafetySettings,
    SensitivityLevel,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_test_image(
    path: Path,
    width: int = 100,
    height: int = 100,
    color: str = "red",
    exif_datetime: datetime | None = None,
) -> Path:
    """Helper to create a test image with optional EXIF metadata.

    Args:
        path: Where to save the image.
        width: Width in pixels.
        height: Height in pixels.
        color: Solid color for the image.
        exif_datetime: Datetime to embed in EXIF (DateTimeOriginal).

    Returns:
        Path to the created image.
    """
    img = Image.new("RGB", (width, height), color=color)

    if exif_datetime:
        # Simple EXIF embedding
        exif = img.getexif()
        # 306 is DateTime, 36867 is DateTimeOriginal, 36868 is DateTimeDigitized
        exif[36867] = exif_datetime.strftime("%Y:%m:%d %H:%M:%S")
        img.save(path, exif=exif)
    else:
        img.save(path)

    return path


def create_test_json(path: Path, data: dict) -> Path:
    """Helper to create a JSON file.

    Args:
        path: Where to save the JSON.
        data: Dictionary to write.

    Returns:
        Path to the created JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def generate_mock_chapter_response() -> dict:
    """Generate the JSON structure that AI would return for chapter detection."""
    return {
        "chapters": [
            {
                "title": "Early Years & Exploration",
                "start_date": "2017-01-01",
                "end_date": "2018-12-31",
                "themes": ["discovery", "travel", "outdoors"],
                "location_summary": "National Parks, Various Cities",
                "confidence": "high",
                "reasoning": "Consistent travel patterns and outdoor activity.",
            },
            {
                "title": "The Big Transition",
                "start_date": "2019-01-01",
                "end_date": "2020-06-30",
                "themes": ["career", "new beginnings", "urban life"],
                "location_summary": "San Francisco, CA",
                "confidence": "high",
                "reasoning": "Major shift in location and daily routines.",
            },
            {
                "title": "Finding Stability",
                "start_date": "2020-07-01",
                "end_date": "2022-12-31",
                "themes": ["community", "growth", "home"],
                "location_summary": "San Francisco & Pacific Northwest",
                "confidence": "medium",
                "reasoning": "Settled patterns with occasional regional travel.",
            },
        ]
    }


def generate_mock_narrative_response(title: str = "A Period of Growth") -> dict:
    """Generate the JSON structure for narrative generation."""
    return {
        "narrative": f"This period, known as '{title}', represents a significant chapter in the journey. The documentation shows a shift towards more intentional activities and consistent social connections. It was a time of exploration that eventually led to a more grounded sense of purpose.",
        "key_events": [
            "Moving to a new city and establishing a home base",
            "Consistent documentation of morning routines",
            "Major career milestone captured in celebratory photos",
        ],
        "emotional_arc": "Transition from uncertainty to a confident equilibrium.",
    }


# =============================================================================
# Core Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_memories() -> list[Memory]:
    """Create a realistic set of 20-25 Memory objects for testing.

    - Spans 2017-2022 (plus some edge cases)
    - 3 Platforms: SNAPCHAT, GOOGLE_PHOTOS, LOCAL
    - Media Types: PHOTO, VIDEO, STORY, MESSAGE
    - Mixed metadata (captions, locations, people)
    - Includes edge cases: sensitive, None timestamps, future/old dates
    """
    memories = []
    base_date = datetime(2017, 1, 1, tzinfo=timezone.utc)

    # --- SNAPCHAT (8 memories) ---
    for i in range(8):
        dt = base_date + timedelta(days=i * 100)
        memories.append(
            Memory(
                source_platform=SourcePlatform.SNAPCHAT,
                media_type=MediaType.STORY if i % 3 == 0 else MediaType.PHOTO,
                created_at=dt,
                caption=f"Snapchat memory {i}" if i % 2 == 0 else None,
                location=Location(locality="SnapCity") if i % 4 == 0 else None,
                people=[PersonTag(name="Alice")] if i == 1 else [],
            )
        )

    # --- GOOGLE_PHOTOS (10 memories) ---
    for i in range(10):
        dt = base_date + timedelta(days=i * 150 + 200)
        memories.append(
            Memory(
                source_platform=SourcePlatform.GOOGLE_PHOTOS,
                media_type=MediaType.VIDEO if i == 0 else MediaType.PHOTO,
                created_at=dt,
                caption=f"Google Photos item {i}" if i < 7 else None,
                location=(
                    Location(
                        coordinates=GeoPoint(latitude=37.7749, longitude=-122.4194),
                        place_name="San Francisco",
                    )
                    if i % 3 == 0
                    else None
                ),
                people=[PersonTag(name="Bob"), PersonTag(name="Charlie")] if i == 2 else [],
            )
        )

    # --- LOCAL (7 memories) ---
    for i in range(7):
        dt = base_date + timedelta(days=i * 200 + 400)
        memories.append(
            Memory(
                source_platform=SourcePlatform.LOCAL,
                media_type=MediaType.MESSAGE if i % 4 == 0 else MediaType.PHOTO,
                created_at=dt,
                caption=f"Local media {i}",
                location=Location(region="California") if i == 1 else None,
            )
        )

    # --- SPECIAL CASES ---

    # 2 Sensitive captions
    memories.append(
        Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            created_at=base_date + timedelta(days=50),
            caption="A private and hidden nsfw photo",
        )
    )
    memories.append(
        Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.PHOTO,
            created_at=base_date + timedelta(days=60),
            caption="Hidden private folder content",
        )
    )

    # 2 None timestamps
    memories.append(
        Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=None,
            caption="Memory with no date",
        )
    )
    memories.append(
        Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.VIDEO,
            created_at=None,
            caption="Another undated memory",
        )
    )

    # Future timestamp
    memories.append(
        Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            created_at=datetime.now(timezone.utc) + timedelta(days=365),
            caption="Future photo test",
        )
    )

    # Very old timestamp (2010)
    memories.append(
        Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            created_at=datetime(2010, 5, 15, tzinfo=timezone.utc),
            caption="Heritage photo",
        )
    )

    # Sort: chronological, None values last
    memories.sort(key=lambda m: (m.created_at is None, m.created_at))
    return memories


# =============================================================================
# Directory Fixtures
# =============================================================================


@pytest.fixture
def snapchat_export_dir(tmp_path: Path) -> Path:
    """Create a fake Snapchat export directory structure."""
    export_path = tmp_path / "snapchat_export"
    export_path.mkdir()

    create_test_json(
        export_path / "account.json", {"username": "test_user", "email": "test@example.com"}
    )

    create_test_json(
        export_path / "memories_history.json",
        {
            "Saved Media": [
                {
                    "Date": "2019-06-15 14:30:22 UTC",
                    "Media Type": "PHOTO",
                    "Location": "37.7749, -122.4194",
                },
                {"Date": "2020-01-01 12:00:00 UTC", "Media Type": "PHOTO"},
                {"Date": "2021-03-15 09:30:00 UTC", "Media Type": "VIDEO"},
            ]
        },
    )

    mem_dir = export_path / "memories"
    mem_dir.mkdir()
    create_test_image(mem_dir / "memory_001.jpg", 10, 10)
    create_test_image(mem_dir / "memory_002.jpg", 10, 10)
    (mem_dir / "memory_003.mp4").write_text("fake video content")

    chat_dir = export_path / "chat_history" / "friend_alice"
    chat_dir.mkdir(parents=True)
    create_test_json(
        chat_dir / "messages.json",
        [
            {"timestamp": "2020-05-20 10:00:00", "text": "Hello!"},
            {"timestamp": "2020-05-20 10:05:00", "text": "See you soon."},
        ],
    )
    create_test_image(chat_dir / "shared_photo.jpg", 10, 10)

    snap_dir = export_path / "snap_history"
    snap_dir.mkdir()
    create_test_json(
        snap_dir / "snap_history.json",
        {"Sent Snaps": [{"Timestamp": "2021-01-01 00:00:00"}], "Received Snaps": []},
    )

    return export_path


@pytest.fixture
def google_photos_export_dir(tmp_path: Path) -> Path:
    """Create a fake Google Photos Takeout structure."""
    takeout_path = tmp_path / "Takeout"
    gp_path = takeout_path / "Google Photos"
    gp_path.mkdir(parents=True)

    # Photos from 2019
    p2019 = gp_path / "Photos from 2019"
    p2019.mkdir()
    create_test_image(p2019 / "IMG_0001.jpg")
    create_test_json(
        p2019 / "IMG_0001.jpg.json",
        {
            "photoTakenTime": {"timestamp": "1560616222"},
            "geoData": {"latitude": 40.7128, "longitude": -74.0060},
            "description": "New York trip",
        },
    )
    create_test_image(p2019 / "IMG_0002.jpg")
    create_test_json(p2019 / "IMG_0002.jpg.json", {"photoTakenTime": {"timestamp": "1570616222"}})

    # Photos from 2020
    p2020 = gp_path / "Photos from 2020"
    p2020.mkdir()
    (p2020 / "VID_0001.mp4").write_text("video")
    create_test_json(
        p2020 / "VID_0001.mp4.json",
        {"photoTakenTime": {"timestamp": "1580616222"}, "people": [{"name": "Alice"}]},
    )

    # Album
    album_path = gp_path / "Album - Summer Trip"
    album_path.mkdir()
    create_test_image(album_path / "photo.jpg")
    create_test_json(album_path / "photo.jpg.json", {"photoTakenTime": {"timestamp": "1623765022"}})
    create_test_json(album_path / "metadata.json", {"album": "Summer Trip"})

    return takeout_path


@pytest.fixture
def local_photos_dir(tmp_path: Path) -> Path:
    """Create a directory with generic local photos."""
    local_path = tmp_path / "local_photos"
    local_path.mkdir()

    # With EXIF
    create_test_image(
        local_path / "IMG_20190615_143022.jpg", exif_datetime=datetime(2019, 6, 15, 14, 30, 22)
    )
    create_test_image(
        local_path / "IMG_20200101_120000.jpg", exif_datetime=datetime(2020, 1, 1, 12, 0, 0)
    )

    # No EXIF (PNG)
    img_png = Image.new("RGB", (100, 100), color="blue")
    img_png.save(local_path / "Screenshot_20210315-093000.png")

    # No EXIF, no date in name
    create_test_image(local_path / "random_photo.jpg")

    # Video
    (local_path / "video.mp4").write_text("vid")

    return local_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty directory for testing 'no detection' cases."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def non_media_dir(tmp_path: Path) -> Path:
    """Create a directory with non-media files only."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "document.pdf").write_text("%PDF-1.4")
    (d / "readme.txt").write_text("Hello world")
    (d / "data.csv").write_text("id,name\n1,test")
    return d


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a clean output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# =============================================================================
# Mock AI Fixtures
# =============================================================================


@pytest.fixture
def mock_ai_client() -> MagicMock:
    """Create a mock AIClient that returns predictable responses."""
    mock = MagicMock()
    mock.is_available.return_value = True

    def intelligent_generate(prompt, **kwargs):
        prompt_lower = prompt.lower()
        text = "Generic AI response stub."
        if "narrative" in prompt_lower:
            text = generate_mock_narrative_response()["narrative"]
        elif "summary" in prompt_lower:
            text = "This life story traces a remarkable journey of growth and transformation. From early explorations to settling in a vibrant city, every moment captured contributes to a rich tapestry of experiences."

        return AIResponse(
            text=text,
            model="gemini-1.5-pro",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            finish_reason="STOP",
        )

    def intelligent_generate_json(prompt, **kwargs):
        prompt_lower = prompt.lower()
        data = {"message": "Success"}
        if "chapter" in prompt_lower:
            data = generate_mock_chapter_response()
        elif "platform" in prompt_lower:
            data = {
                "insights": [
                    {
                        "platform": "Snapchat",
                        "pattern": "Frequent daily captures of small moments.",
                    },
                    {
                        "platform": "Google Photos",
                        "pattern": "High-quality archival photos of milestone events.",
                    },
                    {"platform": "Local", "pattern": "Diverse mix of personal and captured media."},
                ]
            }
        elif "narrative" in prompt_lower:
            data = generate_mock_narrative_response()

        return StructuredAIResponse(
            data=data,
            raw_text=json.dumps(data),
            model="gemini-1.5-pro",
            tokens_used=200,
            parse_success=True,
        )

    mock.generate.side_effect = intelligent_generate
    mock.generate_json.side_effect = intelligent_generate_json
    mock.generate_structured.side_effect = intelligent_generate_json
    mock.count_tokens.side_effect = lambda text: len(text) // 4

    return mock


@pytest.fixture
def mock_ai_client_unavailable() -> MagicMock:
    """Create a mock AIClient that simulates unavailability."""
    mock = MagicMock()
    mock.is_available.return_value = False
    mock.generate.side_effect = AIUnavailableError("disabled")
    mock.generate_json.side_effect = AIUnavailableError("disabled")
    mock.generate_structured.side_effect = AIUnavailableError("disabled")
    return mock


@pytest.fixture
def mock_ai_client_rate_limited() -> MagicMock:
    """Create a mock that fails with rate limit then succeeds."""
    mock = MagicMock()
    mock.is_available.return_value = True

    # State tracker for calls
    call_counts = {"generate": 0}

    def rate_limited_side_effect(*args, **kwargs):
        call_counts["generate"] += 1
        if call_counts["generate"] == 1:
            raise AIRateLimitError("Rate limit exceeded")
        return AIResponse(text="Success after retry", model="gemini-1.5-pro", total_tokens=100)

    mock.generate.side_effect = rate_limited_side_effect
    return mock


# =============================================================================
# Report Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_life_report() -> LifeStoryReport:
    """Create a complete LifeStoryReport for testing report generation."""
    chapters = [
        LifeChapter(
            title="The Early Chapters",
            start_date=date(2018, 1, 1),
            end_date=date(2019, 12, 31),
            themes=["beginnings", "exploration"],
            narrative="The journey began with a series of exploratory steps. This period was marked by a high frequency of outdoor activities and travel to local destinations.",
            key_events=["First solo trip", "Graduation ceremony"],
        ),
        LifeChapter(
            title="New Horizons",
            start_date=date(2020, 1, 1),
            end_date=date(2021, 6, 30),
            themes=["transition", "challenges"],
            narrative="A shift occurred as new challenges arose. The documentation shows a transition from group-based activities to more reflective, individual pursuits.",
            key_events=["Relocation to San Francisco", "Starting new role"],
        ),
        LifeChapter(
            title="Finding Our Rhythm",
            start_date=date(2021, 7, 1),
            end_date=date(2022, 12, 31),
            themes=["stability", "community"],
            narrative="Stability was found in the routine. This chapter highlights the importance of consistent community connection and the growth of long-term projects.",
            key_events=["Home renovation complete", "Community project launch"],
        ),
    ]

    return LifeStoryReport(
        generated_at=datetime.now(timezone.utc),
        ai_model="gemini-1.5-pro",
        total_memories_analyzed=150,
        executive_summary="This narrative provides a comprehensive look at a five-year journey of transformation. It captures the essential shift from early exploration to finding meaningful stability and community connection.\n\nThrough the lens of over 150 moments, we see an individual who values both adventure and roots, consistently documenting the transitions that define their story.",
        chapters=chapters,
        platform_insights=[
            PlatformBehaviorInsight(
                platform=SourcePlatform.SNAPCHAT,
                usage_pattern="Spontaneous, daily captures of social interactions.",
            ),
            PlatformBehaviorInsight(
                platform=SourcePlatform.GOOGLE_PHOTOS,
                usage_pattern="Curated archive of milestone events.",
            ),
        ],
        detected_patterns=["Consistent Sunday morning routines", "Seasonal travel peaks"],
        data_gaps=[
            DataGap(
                start_date=date(2019, 6, 1),
                end_date=date(2019, 8, 30),
                duration_days=90,
                severity="moderate",
            )
        ],
        is_fallback=False,
    )


@pytest.fixture(scope="session")
def sample_fallback_report() -> LifeStoryReport:
    """Create a fallback mode LifeStoryReport."""
    chapters = [
        LifeChapter(
            title="Year 2019",
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            narrative="Yearly summary for 2019. AI narrative analysis was unavailable for this period.",
        ),
        LifeChapter(
            title="Year 2020",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            narrative="Yearly summary for 2020. AI narrative analysis was unavailable for this period.",
        ),
    ]

    return LifeStoryReport(
        generated_at=datetime.now(timezone.utc),
        ai_model="none (fallback mode)",
        total_memories_analyzed=50,
        executive_summary="This report was generated in fallback mode because AI analysis was unavailable. It provides a simple chronological grouping of your memories by year.",
        chapters=chapters,
        is_fallback=True,
    )


# =============================================================================
# Safety Fixtures
# =============================================================================


@pytest.fixture
def sample_safety_settings() -> SafetySettings:
    """Create SafetySettings for testing."""
    return SafetySettings(
        enabled=True,
        sensitivity=SensitivityLevel.MODERATE,
        nudity_action=SafetyAction.BLUR_IN_REPORT,
        sexual_action=SafetyAction.HIDE_FROM_REPORT,
        violence_action=SafetyAction.FLAG_ONLY,
        use_pixel_analysis=False,
    )


@pytest.fixture
def memories_with_safety_flags(sample_memories) -> list[tuple[Memory, MemorySafetyState]]:
    """Create memories paired with pre-assigned safety states."""
    results = []

    # Select a few memories to flag
    m1 = sample_memories[0]
    s1 = MemorySafetyState(memory_id=m1.id)
    s1.add_flag(
        SafetyFlag(
            category=SafetyCategory.NUDITY,
            detection_method=DetectionMethod.METADATA_HEURISTIC,
            source="test_source",
            confidence=0.9,
        )
    )
    s1.resolve_action(SafetySettings())
    results.append((m1, s1))

    m2 = sample_memories[1]
    s2 = MemorySafetyState(memory_id=m2.id)
    s2.add_flag(
        SafetyFlag(
            category=SafetyCategory.NUDITY,
            detection_method=DetectionMethod.FILENAME_HEURISTIC,
            source="test_source",
            confidence=0.85,
        )
    )
    s2.resolve_action(SafetySettings())
    results.append((m2, s2))

    m3 = sample_memories[2]
    s3 = MemorySafetyState(memory_id=m3.id)
    s3.add_flag(
        SafetyFlag(
            category=SafetyCategory.PRIVATE,
            detection_method=DetectionMethod.FILENAME_HEURISTIC,
            source="test_source",
            confidence=0.95,
        )
    )
    s3.resolve_action(SafetySettings())
    results.append((m3, s3))

    # Rest are allowed
    for m in sample_memories[3:]:
        s = MemorySafetyState(memory_id=m.id, resolved_action=SafetyAction.ALLOW)
        results.append((m, s))

    return results
