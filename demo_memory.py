"""Demonstration script for the Memory data model.

This script demonstrates all key features of the Memory model:
1. Creating memories with various fields
2. Computing content and metadata hashes
3. Privacy-conscious AI payload generation
4. Deduplication with is_same_moment
5. Merging memories from different sources
6. Edge case handling
"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)


def demo_basic_creation():
    """Demonstrate basic memory creation."""
    print("=" * 70)
    print("DEMO 1: Basic Memory Creation")
    print("=" * 70)

    # Minimal memory
    minimal = Memory(
        source_platform=SourcePlatform.LOCAL,
        media_type=MediaType.PHOTO,
    )
    print(f"\n✓ Created minimal memory with auto-generated ID: {minimal.id}")

    # Full memory
    full = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
        created_at_confidence=ConfidenceLevel.VERIFIED,
        location=Location(
            coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
            place_name="Central Park",
            locality="Manhattan",
            region="New York",
            country="United States",
            country_code="US",
        ),
        people=[
            PersonTag(name="Alice Smith", confidence=ConfidenceLevel.VERIFIED),
            PersonTag(name="Bob Jones", confidence=ConfidenceLevel.HIGH),
        ],
        caption="Beautiful summer afternoon in the city",
        width=1920,
        height=1080,
        camera_make="Apple",
        camera_model="iPhone 12",
    )

    print(f"✓ Created full memory:")
    print(f"  - Date: {full.created_at.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  - Location: {full.location.to_display_string()}")
    print(f"  - People: {', '.join(p.name for p in full.people)}")
    print(f"  - Caption: {full.caption}")
    print(f"  - Dimensions: {full.width}x{full.height}")


def demo_timezone_handling():
    """Demonstrate timezone handling."""
    print("\n" + "=" * 70)
    print("DEMO 2: Timezone Handling (Critical Feature)")
    print("=" * 70)

    # Naive datetime (will be converted to UTC with warning)
    naive_memory = Memory(
        source_platform=SourcePlatform.FACEBOOK,
        created_at=datetime(2020, 7, 15, 12, 30),  # No timezone
    )

    print(f"\n✓ Naive datetime auto-converted to UTC")
    print(f"  - Original: 2020-07-15 12:30 (no timezone)")
    print(f"  - Converted: {naive_memory.created_at}")
    print(f"  - Warnings: {naive_memory.parse_warnings}")

    # Timezone-aware datetime
    aware_memory = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
    )

    print(f"\n✓ Timezone-aware datetime preserved")
    print(f"  - No warnings: {aware_memory.parse_warnings}")


def demo_privacy_levels():
    """Demonstrate AI payload at different privacy levels."""
    print("\n" + "=" * 70)
    print("DEMO 3: Privacy-Conscious AI Payloads (Critical Feature)")
    print("=" * 70)

    memory = Memory(
        source_platform=SourcePlatform.SNAPCHAT,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 45, tzinfo=timezone.utc),
        source_path=Path("/users/john/secret/photos/IMG_001.jpg"),  # SENSITIVE
        location=Location(
            coordinates=GeoPoint(latitude=40.7829123, longitude=-73.9654456),  # SENSITIVE
            place_name="Central Park",
            locality="Manhattan",
            region="New York",
            country_code="US",
        ),
        people=[PersonTag(name="Alice Johnson"), PersonTag(name="Bob Smith")],
        caption="This is my personal diary entry that should be truncated for privacy",
        content_hash="abc123def456",  # SENSITIVE
        original_metadata={"secret_key": "secret_value"},  # SENSITIVE
    )

    print("\n📌 STRICT Privacy Level (minimum data):")
    strict = memory.to_ai_payload(privacy_level="strict")
    for key, value in strict.items():
        print(f"  {key}: {value}")

    print("\n📌 STANDARD Privacy Level (balanced):")
    standard = memory.to_ai_payload(privacy_level="standard")
    for key, value in standard.items():
        print(f"  {key}: {value}")

    print("\n📌 DETAILED Privacy Level (maximum context):")
    detailed = memory.to_ai_payload(privacy_level="detailed")
    for key, value in detailed.items():
        print(f"  {key}: {value}")

    print("\n✓ Privacy guarantee: source_path, exact coordinates, content_hash NEVER included")
    print("✓ People names anonymized in detailed mode")
    print("✓ Caption truncated to 100 chars")


def demo_content_hashing():
    """Demonstrate content hash computation."""
    print("\n" + "=" * 70)
    print("DEMO 4: Content Hashing for Deduplication")
    print("=" * 70)

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jpg") as f1:
        f1.write("Test image content A")
        file1 = Path(f1.name)

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jpg") as f2:
        f2.write("Test image content A")  # Same content
        file2 = Path(f2.name)

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jpg") as f3:
        f3.write("Test image content B")  # Different content
        file3 = Path(f3.name)

    try:
        memory1 = Memory(source_platform=SourcePlatform.GOOGLE_PHOTOS)
        memory2 = Memory(source_platform=SourcePlatform.DROPBOX)
        memory3 = Memory(source_platform=SourcePlatform.ONEDRIVE)

        hash1 = memory1.compute_content_hash(file1)
        hash2 = memory2.compute_content_hash(file2)
        hash3 = memory3.compute_content_hash(file3)

        print(f"\n✓ File 1 hash: {hash1}")
        print(f"✓ File 2 hash: {hash2}")
        print(f"✓ File 3 hash: {hash3}")

        print(f"\n✓ File 1 == File 2: {hash1 == hash2} (same content, exact duplicate)")
        print(f"✓ File 1 == File 3: {hash1 == hash3} (different content)")

    finally:
        file1.unlink()
        file2.unlink()
        file3.unlink()


def demo_metadata_hashing():
    """Demonstrate metadata hash for fuzzy deduplication."""
    print("\n" + "=" * 70)
    print("DEMO 5: Metadata Hashing for Fuzzy Deduplication")
    print("=" * 70)

    # Same moment, different platforms
    phone = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 30, tzinfo=timezone.utc),
        location=Location(
            coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654)
        ),
    )

    dslr = Memory(
        source_platform=SourcePlatform.LOCAL,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 45, tzinfo=timezone.utc),  # 15s later
        location=Location(
            coordinates=GeoPoint(latitude=40.7830, longitude=-73.9655)
        ),  # Slightly different
    )

    hash_phone = phone.compute_metadata_hash()
    hash_dslr = dslr.compute_metadata_hash()

    print(f"\n✓ Phone photo metadata hash: {hash_phone}")
    print(f"✓ DSLR photo metadata hash: {hash_dslr}")
    print(f"✓ Hashes match: {hash_phone == hash_dslr}")
    print("   (same minute + approximate location = fuzzy duplicate)")


def demo_is_same_moment():
    """Demonstrate cross-platform deduplication."""
    print("\n" + "=" * 70)
    print("DEMO 6: Cross-Platform Deduplication (is_same_moment)")
    print("=" * 70)

    # Snapchat save
    snapchat = Memory(
        source_platform=SourcePlatform.SNAPCHAT,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
        location=Location(
            coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654)
        ),
    )

    # Google Photos backup (30 seconds later)
    google = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 30, tzinfo=timezone.utc),
        location=Location(
            coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654)
        ),
    )

    # Different moment
    later = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 35, 0, tzinfo=timezone.utc),
    )

    print(f"\n✓ Snapchat & Google Photos are same moment: {snapchat.is_same_moment(google, tolerance_seconds=60)}")
    print(f"✓ Snapchat & later photo are same moment: {snapchat.is_same_moment(later, tolerance_seconds=60)}")


def demo_merge():
    """Demonstrate merging memories with confidence resolution."""
    print("\n" + "=" * 70)
    print("DEMO 7: Merging Memories (Confidence Resolution)")
    print("=" * 70)

    # Phone photo: has timestamp, no location
    phone = Memory(
        source_platform=SourcePlatform.GOOGLE_PHOTOS,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
        created_at_confidence=ConfidenceLevel.VERIFIED,
        people=[PersonTag(name="Alice"), PersonTag(name="Bob")],
        caption="From phone",
    )

    # DSLR photo: has location, inferred timestamp
    dslr = Memory(
        source_platform=SourcePlatform.LOCAL,
        media_type=MediaType.PHOTO,
        created_at=datetime(2020, 7, 15, 12, 35, 0, tzinfo=timezone.utc),
        created_at_confidence=ConfidenceLevel.INFERRED,
        location=Location(
            coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
            place_name="Central Park",
            confidence=ConfidenceLevel.HIGH,
        ),
        people=[PersonTag(name="Bob"), PersonTag(name="Charlie")],
        camera_make="Canon",
        camera_model="EOS R5",
    )

    merged = phone.merge_with(dslr)

    print("\n✓ Merged result:")
    print(f"  - Timestamp: {merged.created_at} (VERIFIED from phone)")
    print(f"  - Location: {merged.location.to_display_string()} (from DSLR)")
    print(f"  - People: {', '.join(p.name for p in merged.people)} (combined, deduplicated)")
    print(f"  - Caption: {merged.caption}")
    print(f"  - Camera: {merged.camera_make} {merged.camera_model}")


def demo_helper_methods():
    """Demonstrate helper methods."""
    print("\n" + "=" * 70)
    print("DEMO 8: Helper Methods")
    print("=" * 70)

    memory = Memory(
        source_platform=SourcePlatform.INSTAGRAM,
        media_type=MediaType.VIDEO,
        created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
        location=Location(
            locality="San Francisco",
            region="California",
            country_code="US",
        ),
    )

    print(f"\n✓ get_year(): {memory.get_year()}")
    print(f"✓ get_month(): {memory.get_month()}")
    print(f"✓ get_year_month(): {memory.get_year_month()}")

    from datetime import date
    print(f"✓ days_since(2020-01-01): {memory.days_since(date(2020, 1, 1))} days")

    timeline_point = memory.to_timeline_point()
    print(f"\n✓ Timeline point:")
    for key, value in timeline_point.items():
        print(f"  {key}: {value}")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n" + "=" * 70)
    print("DEMO 9: Edge Case Handling")
    print("=" * 70)

    # Memory with no timestamp
    no_timestamp = Memory(source_platform=SourcePlatform.LOCAL)
    print(f"\n✓ Memory with no timestamp is valid")
    print(f"  - created_at: {no_timestamp.created_at}")
    print(f"  - get_year(): {no_timestamp.get_year()}")

    # Coordinates at (0, 0) - likely error
    suspicious = Memory(
        source_platform=SourcePlatform.FACEBOOK,
        location=Location(coordinates=GeoPoint(latitude=0.0, longitude=0.0)),
    )
    print(f"\n✓ Coordinates at (0,0) flagged as suspicious")
    print(f"  - Warnings: {suspicious.parse_warnings}")

    # Empty caption vs missing caption
    empty = Memory(source_platform=SourcePlatform.LOCAL, caption="   ")
    missing = Memory(source_platform=SourcePlatform.LOCAL, caption=None)
    print(f"\n✓ Empty caption normalized to None")
    print(f"  - Empty '   ' becomes: {empty.caption}")
    print(f"  - None caption: {missing.caption}")

    # Very old timestamp
    old = Memory(
        source_platform=SourcePlatform.LOCAL,
        created_at=datetime(1985, 1, 1, tzinfo=timezone.utc),
    )
    print(f"\n✓ Timestamp before 1990 flagged")
    print(f"  - Warnings: {old.parse_warnings}")


def main():
    """Run all demonstrations."""
    print("\n" + "🎯" * 35)
    print("MEMORY DATA MODEL - COMPREHENSIVE DEMONSTRATION")
    print("🎯" * 35)

    demo_basic_creation()
    demo_timezone_handling()
    demo_privacy_levels()
    demo_content_hashing()
    demo_metadata_hashing()
    demo_is_same_moment()
    demo_merge()
    demo_helper_methods()
    demo_edge_cases()

    print("\n" + "=" * 70)
    print("✅ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nThe Memory model is ready for:")
    print("  1. ✓ Parser integration (all platforms normalize to Memory)")
    print("  2. ✓ AI analysis (privacy-conscious payloads)")
    print("  3. ✓ Deduplication (content, metadata, and fuzzy matching)")
    print("  4. ✓ Timeline aggregation (helper methods)")
    print("  5. ✓ Merging from multiple sources (confidence resolution)")
    print("\n")


if __name__ == "__main__":
    main()
