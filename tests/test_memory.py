"""Tests for the Memory data model.

Tests cover:
- Memory creation with all fields
- Memory creation with minimal fields
- Validation rejection of invalid coordinates
- to_ai_payload() output at each privacy level
- content_hash computation
- is_same_moment() with various tolerances
- merge_with() confidence resolution
- Timezone handling for naive vs aware datetimes
- Edge cases: None created_at, empty vs missing caption
"""

import hashlib
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.core.memory import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_media_type_values(self):
        """Test MediaType enum has expected values."""
        assert MediaType.PHOTO.value == "photo"
        assert MediaType.SCREENSHOT.value == "screenshot"
        assert MediaType.LIVE_PHOTO.value == "live_photo"
    
    def test_source_platform_values(self):
        """Test SourcePlatform enum has expected values."""
        assert SourcePlatform.SNAPCHAT.value == "snapchat"
        assert SourcePlatform.GOOGLE_PHOTOS.value == "google_photos"
        assert SourcePlatform.LOCAL.value == "local"
    
    def test_confidence_level_values(self):
        """Test ConfidenceLevel enum has expected values."""
        assert ConfidenceLevel.VERIFIED.value == "verified"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.INFERRED.value == "inferred"


class TestGeoPoint:
    """Test GeoPoint model."""
    
    def test_valid_coordinates(self):
        """Test creation with valid coordinates."""
        point = GeoPoint(latitude=40.7829, longitude=-73.9654)
        assert point.latitude == 40.7829
        assert point.longitude == -73.9654
    
    def test_invalid_latitude_too_high(self):
        """Test validation rejects latitude > 90."""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            GeoPoint(latitude=91.0, longitude=0.0)
    
    def test_invalid_latitude_too_low(self):
        """Test validation rejects latitude < -90."""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            GeoPoint(latitude=-91.0, longitude=0.0)
    
    def test_invalid_longitude_too_high(self):
        """Test validation rejects longitude > 180."""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            GeoPoint(latitude=0.0, longitude=181.0)
    
    def test_invalid_longitude_too_low(self):
        """Test validation rejects longitude < -180."""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            GeoPoint(latitude=0.0, longitude=-181.0)
    
    def test_distance_calculation(self):
        """Test Haversine distance calculation."""
        nyc = GeoPoint(latitude=40.7829, longitude=-73.9654)
        la = GeoPoint(latitude=34.0522, longitude=-118.2437)
        
        distance = nyc.distance_km(la)
        
        # NYC to LA is approximately 3,944 km
        assert 3900 < distance < 4000
    
    def test_to_approximate(self):
        """Test coordinate rounding for privacy."""
        exact = GeoPoint(latitude=40.7829123, longitude=-73.9654456)
        approx = exact.to_approximate(precision=2)
        
        assert approx.latitude == 40.78
        assert approx.longitude == -73.97
        assert approx.altitude_meters is None
        assert approx.accuracy_meters is None


class TestLocation:
    """Test Location model."""
    
    def test_is_empty_with_no_data(self):
        """Test is_empty returns True when all fields are None."""
        location = Location()
        assert location.is_empty() is True
    
    def test_is_empty_with_coordinates(self):
        """Test is_empty returns False when coordinates present."""
        location = Location(coordinates=GeoPoint(latitude=40.0, longitude=-73.0))
        assert location.is_empty() is False
    
    def test_to_display_string_hierarchical(self):
        """Test human-readable location string."""
        location = Location(
            place_name="Central Park",
            locality="Manhattan",
            region="New York",
            country="United States",
        )
        
        display = location.to_display_string()
        assert "Central Park" in display
        assert "Manhattan" in display
        assert "New York" in display
    
    def test_to_ai_summary_privacy(self):
        """Test condensed location for AI (privacy-conscious)."""
        location = Location(
            place_name="Central Park",
            locality="Manhattan",
            region="New York",
            country_code="US",
        )
        
        summary = location.to_ai_summary()
        # Should prefer region + country code
        assert "New York, US" == summary


class TestPersonTag:
    """Test PersonTag model."""
    
    def test_auto_normalize_name(self):
        """Test normalized_name auto-generated from name."""
        person = PersonTag(name="Alice Smith")
        assert person.normalized_name == "alice smith"
    
    def test_to_anonymous(self):
        """Test anonymization hashes name consistently."""
        person = PersonTag(name="Alice Smith")
        anon = person.to_anonymous()
        
        assert anon.name.startswith("person_")
        assert len(anon.name) == 15  # "person_" + 8 hex chars
        assert anon.platform_id is None
        
        # Same name should produce same hash
        person2 = PersonTag(name="Alice Smith")
        anon2 = person2.to_anonymous()
        assert anon.name == anon2.name


class TestMemory:
    """Test Memory model."""
    
    def test_create_minimal_memory(self):
        """Test creation with only required fields."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
        )
        
        assert memory.source_platform == SourcePlatform.LOCAL
        assert memory.media_type == MediaType.PHOTO
        assert memory.id is not None  # Auto-generated
        assert memory.created_at is None  # Optional
    
    def test_create_full_memory(self):
        """Test creation with all fields populated."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
            created_at_confidence=ConfidenceLevel.VERIFIED,
            location=Location(
                coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
                place_name="Central Park",
                locality="Manhattan",
                region="New York",
            ),
            people=[PersonTag(name="Alice"), PersonTag(name="Bob")],
            caption="Summer afternoon",
            width=1920,
            height=1080,
        )
        
        assert memory.created_at.year == 2020
        assert len(memory.people) == 2
        assert memory.caption == "Summer afternoon"
    
    def test_id_auto_generation(self):
        """Test ID is auto-generated when not provided."""
        memory1 = Memory(source_platform=SourcePlatform.LOCAL)
        memory2 = Memory(source_platform=SourcePlatform.LOCAL)
        
        assert memory1.id != memory2.id
        assert len(memory1.id) == 36  # UUID format
    
    def test_timezone_aware_datetime(self):
        """Test timezone-aware datetime is preserved."""
        aware_dt = datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc)
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=aware_dt,
        )
        
        assert memory.created_at.tzinfo is not None
        assert "Naive datetime" not in str(memory.parse_warnings)
    
    def test_naive_datetime_converted_to_utc(self):
        """Test naive datetime is converted to UTC with warning."""
        naive_dt = datetime(2020, 7, 15, 12, 30)
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=naive_dt,
        )
        
        assert memory.created_at.tzinfo is not None
        assert "Naive datetime converted to UTC" in memory.parse_warnings
    
    def test_timestamp_before_1990_warning(self):
        """Test warning for timestamps before 1990."""
        old_dt = datetime(1985, 1, 1, tzinfo=timezone.utc)
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=old_dt,
        )
        
        assert "Timestamp before 1990" in memory.parse_warnings
    
    def test_caption_normalization_strips_whitespace(self):
        """Test caption whitespace is stripped."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            caption="  Test caption  ",
        )
        
        assert memory.caption == "Test caption"
    
    def test_caption_empty_string_becomes_none(self):
        """Test empty caption becomes None."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            caption="   ",
        )
        
        assert memory.caption is None
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            memory = Memory(source_platform=SourcePlatform.LOCAL)
            hash1 = memory.compute_content_hash(temp_path)
            
            # Hash should be consistent
            memory2 = Memory(source_platform=SourcePlatform.LOCAL)
            hash2 = memory2.compute_content_hash(temp_path)
            
            assert hash1 == hash2
            assert memory.content_hash == hash1
            assert len(hash1) == 32  # MD5 hex digest
        finally:
            temp_path.unlink()
    
    def test_compute_metadata_hash(self):
        """Test metadata hash for fuzzy deduplication."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 45, tzinfo=timezone.utc),
        )
        
        hash1 = memory.compute_metadata_hash()
        
        # Same metadata (to minute precision) should produce same hash
        memory2 = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 59, tzinfo=timezone.utc),
        )
        
        hash2 = memory2.compute_metadata_hash()
        
        assert hash1 == hash2
    
    def test_to_ai_payload_strict(self):
        """Test AI payload with strict privacy level."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
            source_path=Path("/secret/path/photo.jpg"),
            location=Location(
                coordinates=GeoPoint(latitude=40.7829, longitude=-73.9654),
                country="United States",
            ),
            people=[PersonTag(name="Alice Smith")],
            caption="Secret diary entry",
        )
        
        payload = memory.to_ai_payload(privacy_level="strict")
        
        # Should only have minimal data
        assert payload["platform"] == "google_photos"
        assert payload["media_type"] == "photo"
        assert payload["created_at"] == "2020-07"  # Month only
        
        # Should NOT have sensitive data
        assert "source_path" not in payload
        assert "caption" not in payload
        assert "people" not in payload
        assert "location" not in payload
    
    def test_to_ai_payload_standard(self):
        """Test AI payload with standard privacy level."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
            location=Location(country="United States", country_code="US"),
            people=[PersonTag(name="Alice"), PersonTag(name="Bob")],
            caption="Test caption",
        )
        
        payload = memory.to_ai_payload(privacy_level="standard")
        
        assert payload["created_at"] == "2020-07-15"  # Full date
        assert payload["location_country"] == "US"
        assert payload["people_count"] == 2
        assert payload["has_caption"] is True
        
        # Should NOT have full caption or names
        assert "caption" not in payload
        assert "people" not in payload
    
    def test_to_ai_payload_detailed(self):
        """Test AI payload with detailed privacy level."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
            location=Location(
                locality="Manhattan",
                region="New York",
                country_code="US",
            ),
            people=[PersonTag(name="Alice Smith")],
            caption="A" * 200,  # Long caption
            width=1920,
            height=1080,
        )
        
        payload = memory.to_ai_payload(privacy_level="detailed")
        
        assert payload["location"] == "New York, US"
        assert len(payload["caption"]) <= 103  # Truncated to 100 + "..."
        assert payload["caption"].endswith("...")
        assert payload["people"][0].startswith("person_")  # Anonymized
        assert payload["dimensions"] == "1920x1080"
    
    def test_to_ai_payload_never_includes_sensitive_data(self):
        """Test that sensitive data is NEVER in AI payload."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            source_path=Path("/users/john/secret/photo.jpg"),
            location=Location(
                coordinates=GeoPoint(latitude=40.7829123, longitude=-73.9654456)
            ),
            original_metadata={"secret": "data"},
            content_hash="abc123",
        )
        
        for privacy_level in ["strict", "standard", "detailed"]:
            payload = memory.to_ai_payload(privacy_level=privacy_level)
            
            assert "source_path" not in payload
            assert "original_metadata" not in payload
            assert "content_hash" not in payload
            assert "coordinates" not in payload
            # Should not have exact coordinates (only place names)
    
    def test_is_same_moment_within_tolerance(self):
        """Test is_same_moment returns True within tolerance."""
        memory1 = Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
        )
        
        memory2 = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 45, tzinfo=timezone.utc),
        )
        
        assert memory1.is_same_moment(memory2, tolerance_seconds=60)
    
    def test_is_same_moment_outside_tolerance(self):
        """Test is_same_moment returns False outside tolerance."""
        memory1 = Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
        )
        
        memory2 = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 32, 0, tzinfo=timezone.utc),
        )
        
        assert not memory1.is_same_moment(memory2, tolerance_seconds=60)
    
    def test_is_same_moment_different_media_type(self):
        """Test is_same_moment returns False for different media types."""
        memory1 = Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
        )
        
        memory2 = Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.VIDEO,
            created_at=datetime(2020, 7, 15, 12, 30, 10, tzinfo=timezone.utc),
        )
        
        assert not memory1.is_same_moment(memory2, tolerance_seconds=60)
    
    def test_merge_with_prefers_higher_confidence(self):
        """Test merge_with prefers higher confidence data."""
        memory1 = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            created_at=datetime(2020, 7, 15, 12, 30, 0, tzinfo=timezone.utc),
            created_at_confidence=ConfidenceLevel.MEDIUM,
        )
        
        memory2 = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=datetime(2020, 7, 15, 12, 35, 0, tzinfo=timezone.utc),
            created_at_confidence=ConfidenceLevel.VERIFIED,
        )
        
        merged = memory1.merge_with(memory2)
        
        # Should prefer VERIFIED timestamp
        assert merged.created_at == memory2.created_at
        assert merged.created_at_confidence == ConfidenceLevel.VERIFIED
    
    def test_merge_with_combines_people(self):
        """Test merge_with combines and deduplicates people."""
        memory1 = Memory(
            source_platform=SourcePlatform.FACEBOOK,
            people=[
                PersonTag(name="Alice Smith"),
                PersonTag(name="Bob Jones"),
            ],
        )
        
        memory2 = Memory(
            source_platform=SourcePlatform.INSTAGRAM,
            people=[
                PersonTag(name="Alice Smith"),  # Duplicate
                PersonTag(name="Charlie Brown"),
            ],
        )
        
        merged = memory1.merge_with(memory2)
        
        # Should have 3 unique people
        assert len(merged.people) == 3
        names = {p.normalized_name for p in merged.people}
        assert names == {"alice smith", "bob jones", "charlie brown"}
    
    def test_get_year_month(self):
        """Test get_year_month returns correct format."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=datetime(2020, 7, 15, 12, 30, tzinfo=timezone.utc),
        )
        
        assert memory.get_year_month() == "2020-07"
    
    def test_get_year_month_no_timestamp(self):
        """Test get_year_month returns None when no timestamp."""
        memory = Memory(source_platform=SourcePlatform.LOCAL)
        assert memory.get_year_month() is None
    
    def test_days_since(self):
        """Test days_since calculation."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            created_at=datetime(2020, 7, 15, tzinfo=timezone.utc),
        )
        
        reference = date(2020, 7, 20)
        assert memory.days_since(reference) == 5
        
        reference = date(2020, 7, 10)
        assert memory.days_since(reference) == -5
    
    def test_from_dict_basic(self):
        """Test from_dict creates Memory from dict."""
        data = {
            "source_platform": "google_photos",
            "media_type": "photo",
            "caption": "Test",
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.source_platform == SourcePlatform.GOOGLE_PHOTOS
        assert memory.media_type == MediaType.PHOTO
        assert memory.caption == "Test"
    
    def test_from_dict_with_nested_location(self):
        """Test from_dict handles nested Location object."""
        data = {
            "source_platform": "google_photos",
            "media_type": "photo",
            "location": {
                "coordinates": {"latitude": 40.7829, "longitude": -73.9654},
                "place_name": "Central Park",
            },
        }
        
        memory = Memory.from_dict(data)
        
        assert memory.location is not None
        assert memory.location.place_name == "Central Park"
        assert memory.location.coordinates.latitude == 40.7829
    
    def test_from_dict_with_people_strings(self):
        """Test from_dict handles legacy people as strings."""
        data = {
            "source_platform": "facebook",
            "media_type": "photo",
            "people": ["Alice", "Bob"],
        }
        
        memory = Memory.from_dict(data)
        
        assert len(memory.people) == 2
        assert memory.people[0].name == "Alice"
        assert memory.people[1].name == "Bob"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_memory_with_none_created_at(self):
        """Test Memory with no timestamp is valid."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            media_type=MediaType.PHOTO,
            created_at=None,
        )
        
        assert memory.created_at is None
        assert memory.get_year() is None
        assert memory.get_month() is None
    
    def test_coordinates_near_zero_warning(self):
        """Test warning for coordinates at (0, 0) - likely error."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            location=Location(coordinates=GeoPoint(latitude=0.0, longitude=0.0)),
        )
        
        assert "Coordinates near (0,0)" in memory.parse_warnings
    
    def test_invalid_privacy_level_raises_error(self):
        """Test invalid privacy_level raises ValueError."""
        memory = Memory(source_platform=SourcePlatform.LOCAL)
        
        with pytest.raises(ValueError, match="Invalid privacy_level"):
            memory.to_ai_payload(privacy_level="invalid")
    
    def test_unicode_in_caption_and_names(self):
        """Test Unicode characters are handled properly."""
        memory = Memory(
            source_platform=SourcePlatform.LOCAL,
            caption="日本語 caption with émojis 🎉",
            people=[PersonTag(name="François Müller")],
        )
        
        assert "日本語" in memory.caption
        assert memory.people[0].name == "François Müller"
        
        payload = memory.to_ai_payload(privacy_level="detailed")
        # Should handle Unicode in truncation
        assert isinstance(payload.get("caption"), str)
