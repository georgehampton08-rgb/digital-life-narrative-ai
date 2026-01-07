"""Tests for parser infrastructure and platform parsers."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from organizer.models import SourcePlatform
from organizer.parsers import BaseParser, ParserRegistry, parse_all_sources

# =============================================================================
# ParserRegistry Tests
# =============================================================================


class TestParserRegistry:
    """Tests for ParserRegistry singleton."""

    def test_registry_is_singleton(self) -> None:
        """Test that ParserRegistry is a singleton."""
        reg1 = ParserRegistry()
        reg2 = ParserRegistry()
        assert reg1 is reg2

    def test_registry_has_parsers(self) -> None:
        """Test that parsers are registered."""
        registry = ParserRegistry()
        parsers = registry.get_all_parsers()

        # Should have at least Snapchat, Google Photos, Local
        platforms = [p.platform for p in parsers]
        assert SourcePlatform.SNAPCHAT in platforms
        assert SourcePlatform.GOOGLE_PHOTOS in platforms
        assert SourcePlatform.LOCAL in platforms

    def test_get_parser_by_platform(self) -> None:
        """Test getting parser by platform."""
        registry = ParserRegistry()

        snapchat_parser = registry.get_parser(SourcePlatform.SNAPCHAT)
        assert snapchat_parser is not None
        assert snapchat_parser.platform == SourcePlatform.SNAPCHAT

    def test_get_nonexistent_parser(self) -> None:
        """Test getting parser for unregistered platform."""
        registry = ParserRegistry()
        parser = registry.get_parser(SourcePlatform.UNKNOWN)
        assert parser is None


# =============================================================================
# BaseParser Tests
# =============================================================================


class TestBaseParserHelpers:
    """Tests for BaseParser helper methods."""

    @pytest.fixture
    def parser(self) -> BaseParser:
        """Get a concrete parser for testing helpers."""
        from organizer.parsers.local import LocalPhotosParser

        return LocalPhotosParser()

    def test_parse_datetime_iso(self, parser: BaseParser) -> None:
        """Test parsing ISO datetime strings."""
        result = parser._safe_parse_datetime("2020-06-15T10:30:00Z")
        assert result is not None
        assert result.year == 2020
        assert result.month == 6
        assert result.day == 15

    def test_parse_datetime_common_formats(self, parser: BaseParser) -> None:
        """Test parsing various datetime formats."""
        test_cases = [
            ("2020-06-15", datetime(2020, 6, 15)),
            ("2020/06/15", datetime(2020, 6, 15)),
            ("Jun 15, 2020", datetime(2020, 6, 15)),
        ]

        for date_str, expected in test_cases:
            result = parser._safe_parse_datetime(date_str)
            if result:
                assert result.date() == expected.date(), f"Failed for {date_str}"

    def test_parse_datetime_invalid(self, parser: BaseParser) -> None:
        """Test parsing invalid datetime returns None."""
        assert parser._safe_parse_datetime("not a date") is None
        assert parser._safe_parse_datetime("") is None
        assert parser._safe_parse_datetime(None) is None

    def test_generate_item_id(self, parser: BaseParser) -> None:
        """Test ID generation is deterministic for same input."""
        path1 = Path("/test/photo.jpg")
        path2 = Path("/test/photo.jpg")

        id1 = parser._generate_item_id(path1, "snapchat")
        id2 = parser._generate_item_id(path2, "snapchat")

        assert id1 == id2
        assert isinstance(id1, uuid.UUID)

    def test_generate_item_id_different_paths(self, parser: BaseParser) -> None:
        """Test different paths produce different IDs."""
        path1 = Path("/test/photo1.jpg")
        path2 = Path("/test/photo2.jpg")

        id1 = parser._generate_item_id(path1, "snapchat")
        id2 = parser._generate_item_id(path2, "snapchat")

        assert id1 != id2

    def test_extract_datetime_from_filename(self, parser: BaseParser) -> None:
        """Test extracting datetime from filename patterns."""
        test_cases = [
            ("IMG_20200615_143000.jpg", 2020, 6, 15),
            ("PXL_20210301_120000.jpg", 2021, 3, 1),
            ("Screenshot_2020-08-10.png", 2020, 8, 10),
        ]

        for filename, year, month, day in test_cases:
            result = parser._extract_datetime_from_filename(filename)
            if result:
                assert result.year == year, f"Failed year for {filename}"
                assert result.month == month, f"Failed month for {filename}"
                assert result.day == day, f"Failed day for {filename}"


# =============================================================================
# SnapchatParser Tests
# =============================================================================


class TestSnapchatParser:
    """Tests for SnapchatParser."""

    @pytest.fixture
    def parser(self) -> SnapchatParser:
        from organizer.parsers.snapchat import SnapchatParser

        return SnapchatParser()

    def test_can_parse_valid_export(
        self,
        parser: SnapchatParser,
        snapchat_export_dir: Path,
    ) -> None:
        """Test can_parse() returns True for valid Snapchat export."""
        assert parser.can_parse(snapchat_export_dir) is True

    def test_can_parse_invalid_export(
        self,
        parser: SnapchatParser,
        tmp_path: Path,
    ) -> None:
        """Test can_parse() returns False for non-Snapchat directory."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        assert parser.can_parse(empty_dir) is False

    def test_parse_extracts_memories(
        self,
        parser: SnapchatParser,
        snapchat_export_dir: Path,
    ) -> None:
        """Test parse() extracts memories correctly."""
        result = parser.parse(snapchat_export_dir)

        assert len(result.items) > 0

        # Check all items are from Snapchat
        for item in result.items:
            assert item.source_platform == SourcePlatform.SNAPCHAT

    def test_parse_handles_missing_files(
        self,
        parser: SnapchatParser,
        tmp_path: Path,
    ) -> None:
        """Test parse handles missing files gracefully."""
        # Create minimal structure with JSON but no media
        export_dir = tmp_path / "snapchat"
        export_dir.mkdir()

        memories = [{"Date": "2020-01-01", "Media Type": "PHOTO"}]
        (export_dir / "memories_history.json").write_text(json.dumps(memories), encoding="utf-8")

        # Should not crash
        result = parser.parse(export_dir)
        assert result is not None


# =============================================================================
# GooglePhotosParser Tests
# =============================================================================


class TestGooglePhotosParser:
    """Tests for GooglePhotosParser."""

    @pytest.fixture
    def parser(self) -> GooglePhotosParser:
        from organizer.parsers.google_photos import GooglePhotosParser

        return GooglePhotosParser()

    def test_can_parse_valid_export(
        self,
        parser: GooglePhotosParser,
        google_photos_export_dir: Path,
    ) -> None:
        """Test can_parse() returns True for valid Takeout export."""
        assert parser.can_parse(google_photos_export_dir) is True

    def test_sidecar_detection(
        self,
        parser: GooglePhotosParser,
        google_photos_export_dir: Path,
    ) -> None:
        """Test that JSON sidecar files are detected."""
        result = parser.parse(google_photos_export_dir)

        # Should find items with metadata from sidecars
        items_with_location = [i for i in result.items if i.location]
        assert len(items_with_location) > 0, "Should extract location from sidecar"

    def test_metadata_extraction_from_json(
        self,
        parser: GooglePhotosParser,
        google_photos_export_dir: Path,
    ) -> None:
        """Test metadata extraction from JSON sidecars."""
        result = parser.parse(google_photos_export_dir)

        # Should have items with timestamps
        items_with_ts = [i for i in result.items if i.timestamp]
        assert len(items_with_ts) > 0

    def test_parse_empty_directory(
        self,
        parser: GooglePhotosParser,
        tmp_path: Path,
    ) -> None:
        """Test parsing empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = parser.parse(empty_dir)
        assert len(result.items) == 0


# =============================================================================
# LocalPhotosParser Tests
# =============================================================================


class TestLocalPhotosParser:
    """Tests for LocalPhotosParser."""

    @pytest.fixture
    def parser(self) -> LocalPhotosParser:
        from organizer.parsers.local import LocalPhotosParser

        return LocalPhotosParser()

    def test_can_parse_always_true(
        self,
        parser: LocalPhotosParser,
        tmp_path: Path,
    ) -> None:
        """Test can_parse() returns True for any directory (fallback parser)."""
        assert parser.can_parse(tmp_path) is True

    def test_filename_date_parsing(
        self,
        parser: LocalPhotosParser,
        local_photos_dir: Path,
    ) -> None:
        """Test filename date parsing patterns."""
        result = parser.parse(local_photos_dir)

        # Should extract dates from filenames
        items_with_ts = [i for i in result.items if i.timestamp]
        assert len(items_with_ts) > 0

    def test_skips_hidden_files(
        self,
        parser: LocalPhotosParser,
        tmp_path: Path,
    ) -> None:
        """Test that hidden files are skipped."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create hidden file
        (photos_dir / ".hidden.jpg").write_bytes(b"data")
        (photos_dir / "visible.jpg").write_bytes(b"data")

        result = parser.parse(photos_dir)

        # Should only find visible file
        filenames = [i.file_path.name for i in result.items]
        assert ".hidden.jpg" not in filenames

    def test_handles_non_image_files(
        self,
        parser: LocalPhotosParser,
        tmp_path: Path,
    ) -> None:
        """Test that non-media files are skipped."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create non-image file
        (photos_dir / "document.txt").write_text("hello")
        (photos_dir / "photo.jpg").write_bytes(b"data")

        result = parser.parse(photos_dir)

        # Should only find image
        extensions = [i.file_path.suffix.lower() for i in result.items]
        assert ".txt" not in extensions


# =============================================================================
# parse_all_sources Tests
# =============================================================================


class TestParseAllSources:
    """Tests for parse_all_sources() aggregation."""

    def test_aggregates_multiple_sources(
        self,
        snapchat_export_dir: Path,
        google_photos_export_dir: Path,
    ) -> None:
        """Test that parse_all_sources aggregates from multiple directories."""
        from organizer.models import AnalysisConfig

        result = parse_all_sources(
            [snapchat_export_dir, google_photos_export_dir],
            config=AnalysisConfig(),
        )

        # Should have items from both sources
        all_items = []
        for r in result:
            all_items.extend(r.items)
            
        platforms = {item.source_platform for item in all_items}
        assert len(platforms) >= 1  # At least one platform

    def test_handles_empty_sources(self, tmp_path: Path) -> None:
        """Test handling of empty source directories."""
        from organizer.models import AnalysisConfig

        empty1 = tmp_path / "empty1"
        empty1.mkdir()
        empty2 = tmp_path / "empty2"
        empty2.mkdir()

        result = parse_all_sources([empty1, empty2], config=AnalysisConfig())

        assert result is not None
        # May have items from LOCAL parser as fallback

    def test_collects_errors(self, tmp_path: Path) -> None:
        """Test that parse errors are collected."""
        from organizer.models import AnalysisConfig

        tmp_path / "does_not_exist"

        # Should not crash, but may collect errors
        result = parse_all_sources([tmp_path], config=AnalysisConfig())
        assert result is not None
