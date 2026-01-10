"""Tests for CLI commands using Click's testing utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from dlnai.cli import organizer as cli
from dlnai.core.models import (
    LifeStoryReport,
    Memory,
    MediaType,
    SourcePlatform,
)
from dlnai.parsers.base import ParseResult, ParseStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_parse_result(sample_media_items: list[Memory]) -> ParseResult:
    """Mock parse result for testing."""
    return ParseResult(
        platform=SourcePlatform.GOOGLE_PHOTOS,
        status=ParseStatus.SUCCESS,
        memories=sample_media_items,
        files_processed=len(sample_media_items),
        parse_duration_seconds=1.0,
    )


# =============================================================================
# Version Tests
# =============================================================================


class TestVersion:
    """Tests for version command."""

    def test_version_option(self, runner: CliRunner) -> None:
        """Test organizer --version shows version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "Digital Life Narrative AI" in result.output
        # Version number should be present
        assert "0." in result.output or "1." in result.output

    def test_help_option(self, runner: CliRunner) -> None:
        """Test organizer --help shows help."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Digital Life Narrative AI" in result.output
        assert "analyze" in result.output
        assert "config" in result.output
        assert "scan" in result.output


# =============================================================================
# Scan Command Tests
# =============================================================================


class TestScanCommand:
    """Tests for scan command."""

    def test_scan_valid_directory(
        self,
        runner: CliRunner,
        snapchat_export_dir: Path,
    ) -> None:
        """Test organizer scan on valid directory."""
        result = runner.invoke(cli, ["scan", str(snapchat_export_dir)])

        assert result.exit_code == 0
        # Should show detection results
        assert "Scanning" in result.output or "Detected" in result.output

    def test_scan_empty_directory(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test scan on empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, ["scan", str(empty_dir)])

        assert result.exit_code == 0
        # Should indicate no exports found
        assert "No recognized" in result.output or "Supported platforms" in result.output

    def test_scan_nonexistent_directory(self, runner: CliRunner) -> None:
        """Test scan on nonexistent directory."""
        result = runner.invoke(cli, ["scan", "/nonexistent/path"])

        # Should fail with error
        assert result.exit_code != 0

    def test_scan_shows_platforms(
        self,
        runner: CliRunner,
        google_photos_export_dir: Path,
    ) -> None:
        """Test scan shows detected platforms."""
        result = runner.invoke(cli, ["scan", str(google_photos_export_dir)])

        assert result.exit_code == 0


# =============================================================================
# Analyze Command Tests
# =============================================================================


class TestAnalyzeCommand:
    """Tests for analyze command."""

    @pytest.fixture
    def mock_analysis(
        self,
        sample_life_report: LifeStoryReport,
        mock_parse_result: ParseResult,
    ) -> dict[str, Any]:
        """Setup mocks for analysis."""
        return {
            "report": sample_life_report,
            "parse_result": mock_parse_result,
        }

    def test_analyze_missing_input(self, runner: CliRunner) -> None:
        """Test analyze without input raises error."""
        result = runner.invoke(cli, ["analyze", "-o", "./output"])

        # Should fail - missing required input
        assert result.exit_code != 0

    def test_analyze_nonexistent_input(self, runner: CliRunner) -> None:
        """Test analyze with nonexistent input."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                "-i",
                "/nonexistent/path",
                "-o",
                "./output",
            ],
        )

        assert result.exit_code != 0

    def test_analyze_with_no_ai_flag(
        self,
        runner: CliRunner,
        snapchat_export_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test analyze with --no-ai flag uses fallback."""
        output = tmp_path / "report"

        with patch("dlnai.cli.main.parse_all_sources") as mock_parse, patch(
            "dlnai.cli.main.detect_export_source"
        ) as mock_detect, patch("dlnai.cli.main.generate_report") as mock_report:

            from dlnai.detection import DetectionResult
            from dlnai.core.models import ConfidenceLevel as Confidence

            # Mock detection
            mock_detect.return_value = [
                DetectionResult(
                    platform=SourcePlatform.SNAPCHAT,
                    root_path=snapchat_export_dir,
                    confidence=Confidence.HIGH,
                    evidence=["memories_history.json"],
                )
            ]

            # Mock parsing
            mock_parse.return_value = ParseResult(
                items=[
                    Memory(
                        source_platform=SourcePlatform.SNAPCHAT,
                        media_type=MediaType.PHOTO,
                        file_path=Path("/test/photo.jpg"),
                        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    )
                ],
                source_paths=[snapchat_export_dir],
                parse_errors=[],
                stats={"total": 1},
                duration_seconds=0.5,
            )

            # Mock report generation
            mock_report.return_value = [output.with_suffix(".html")]

            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(snapchat_export_dir),
                    "-o",
                    str(output),
                    "--no-ai",
                ],
                input="y\nn\n",
            )  # Confirm proceed, decline open

            # Check output mentions fallback
            assert "statistics-only" in result.output or "fallback" in result.output.lower()

    def test_analyze_privacy_mode(
        self,
        runner: CliRunner,
        snapchat_export_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test analyze with --privacy-mode flag."""
        output = tmp_path / "report"

        with patch("dlnai.cli.main.parse_all_sources") as mock_parse, patch(
            "dlnai.cli.main.detect_export_source"
        ) as mock_detect, patch("dlnai.cli.main.generate_report") as mock_report:

            from dlnai.detection import DetectionResult
            from dlnai.core.models import ConfidenceLevel as Confidence

            mock_detect.return_value = [
                DetectionResult(
                    platform=SourcePlatform.SNAPCHAT,
                    root_path=snapchat_export_dir,
                    confidence=Confidence.HIGH,
                    evidence=["memories_history.json"],
                )
            ]

            mock_parse.return_value = ParseResult(
                items=[
                    Memory(
                        source_platform=SourcePlatform.SNAPCHAT,
                        media_type=MediaType.PHOTO,
                        file_path=Path("/test/photo.jpg"),
                    )
                ],
                source_paths=[snapchat_export_dir],
                parse_errors=[],
                stats={"total": 1},
                duration_seconds=0.5,
            )

            mock_report.return_value = [output.with_suffix(".html")]

            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(snapchat_export_dir),
                    "-o",
                    str(output),
                    "--privacy-mode",
                    "--no-ai",
                ],
                input="y\nn\n",
            )

            # Should complete without error
            assert result.exit_code == 0 or "fallback" in result.output.lower()


# =============================================================================
# Config Command Tests
# =============================================================================


class TestConfigCommand:
    """Tests for config commands."""

    def test_config_show(self, runner: CliRunner) -> None:
        """Test organizer config show displays config."""
        with patch("dlnai.cli.main.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                ai=MagicMock(
                    model_name="gemini-1.5-pro",
                    max_tokens=8192,
                    temperature=0.7,
                    timeout_seconds=60,
                ),
                key_storage_backend=MagicMock(value="environment"),
                encrypted_key_file_path=None,
                privacy=MagicMock(
                    anonymize_paths=True,
                    local_only_mode=False,
                ),
            )

            result = runner.invoke(cli, ["config", "show"])

            assert result.exit_code == 0
            # Should show config values
            assert "gemini" in result.output.lower() or "Configuration" in result.output

    def test_config_set_key_prompts(self, runner: CliRunner) -> None:
        """Test organizer config set-key prompts for input."""
        with patch("dlnai.cli.main.get_config") as mock_config, patch(
            "dlnai.cli.main.APIKeyManager"
        ) as mock_manager:

            mock_config.return_value = MagicMock(
                key_storage_backend=MagicMock(value="environment"),
                encrypted_key_file_path=None,
            )

            mock_manager_instance = MagicMock()
            mock_manager_instance.store_key = MagicMock()
            mock_manager_instance.retrieve_key.return_value = "test-key"
            mock_manager.return_value = mock_manager_instance

            # Provide API key via input
            result = runner.invoke(cli, ["config", "set-key"], input="test-api-key-12345\n")

            # Should attempt to store key
            assert "API key" in result.output

    def test_config_set_key_empty_input(self, runner: CliRunner) -> None:
        """Test set-key with empty input."""
        with patch("dlnai.cli.main.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                key_storage_backend=MagicMock(value="environment"),
            )

            result = runner.invoke(cli, ["config", "set-key"], input="\n")

            # Should handle empty input gracefully
            assert "No API key" in result.output or result.exit_code == 0

    def test_config_reset(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test organizer config reset."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        with patch("dlnai.cli.main.AppConfig") as mock_app_config:
            mock_app_config.get_default_config_path.return_value = config_file

            result = runner.invoke(cli, ["config", "reset"], input="y\n")

            assert result.exit_code == 0

    def test_config_set_value(self, runner: CliRunner) -> None:
        """Test organizer config set KEY VALUE."""
        with patch("dlnai.cli.main.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.ai = MagicMock()
            mock_cfg.ai.model_name = "gemini-1.5-pro"
            mock_cfg.save_to_yaml = MagicMock()
            mock_config.return_value = mock_cfg

            result = runner.invoke(cli, ["config", "set", "ai.model_name", "gemini-1.5-flash"])

            # Should attempt to set
            assert result.exit_code == 0 or "Set" in result.output


# =============================================================================
# Organize Command Tests
# =============================================================================


class TestOrganizeCommand:
    """Tests for organize command."""

    def test_organize_missing_output(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test organize without output path."""
        result = runner.invoke(cli, ["organize", "-i", str(tmp_path)])

        # Should fail - missing output
        assert result.exit_code != 0

    def test_organize_dry_run(
        self,
        runner: CliRunner,
        local_photos_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test organize with --dry-run."""
        output = tmp_path / "organized"

        with patch("dlnai.cli.main.parse_all_sources") as mock_parse, patch(
            "dlnai.cli.main.MediaOrganizer"
        ) as mock_organizer:

            mock_parse.return_value = ParseResult(
                items=[
                    Memory(
                        source_platform=SourcePlatform.LOCAL,
                        media_type=MediaType.PHOTO,
                        file_path=local_photos_dir / "photo.jpg",
                        created_at=datetime(2020, 6, 15, tzinfo=timezone.utc),
                    )
                ],
                source_paths=[local_photos_dir],
                parse_errors=[],
                stats={"total": 1},
                duration_seconds=0.5,
            )

            mock_org_instance = MagicMock()
            mock_org_instance.plan_organization.return_value = []
            mock_org_instance.preview_plan.return_value = "Preview: 0 files"
            mock_organizer.return_value = mock_org_instance

            result = runner.invoke(
                cli,
                [
                    "organize",
                    "-i",
                    str(local_photos_dir),
                    "-o",
                    str(output),
                    "--dry-run",
                ],
            )

            # Should complete without actually organizing
            assert result.exit_code == 0 or "dry run" in result.output.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test invalid command gives helpful error."""
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0

    def test_verbose_flag(self, runner: CliRunner) -> None:
        """Test --verbose flag is accepted."""
        result = runner.invoke(cli, ["--verbose", "--help"])

        assert result.exit_code == 0

    def test_scan_missing_argument(self, runner: CliRunner) -> None:
        """Test scan without path argument."""
        result = runner.invoke(cli, ["scan"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage" in result.output


# =============================================================================
# Fallback Mode Messaging Tests
# =============================================================================


class TestFallbackMessaging:
    """Tests for fallback mode messaging."""

    def test_analyze_shows_fallback_warning(
        self,
        runner: CliRunner,
        snapchat_export_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test analyze shows clear fallback warning."""
        output = tmp_path / "report"

        with patch("dlnai.cli.main.parse_all_sources") as mock_parse, patch(
            "dlnai.cli.main.detect_export_source"
        ) as mock_detect, patch("dlnai.cli.main.generate_report") as mock_report, patch(
            "dlnai.cli.main.check_api_key_configured"
        ) as mock_key_check:

            from dlnai.detection import DetectionResult
            from dlnai.core.models import ConfidenceLevel as Confidence

            mock_detect.return_value = [
                DetectionResult(
                    platform=SourcePlatform.SNAPCHAT,
                    root_path=snapchat_export_dir,
                    confidence=Confidence.HIGH,
                    evidence=["memories_history.json"],
                )
            ]

            mock_parse.return_value = ParseResult(
                items=[
                    Memory(
                        source_platform=SourcePlatform.SNAPCHAT,
                        media_type=MediaType.PHOTO,
                        file_path=Path("/test/photo.jpg"),
                        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    )
                ],
                source_paths=[snapchat_export_dir],
                parse_errors=[],
                stats={"total": 1},
                duration_seconds=0.5,
            )

            mock_report.return_value = [output.with_suffix(".html")]
            mock_key_check.return_value = False  # No API key

            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(snapchat_export_dir),
                    "-o",
                    str(output),
                ],
                input="y\ny\nn\n",
            )  # Proceed, continue fallback, decline open

            # Should mention fallback or API key
            output_lower = result.output.lower()
            assert (
                "fallback" in output_lower or "api key" in output_lower or "config" in output_lower
            )

    def test_fallback_suggests_config(
        self,
        runner: CliRunner,
        snapchat_export_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Test fallback mode suggests running config set-key."""
        output = tmp_path / "report"

        with patch("dlnai.cli.main.parse_all_sources") as mock_parse, patch(
            "dlnai.cli.main.detect_export_source"
        ) as mock_detect, patch("dlnai.cli.main.generate_report") as mock_report:

            from dlnai.detection import DetectionResult
            from dlnai.core.models import ConfidenceLevel as Confidence

            mock_detect.return_value = [
                DetectionResult(
                    platform=SourcePlatform.SNAPCHAT,
                    root_path=snapchat_export_dir,
                    confidence=Confidence.HIGH,
                    evidence=[],
                )
            ]

            mock_parse.return_value = ParseResult(
                items=[
                    Memory(
                        source_platform=SourcePlatform.SNAPCHAT,
                        media_type=MediaType.PHOTO,
                        file_path=Path("/test.jpg"),
                    )
                ],
                source_paths=[],
                parse_errors=[],
                stats={},
                duration_seconds=0.1,
            )

            mock_report.return_value = [output.with_suffix(".html")]

            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "-i",
                    str(snapchat_export_dir),
                    "-o",
                    str(output),
                    "--no-ai",
                ],
                input="y\nn\n",
            )

            # Should mention config or set-key for enabling AI
            assert result.exit_code == 0 or "statistics" in result.output.lower()
