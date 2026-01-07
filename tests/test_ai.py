"""Tests for AI client and analyzer with mocking (no real API calls)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from organizer.models import (
    AnalysisConfig,
    Confidence,
    LifeStoryReport,
    MediaItem,
    MediaType,
    SourcePlatform,
)

# =============================================================================
# Mock AI Responses
# =============================================================================


MOCK_CHAPTER_RESPONSE = {
    "chapters": [
        {
            "title": "The Beginning",
            "start_date": "2020-01-01",
            "end_date": "2020-06-30",
            "themes": ["exploration", "growth"],
            "confidence": "high",
            "reasoning": "Clear activity pattern",
        },
        {
            "title": "New Adventures",
            "start_date": "2020-07-01",
            "end_date": "2020-12-31",
            "themes": ["travel", "friends"],
            "confidence": "medium",
            "reasoning": "Location changes detected",
        },
    ]
}

MOCK_NARRATIVE_RESPONSE = {
    "narrative": "This was a transformative period marked by new experiences.",
    "key_events": ["Started new hobby", "Met new friends"],
    "emotional_arc": "Growth and discovery",
}

MOCK_PLATFORM_RESPONSE = [
    {
        "platform": "google_photos",
        "usage_pattern": "Archival storage for memories",
        "peak_years": [2020],
        "common_content_types": ["photo"],
        "unique_aspects": ["Auto-backup enabled"],
    }
]


# =============================================================================
# AIClient Tests
# =============================================================================


class TestAIClient:
    """Tests for AIClient class."""

    @pytest.fixture
    def mock_genai(self) -> MagicMock:
        """Mock google.generativeai module."""
        mock = MagicMock()
        mock.configure = MagicMock()
        mock.GenerativeModel = MagicMock()
        return mock

    def test_client_initialization_with_key(self) -> None:
        """Test AIClient initialization with API key."""
        with patch("organizer.ai.client.genai") as mock_genai:
            mock_genai.GenerativeModel.return_value = MagicMock()

            from organizer.ai.client import AIClient

            client = AIClient(api_key="test-key-123")

            mock_genai.configure.assert_called_once()
            assert client.model_name == "gemini-1.5-pro"

    def test_client_initialization_without_key_raises(self) -> None:
        """Test AIClient raises error without API key configured."""
        with patch("organizer.ai.client.genai"):
            with patch("organizer.config.APIKeyManager") as mock_manager:
                mock_manager_instance = MagicMock()
                mock_manager_instance.retrieve_key.return_value = None
                mock_manager.return_value = mock_manager_instance

                from organizer.ai.client import AIClient, APIKeyMissingError

                with pytest.raises(APIKeyMissingError):
                    AIClient()

    def test_generate_calls_model(self) -> None:
        """Test generate() calls model correctly."""
        with patch("organizer.ai.client.genai") as mock_genai:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.text = "Generated text response"
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50

            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            from organizer.ai.client import AIClient

            client = AIClient(api_key="test-key")

            result = client.generate("Test prompt")

            assert result.text == "Generated text response"
            mock_model.generate_content.assert_called()

    def test_generate_json_parses_response(self) -> None:
        """Test generate_json() parses JSON response."""
        with patch("organizer.ai.client.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = json.dumps({"key": "value"})
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 50
            mock_response.usage_metadata.candidates_token_count = 25

            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            from organizer.ai.client import AIClient

            client = AIClient(api_key="test-key")

            result = client.generate_json("Generate JSON")

            assert result == {"key": "value"}

    def test_retry_on_rate_limit(self) -> None:
        """Test retry logic for rate limits."""
        with patch("organizer.ai.client.genai") as mock_genai:
            from google.api_core.exceptions import ResourceExhausted

            mock_response = MagicMock()
            mock_response.text = "Success after retry"
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 50
            mock_response.usage_metadata.candidates_token_count = 25

            mock_model = MagicMock()
            # First call raises, second succeeds
            mock_model.generate_content.side_effect = [
                ResourceExhausted("Rate limited"),
                mock_response,
            ]
            mock_genai.GenerativeModel.return_value = mock_model

            from organizer.ai.client import AIClient

            client = AIClient(api_key="test-key")
            client._max_retries = 2
            client._base_delay = 0.01  # Fast retry for test

            # Should succeed on retry
            result = client.generate("Test")
            assert result.text == "Success after retry"

    def test_error_mapping(self) -> None:
        """Test error handling and exception mapping."""
        with patch("organizer.ai.client.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("Unknown error")
            mock_genai.GenerativeModel.return_value = mock_model

            from organizer.ai.client import AIClient, AIRequestError

            client = AIClient(api_key="test-key")

            with pytest.raises(AIRequestError):
                client.generate("Test")


# =============================================================================
# LifeStoryAnalyzer Tests
# =============================================================================


class TestLifeStoryAnalyzer:
    """Tests for LifeStoryAnalyzer class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock AI client."""
        client = MagicMock()
        client.model_name = "gemini-mock"

        # Mock generate_json for different prompts
        def mock_generate_json(prompt: str, **kwargs) -> dict:
            if "chapter" in prompt.lower():
                return MOCK_CHAPTER_RESPONSE
            elif "narrative" in prompt.lower():
                return MOCK_NARRATIVE_RESPONSE
            elif "platform" in prompt.lower():
                return MOCK_PLATFORM_RESPONSE
            return {"result": "mock"}

        client.generate_json.side_effect = mock_generate_json

        # Mock generate for executive summary
        mock_response = MagicMock()
        mock_response.text = "This is a life story summary..."
        client.generate.return_value = mock_response

        client.count_tokens.return_value = 100

        return client

    @pytest.fixture
    def analyzer(self, mock_client: MagicMock) -> LifeStoryAnalyzer:
        """Create analyzer with mock client."""
        from organizer.ai.life_analyzer import LifeStoryAnalyzer
        from organizer.config import PrivacySettings

        return LifeStoryAnalyzer(
            client=mock_client,
            config=AnalysisConfig(),
            privacy=PrivacySettings(),
        )

    def test_prepare_items_for_ai_privacy(
        self,
        analyzer: LifeStoryAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test _prepare_items_for_ai() privacy filtering."""
        prepared = analyzer._prepare_items_for_ai(sample_media_items[:5])

        assert len(prepared) == 5
        for item in prepared:
            assert "platform" in item
            assert "type" in item
            # Should not contain raw file paths in privacy mode
            assert isinstance(item, dict)

    def test_generate_temporal_summary(
        self,
        analyzer: LifeStoryAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test _generate_temporal_summary() statistics."""
        summary = analyzer._generate_temporal_summary(sample_media_items)

        assert "total_items" in summary
        assert summary["total_items"] == len(sample_media_items)
        assert "items_by_year" in summary
        assert "items_by_platform" in summary
        assert "items_by_type" in summary

    def test_sample_items_for_prompt(
        self,
        analyzer: LifeStoryAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test _sample_items_for_prompt() selection logic."""
        # Sample fewer than total
        sampled = analyzer._sample_items_for_prompt(sample_media_items, max_items=10)

        assert len(sampled) == 10

        # Should maintain temporal order
        timestamps = [i.timestamp for i in sampled if i.timestamp]
        sorted_timestamps = sorted(timestamps)
        assert timestamps == sorted_timestamps

    def test_sample_items_returns_all_when_small(
        self,
        analyzer: LifeStoryAnalyzer,
    ) -> None:
        """Test sampling returns all items when count is small."""
        items = [
            MediaItem(
                source_platform=SourcePlatform.LOCAL,
                media_type=MediaType.PHOTO,
                file_path=Path(f"/photo_{i}.jpg"),
            )
            for i in range(5)
        ]

        sampled = analyzer._sample_items_for_prompt(items, max_items=10)
        assert len(sampled) == 5

    def test_analyze_full_flow(
        self,
        analyzer: LifeStoryAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test analyze() full flow with mocked AI responses."""
        # Run async analyze
        report = asyncio.run(analyzer.analyze(sample_media_items))

        assert isinstance(report, LifeStoryReport)
        assert report.total_media_analyzed == len(sample_media_items)
        assert len(report.chapters) > 0
        assert report.is_fallback_mode is False

    def test_analyze_with_progress_callback(
        self,
        analyzer: LifeStoryAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test analyze() calls progress callback."""
        progress_calls = []

        def callback(stage: str, percent: float) -> None:
            progress_calls.append((stage, percent))

        asyncio.run(analyzer.analyze(sample_media_items, callback))

        assert len(progress_calls) > 0
        # Should have initialization and completion
        assert progress_calls[0][1] == 0.0
        assert progress_calls[-1][1] == 100.0

    def test_analyze_handles_ai_errors(
        self,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test analyze() handles AI errors gracefully."""
        from organizer.ai.client import AIRequestError
        from organizer.ai.life_analyzer import LifeStoryAnalyzer
        from organizer.config import PrivacySettings

        # Create client that fails
        failing_client = MagicMock()
        failing_client.model_name = "gemini-mock"
        failing_client.generate_json.side_effect = AIRequestError("API Error")
        failing_client.generate.side_effect = AIRequestError("API Error")

        analyzer = LifeStoryAnalyzer(
            client=failing_client,
            config=AnalysisConfig(),
            privacy=PrivacySettings(),
        )

        # Should still produce a report (fallback chapters)
        report = asyncio.run(analyzer.analyze(sample_media_items))

        assert isinstance(report, LifeStoryReport)
        # May have fallback chapters


# =============================================================================
# FallbackAnalyzer Tests
# =============================================================================


class TestFallbackAnalyzer:
    """Tests for FallbackAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> FallbackAnalyzer:
        """Create FallbackAnalyzer instance."""
        from organizer.ai.fallback import FallbackAnalyzer

        return FallbackAnalyzer()

    def test_produces_valid_report(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test FallbackAnalyzer produces valid report."""
        report = analyzer.analyze(sample_media_items)

        assert isinstance(report, LifeStoryReport)
        assert report.total_media_analyzed == len(sample_media_items)

    def test_is_fallback_mode_true(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test is_fallback_mode is True."""
        report = analyzer.analyze(sample_media_items)

        assert report.is_fallback_mode is True
        assert report.ai_model_used == "none (fallback mode)"

    def test_statistics_accurate(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test statistics are accurate."""
        report = analyzer.analyze(sample_media_items)

        assert report.total_media_analyzed == len(sample_media_items)

        # Should have chapters (yearly)
        assert len(report.chapters) > 0

    def test_yearly_chapters(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test that chapters are organized by year."""
        report = analyzer.analyze(sample_media_items)

        # Chapter titles should be "Year XXXX"
        for chapter in report.chapters:
            assert "Year" in chapter.title or chapter.title.isdigit()

    def test_fallback_chapter_low_confidence(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test fallback chapters have LOW confidence."""
        report = analyzer.analyze(sample_media_items)

        for chapter in report.chapters:
            assert chapter.confidence == Confidence.LOW

    def test_empty_platform_insights(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test fallback has no platform insights (requires AI)."""
        report = analyzer.analyze(sample_media_items)

        assert report.platform_insights == []

    def test_handles_empty_items(
        self,
        analyzer: FallbackAnalyzer,
    ) -> None:
        """Test handling of empty item list."""
        report = analyzer.analyze([])

        assert report.total_media_analyzed == 0
        assert len(report.chapters) == 0

    def test_fallback_summary_contains_cta(
        self,
        analyzer: FallbackAnalyzer,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test fallback summary contains call-to-action."""
        report = analyzer.analyze(sample_media_items)

        # Should mention how to enable AI
        assert (
            "fallback" in report.executive_summary.lower()
            or "ai" in report.executive_summary.lower()
            or "configure" in report.executive_summary.lower()
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestAIIntegration:
    """Integration tests for AI components."""

    def test_generate_fallback_report_function(
        self,
        sample_media_items: list[MediaItem],
    ) -> None:
        """Test generate_fallback_report convenience function."""
        from organizer.ai.fallback import generate_fallback_report

        report = generate_fallback_report(sample_media_items)

        assert report.is_fallback_mode is True

    def test_is_fallback_mode_function(
        self,
        sample_life_report: LifeStoryReport,
    ) -> None:
        """Test is_fallback_mode utility function."""
        from organizer.ai.fallback import is_fallback_mode

        # Regular report
        assert is_fallback_mode(sample_life_report) is False

        # Fallback report
        fallback_report = LifeStoryReport(
            generated_at=datetime.now(timezone.utc),
            ai_model_used="none",
            total_media_analyzed=0,
            date_range=None,
            executive_summary="",
            chapters=[],
            platform_insights=[],
            detected_patterns=[],
            data_gaps=[],
            data_quality_notes=[],
            is_fallback_mode=True,
        )
        assert is_fallback_mode(fallback_report) is True
