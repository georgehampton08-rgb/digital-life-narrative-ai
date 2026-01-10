"""Tests for src.ai.client module — The Gemini AI client.

Tests cover:
- AIClient initialization and configuration
- Exception hierarchy and error mapping
- Retry logic with exponential backoff
- Consent management system
- Response parsing and structured output
- Edge cases and error handling

All tests mock the google.generativeai SDK — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import test targets - module handles missing SDK gracefully
from dlnai.ai.client import (
    AIAuthError,
    AIBadRequestError,
    # Client
    AIClient,
    # Exceptions
    AIClientError,
    AIContentBlockedError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    # Response models
    AIResponse,
    AIServerError,
    AITimeoutError,
    AITokenLimitError,
    AIUnavailableError,
    StructuredResponse,
    get_client,
    grant_consent_programmatic,
    has_consent,
    # Consent
    request_consent,
    require_ai,
    revoke_consent,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    config = MagicMock()
    config.ai.mode.value = "enabled"
    config.ai.is_enabled.return_value = True
    config.ai.narrative_model = "gemini-1.5-pro"
    config.ai.temperature = 0.7
    config.ai.max_output_tokens = 8192
    config.ai.timeout_seconds = 120
    config.ai.max_retries = 3
    config.ai.retry_base_delay = 0.01  # Fast for tests
    config.ai.require_consent = True
    return config


@pytest.fixture
def mock_disabled_config():
    """Create a mock AppConfig with AI disabled."""
    config = MagicMock()
    config.ai.mode.value = "disabled"
    config.ai.is_enabled.return_value = False
    config.ai.require_consent = True
    return config


@pytest.fixture
def mock_genai_response():
    """Create a mock Gemini API response."""
    response = MagicMock()
    response.text = "Generated response text"
    response.candidates = [MagicMock()]
    response.candidates[0].finish_reason = MagicMock()
    response.candidates[0].finish_reason.name = "STOP"
    response.usage_metadata = MagicMock()
    response.usage_metadata.prompt_token_count = 100
    response.usage_metadata.candidates_token_count = 50
    response.usage_metadata.total_token_count = 150
    response.prompt_feedback = MagicMock()
    response.prompt_feedback.block_reason = None
    return response


@pytest.fixture(autouse=True)
def reset_consent():
    """Reset consent state before each test."""
    revoke_consent()
    yield
    revoke_consent()


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Test exception hierarchy."""

    def test_ai_client_error_base(self):
        """Test AIClientError base class."""
        error = AIClientError("Test error", retriable=True)
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.retriable is True
        assert error.original_error is None

    def test_ai_unavailable_error_reasons(self):
        """Test AIUnavailableError with different reasons."""
        reasons = ["disabled", "no_key", "offline", "no_consent", "sdk_missing"]

        for reason in reasons:
            error = AIUnavailableError(reason)
            assert error.reason == reason
            assert error.retriable is False

    def test_ai_rate_limit_error(self):
        """Test AIRateLimitError with retry_after."""
        error = AIRateLimitError(retry_after_seconds=60.0)
        assert error.retry_after_seconds == 60.0
        assert error.retriable is True

    def test_ai_token_limit_error(self):
        """Test AITokenLimitError with limits."""
        error = AITokenLimitError(limit=8192, actual=10000)
        assert error.limit == 8192
        assert error.actual == 10000
        assert error.retriable is False

    def test_ai_model_not_found_error(self):
        """Test AIModelNotFoundError with model name."""
        error = AIModelNotFoundError("gemini-invalid")
        assert error.model_name == "gemini-invalid"
        assert "gemini-invalid" in str(error)

    def test_exception_inheritance(self):
        """Test all exceptions inherit from AIClientError."""
        exceptions = [
            AIUnavailableError("disabled"),
            AIAuthError(),
            AIRateLimitError(),
            AIQuotaExceededError(),
            AIServerError(),
            AIBadRequestError(),
            AITimeoutError(30.0),
            AITokenLimitError(),
            AIModelNotFoundError("model"),
            AIContentBlockedError(),
        ]

        for exc in exceptions:
            assert isinstance(exc, AIClientError)


# =============================================================================
# Response Model Tests
# =============================================================================


class TestResponseModels:
    """Test response models."""

    def test_ai_response_creation(self):
        """Test AIResponse creation."""
        response = AIResponse(
            text="Hello world",
            model="gemini-1.5-pro",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            finish_reason="STOP",
            latency_ms=100.0,
        )

        assert response.text == "Hello world"
        assert response.model == "gemini-1.5-pro"
        assert response.total_tokens == 15

    def test_ai_response_is_truncated(self):
        """Test is_truncated detection."""
        # Not truncated
        response = AIResponse(text="", model="test", finish_reason="STOP")
        assert response.is_truncated() is False

        # Truncated
        response = AIResponse(text="", model="test", finish_reason="MAX_TOKENS")
        assert response.is_truncated() is True

    def test_ai_response_to_dict(self):
        """Test to_dict excludes raw_response."""
        response = AIResponse(
            text="Hello",
            model="test",
            raw_response={"secret": "data"},
        )

        d = response.to_dict()
        assert "raw_response" not in d
        assert d["text"] == "Hello"

    def test_structured_response_success(self):
        """Test StructuredResponse with successful parse."""
        response = StructuredResponse(
            data={"key": "value"},
            raw_text='{"key": "value"}',
            model="test",
            parse_success=True,
        )

        assert response.data["key"] == "value"
        assert response.parse_success is True

    def test_structured_response_failure(self):
        """Test StructuredResponse with failed parse."""
        response = StructuredResponse(
            data={},
            raw_text="not json",
            model="test",
            parse_success=False,
            parse_error="Invalid JSON",
        )

        assert response.parse_success is False
        assert response.parse_error == "Invalid JSON"


# =============================================================================
# Consent Management Tests
# =============================================================================


class TestConsentManagement:
    """Test consent management system."""

    def test_initial_consent_state(self):
        """Test consent is initially False."""
        revoke_consent()
        assert has_consent() is False

    def test_grant_consent_programmatic(self):
        """Test programmatic consent grant."""
        revoke_consent()
        assert has_consent() is False

        grant_consent_programmatic()
        assert has_consent() is True

    def test_revoke_consent(self):
        """Test consent revocation."""
        grant_consent_programmatic()
        assert has_consent() is True

        revoke_consent()
        assert has_consent() is False

    def test_request_consent_with_input(self):
        """Test interactive consent request."""
        revoke_consent()

        # Mock user input "y"
        with patch("builtins.input", return_value="y"), patch("builtins.print"):
            result = request_consent(force=True)

        assert result is True
        assert has_consent() is True

    def test_request_consent_declined(self):
        """Test consent declined."""
        revoke_consent()

        # Mock user input "n"
        with patch("builtins.input", return_value="n"), patch("builtins.print"):
            result = request_consent(force=True)

        assert result is False
        assert has_consent() is False


# =============================================================================
# AIClient Tests
# =============================================================================


class TestAIClient:
    """Test AIClient class."""

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_client_initialization(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test successful client initialization."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        assert client._is_configured is True
        mock_genai.Client.assert_called_once()

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    def test_client_disabled_mode(
        self,
        mock_get_config,
        mock_genai,
        mock_disabled_config,
    ):
        """Test client with AI disabled."""
        mock_get_config.return_value = mock_disabled_config

        client = AIClient(config=mock_disabled_config)

        assert client.is_available() is False

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_is_available(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test is_available() check."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        assert client.is_available() is True

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_ensure_available_no_consent(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test _ensure_available raises when no consent."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        revoke_consent()
        client = AIClient(config=mock_config)

        with pytest.raises(AIUnavailableError) as exc_info:
            client._ensure_available()

        assert exc_info.value.reason == "no_consent"

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_generate_success(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
        mock_genai_response,
    ):
        """Test successful generate() call."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_genai_response
        mock_genai.Client.return_value = mock_client

        grant_consent_programmatic()
        client = AIClient(config=mock_config)

        response = client.generate("Test prompt")

        assert isinstance(response, AIResponse)
        assert response.text == "Generated response text"
        assert response.total_tokens == 150

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_generate_structured_success(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test generate_structured() with valid JSON."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        # Create response with JSON text
        mock_response = MagicMock()
        mock_response.text = '{"key": "value", "count": 42}'
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=25,
            total_token_count=75,
        )
        mock_response.prompt_feedback = MagicMock(block_reason=None)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        grant_consent_programmatic()
        client = AIClient(config=mock_config)

        response = client.generate_structured("Generate JSON")

        assert response.parse_success is True
        assert response.data["key"] == "value"
        assert response.data["count"] == 42

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_generate_structured_invalid_json(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test generate_structured() with invalid JSON."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        # Response with invalid JSON
        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=25,
            total_token_count=75,
        )
        mock_response.prompt_feedback = MagicMock(block_reason=None)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        grant_consent_programmatic()
        client = AIClient(config=mock_config)

        response = client.generate_structured("Generate JSON")

        assert response.parse_success is False
        assert response.parse_error is not None

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_estimate_tokens(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test token estimation."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        # ~4 chars per token
        estimate = client.estimate_tokens("Hello world")  # 11 chars
        assert estimate == 2  # 11 // 4 = 2

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_get_model_info(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test get_model_info()."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        info = client.get_model_info()

        assert info["name"] == "gemini-1.5-pro"
        assert info["is_available"] is True


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestRetryLogic:
    """Test retry behavior."""

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.google_exceptions")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_retry_on_rate_limit(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_google_exc,
        mock_genai,
        mock_config,
        mock_genai_response,
    ):
        """Test retry on rate limit error."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        # Create rate limit exception
        rate_limit_exc = Exception("Rate limit exceeded")
        mock_google_exc.ResourceExhausted = type(rate_limit_exc)

        mock_client = MagicMock()
        # First call fails, second succeeds
        mock_client.models.generate_content.side_effect = [
            rate_limit_exc,
            mock_genai_response,
        ]
        mock_genai.Client.return_value = mock_client

        grant_consent_programmatic()
        client = AIClient(config=mock_config)

        response = client.generate("Test prompt")

        assert response.text == "Generated response text"
        assert mock_client.models.generate_content.call_count == 2

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_no_retry_on_auth_error(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test no retry on auth error."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        # Create auth error
        auth_exc = Exception("401 Unauthorized")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = auth_exc
        mock_genai.Client.return_value = mock_client

        grant_consent_programmatic()
        client = AIClient(config=mock_config)

        with pytest.raises(AIClientError):
            client.generate("Test prompt")

        # Should only try once (no retries for auth)
        assert mock_client.models.generate_content.call_count == 1


# =============================================================================
# Exception Mapping Tests
# =============================================================================


class TestExceptionMapping:
    """Test exception mapping from SDK errors."""

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_map_rate_limit_error(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test mapping of rate limit errors."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        error = Exception("429 rate limit exceeded")
        mapped = client._map_exception(error)

        assert isinstance(mapped, AIRateLimitError)
        assert mapped.retriable is True

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_map_auth_error(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test mapping of auth errors."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        error = Exception("401 unauthorized")
        mapped = client._map_exception(error)

        assert isinstance(mapped, AIAuthError)
        assert mapped.retriable is False

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_map_token_limit_error(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test mapping of token limit errors."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = AIClient(config=mock_config)

        error = Exception("Token limit exceeded")
        mapped = client._map_exception(error)

        assert isinstance(mapped, AITokenLimitError)
        assert mapped.retriable is False


# =============================================================================
# Decorator Tests
# =============================================================================


class TestRequireAIDecorator:
    """Test @require_ai decorator."""

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_require_ai_passes_when_available(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_config,
    ):
        """Test decorator passes when AI available."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        @require_ai
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.get_config")
    def test_require_ai_raises_when_disabled(
        self,
        mock_get_config,
        mock_disabled_config,
    ):
        """Test decorator raises when AI disabled."""
        mock_get_config.return_value = mock_disabled_config

        @require_ai
        def my_function():
            return "success"

        with pytest.raises(AIUnavailableError) as exc_info:
            my_function()

        assert exc_info.value.reason == "disabled"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetClient:
    """Test get_client() factory function."""

    @patch("dlnai.ai.client.GENAI_AVAILABLE", True)
    @patch("dlnai.ai.client.genai")
    @patch("dlnai.ai.client.get_config")
    @patch("dlnai.ai.client.get_api_key")
    def test_get_client_success(
        self,
        mock_get_api_key,
        mock_get_config,
        mock_genai,
        mock_config,
    ):
        """Test successful client creation."""
        mock_get_config.return_value = mock_config
        mock_get_api_key.return_value = MagicMock(get_secret_value=lambda: "test-key")

        client = get_client(config=mock_config)

        assert isinstance(client, AIClient)
        assert client.is_available() is True

    @patch("dlnai.ai.client.GENAI_AVAILABLE", False)
    def test_get_client_sdk_missing(self):
        """Test client creation when SDK missing."""
        client = get_client()

        assert client.is_available() is False
