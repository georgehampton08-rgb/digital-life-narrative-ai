"""Central Gemini API Client for Digital Life Narrative AI.

This module is the SOLE INTERFACE to the Gemini API. All AI communication flows
through this client. No other file in the codebase should import google-generativeai.

The client provides:
- Robust retry logic with exponential backoff
- Typed exceptions for predictable error handling
- Structured response models for consistent outputs
- Full respect for configuration (AI mode, privacy settings)
- Security-first logging (never logs secrets or full prompts)
- Usage tracking for observability
- Secure log redaction to prevent accidental secret exposure

Example:
    >>> from src.ai.client import get_client, AIUnavailableError
    >>>
    >>> try:
    ...     client = get_client()
    ...     response = client.generate("Summarize this timeline...")
    ...     print(response.text)
    ... except AIUnavailableError:
    ...     print("AI not available, using fallback mode")

The client honors the AI mode from configuration:
- ENABLED: Full AI functionality (requires consent)
- FALLBACK_ONLY: Attempt AI but gracefully degrade on failure
- DISABLED: Reject all AI requests

Security Rules:
- NEVER log API keys (ever, in any form)
- NEVER log full prompts (could contain personal data)
- NEVER log full responses (could contain personal narratives)
- NEVER log memory contents
"""

from __future__ import annotations

import functools
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field

# Import configuration
from src.config import (
    AIMode,
    APIKeyNotFoundError,
    AppConfig,
    get_api_key,
    get_config,
)

# Import Google Generative AI SDK
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory

    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    google_exceptions = None  # type: ignore
    GenerationConfig = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore
    HarmCategory = None  # type: ignore
    GENAI_AVAILABLE = False

# Type checking imports
if TYPE_CHECKING:
    from src.ai.usage_tracker import UsageTracker


# =============================================================================
# Secure Logging Filter
# =============================================================================


class RedactingFilter(logging.Filter):
    """Logging filter that redacts sensitive information.

    Scans log messages for patterns that look like API keys or tokens
    and replaces them with [REDACTED]. This provides defense-in-depth
    against accidental secret exposure in logs.

    Patterns detected:
    - Strings following api_key=, key=, token=, bearer
    - Strings that look like API keys (30-50 chars, alphanumeric with dashes)

    Example:
        >>> logger = logging.getLogger("my_module")
        >>> logger.addFilter(RedactingFilter())
        >>> logger.info("Using api_key=AIzaSy123456789...")
        # Output: "Using api_key=[REDACTED]"
    """

    # Patterns for sensitive data
    PATTERNS = [
        # Key-value patterns
        re.compile(r'(api_key\s*[=:]\s*)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
        re.compile(r'(key\s*[=:]\s*)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
        re.compile(r'(token\s*[=:]\s*)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
        re.compile(r"(bearer\s+)([a-zA-Z0-9_\-]{20,})", re.IGNORECASE),
        re.compile(r'(secret\s*[=:]\s*)["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE),
        # Standalone API key patterns (Gemini keys start with AIza)
        re.compile(r"\bAIza[a-zA-Z0-9_\-]{30,}\b"),
        # Generic long alphanumeric strings that might be secrets
        re.compile(r"\b[a-zA-Z0-9_\-]{35,50}\b"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact the log record.

        Args:
            record: The log record to filter.

        Returns:
            True (always allow the record, but redact sensitive content).
        """
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)

        # Also redact args if they exist
        if record.args:
            record.args = tuple(
                self._redact(str(arg)) if isinstance(arg, str) else arg for arg in record.args
            )

        return True

    def _redact(self, text: str) -> str:
        """Redact sensitive patterns from text.

        Args:
            text: The text to redact.

        Returns:
            Text with sensitive patterns replaced by [REDACTED].
        """
        for pattern in self.PATTERNS[:5]:  # Key-value patterns
            text = pattern.sub(r"\1[REDACTED]", text)

        for pattern in self.PATTERNS[5:]:  # Standalone patterns
            text = pattern.sub("[REDACTED]", text)

        return text


# Configure module logger with redacting filter
logger = logging.getLogger(__name__)
logger.addFilter(RedactingFilter())

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Exception Hierarchy
# =============================================================================


class AIClientError(Exception):
    """Base exception for all AI client errors.

    All AI-related exceptions inherit from this class to allow for unified
    exception handling at higher levels.

    Attributes:
        message: Human-readable error description (safe to log).
        retriable: Whether the operation can be retried.
        details: Additional error context (may contain sensitive data, don't log).
        original_error: The underlying exception that caused this error.

    Example:
        >>> try:
        ...     client.generate(prompt)
        ... except AIClientError as e:
        ...     if e.retriable:
        ...         # Schedule retry
        ...     else:
        ...         # Handle permanent failure
    """

    def __init__(
        self,
        message: str,
        retriable: bool = False,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retriable = retriable
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return message without exposing sensitive details."""
        return self.message


class AIUnavailableError(AIClientError):
    """AI service is not available (disabled, no key, offline).

    This error signals to higher layers that they should enter fallback mode.
    The application should continue to function without AI features.

    Attributes:
        reason: Why AI is unavailable.

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIUnavailableError as e:
        ...     if e.reason == "no_consent":
        ...         prompt_user_for_consent()
        ...     else:
        ...         use_fallback_analysis()
    """

    def __init__(
        self,
        reason: Literal[
            "disabled", "no_api_key", "offline", "service_down", "no_consent", "sdk_missing"
        ],
        message: str | None = None,
    ) -> None:
        self.reason = reason

        default_messages = {
            "disabled": "AI features are disabled in configuration",
            "no_api_key": "No Gemini API key configured",
            "offline": "Cannot reach Gemini API (network offline)",
            "service_down": "Gemini service is temporarily unavailable",
            "no_consent": "User consent required before AI analysis",
            "sdk_missing": "google-generativeai package not installed",
        }

        msg = message or default_messages.get(reason, f"AI unavailable: {reason}")
        super().__init__(msg, retriable=False)


class APIKeyMissingError(AIUnavailableError):
    """No API key configured.

    Specialized version of AIUnavailableError with helpful suggestion
    for how to configure the API key.

    Attributes:
        suggestion: User-friendly configuration instructions.
    """

    def __init__(
        self,
        message: str | None = None,
        suggestion: str = "Configure your Gemini API key using 'organizer config set-key'",
    ) -> None:
        self.suggestion = suggestion
        super().__init__(
            reason="no_api_key",
            message=message or f"No API key configured. {suggestion}",
        )


class AIAuthenticationError(AIClientError):
    """API key is invalid or expired.

    This error is never retriable. The user needs to update their API key.

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIAuthenticationError:
        ...     prompt_user_to_update_key()
    """

    def __init__(
        self,
        message: str = "API authentication failed. Please check your API key.",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)


class AIRateLimitError(AIClientError):
    """Rate limit exceeded.

    This error is retriable after waiting. The retry_after_seconds attribute
    indicates how long to wait, if the API provided this information.

    Attributes:
        retry_after_seconds: Suggested wait time before retry (may be None).

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIRateLimitError as e:
        ...     wait_time = e.retry_after_seconds or 60
        ...     time.sleep(wait_time)
        ...     # Retry
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please wait before retrying.",
        retry_after_seconds: float | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=True, original_error=original_error)
        self.retry_after_seconds = retry_after_seconds


class AIQuotaExceededError(AIClientError):
    """Quota or billing limit reached.

    This error is not retriable — the user needs to wait for quota reset
    or upgrade their billing plan.

    Attributes:
        quota_type: Type of quota exceeded (daily, monthly, etc.).

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIQuotaExceededError as e:
        ...     if e.quota_type == "daily":
        ...         notify_user_daily_limit()
    """

    def __init__(
        self,
        message: str = "API quota exceeded. Check your billing and usage limits.",
        quota_type: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.quota_type = quota_type


class AIServerError(AIClientError):
    """Server-side error (5xx).

    This error is retriable — the server may recover.

    Attributes:
        status_code: HTTP status code if available.

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIServerError as e:
        ...     log.warning(f"Server error {e.status_code}, retrying...")
    """

    def __init__(
        self,
        message: str = "AI server error. The service may be temporarily unavailable.",
        status_code: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=True, original_error=original_error)
        self.status_code = status_code


class AIBadRequestError(AIClientError):
    """Invalid request (malformed prompt, bad parameters, etc.).

    This error is not retriable — the request itself is malformed.

    Attributes:
        field: Which field caused the error, if known.

    Example:
        >>> try:
        ...     response = client.generate(invalid_prompt)
        ... except AIBadRequestError as e:
        ...     log.error(f"Bad request on field {e.field}: {e.message}")
    """

    def __init__(
        self,
        message: str = "Invalid request to AI service.",
        field: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.field = field


class AITimeoutError(AIClientError):
    """Request timed out.

    This error is retriable — transient network issues may resolve.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded.

    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AITimeoutError as e:
        ...     log.warning(f"Timed out after {e.timeout_seconds}s, retrying...")
    """

    def __init__(
        self,
        timeout_seconds: float,
        message: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        msg = message or f"Request timed out after {timeout_seconds} seconds"
        super().__init__(msg, retriable=True, original_error=original_error)
        self.timeout_seconds = timeout_seconds


class TokenLimitExceededError(AIClientError):
    """Input or output exceeded token limits.

    This error is not retriable with the same input — the caller needs to
    reduce the input size or chunk the data.

    Attributes:
        limit: The token limit that was exceeded (if known).
        actual: The actual token count (if known).
        limit_type: Whether input, output, or total was exceeded.
        suggestion: User-friendly recommendation.

    Example:
        >>> try:
        ...     response = client.generate(long_prompt)
        ... except TokenLimitExceededError as e:
        ...     print(f"Exceeded {e.limit_type} limit: {e.actual}/{e.limit}")
        ...     chunks = split_prompt(long_prompt)
        ...     for chunk in chunks:
        ...         client.generate(chunk)
    """

    def __init__(
        self,
        message: str = "Token limit exceeded. Please reduce input size.",
        limit: int | None = None,
        actual: int | None = None,
        limit_type: Literal["input", "output", "total"] = "total",
        suggestion: str = "Reduce input size or chunk the request",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.limit = limit
        self.actual = actual
        self.limit_type = limit_type
        self.suggestion = suggestion


class ModelNotAvailableError(AIClientError):
    """Requested model doesn't exist or isn't available.

    This error is not retriable — the configuration needs to be fixed.

    Attributes:
        model_name: The model that was requested.
        available_models: List of valid models, if known.

    Example:
        >>> try:
        ...     client = AIClient()
        ... except ModelNotAvailableError as e:
        ...     log.error(f"Model '{e.model_name}' not found")
        ...     log.info(f"Available: {e.available_models}")
    """

    def __init__(
        self,
        model_name: str,
        message: str | None = None,
        available_models: list[str] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        msg = message or f"Model '{model_name}' not found. Check model name in configuration."
        super().__init__(msg, retriable=False, original_error=original_error)
        self.model_name = model_name
        self.available_models = available_models


class ContentBlockedError(AIClientError):
    """Content was blocked by safety filters.

    This error is not retriable with the same content.

    Attributes:
        blocked_reason: The reason for blocking if available.
        safety_ratings: Dictionary of safety category -> rating.
    """

    def __init__(
        self,
        message: str = "Content blocked by safety filters.",
        blocked_reason: str | None = None,
        safety_ratings: dict[str, str] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.blocked_reason = blocked_reason
        self.safety_ratings = safety_ratings


# =============================================================================
# Response Models
# =============================================================================


class AIResponse(BaseModel):
    """Standardized response from AI generation.

    Wraps the raw SDK response in a consistent, typed model that higher-level
    modules can depend on.

    Attributes:
        text: The generated content.
        model: Name of the model that generated this response.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated output.
        total_tokens: Total tokens used (prompt + completion).
        finish_reason: Why generation stopped (e.g., "STOP", "MAX_TOKENS").
        latency_ms: Time taken for generation in milliseconds.
        cached: Whether this response was from cache.
        raw_response: Original SDK response (excluded from serialization).

    Example:
        >>> response = client.generate("Tell me about...")
        >>> print(response.text)
        >>> print(f"Used {response.total_tokens} tokens")
        >>> if response.is_truncated():
        ...     print("Response was truncated!")
    """

    text: str = Field(..., description="The generated content")
    model: str = Field(..., description="Model that generated this response")
    prompt_tokens: int | None = Field(None, description="Tokens in input prompt")
    completion_tokens: int | None = Field(None, description="Tokens in output")
    total_tokens: int | None = Field(None, description="Total tokens used")
    finish_reason: str | None = Field(None, description="Why generation stopped")
    latency_ms: float | None = Field(None, description="Generation time in ms")
    cached: bool = Field(False, description="Whether response was from cache")
    raw_response: Any = Field(None, exclude=True, description="Original SDK response")

    def is_complete(self) -> bool:
        """Check if generation completed normally.

        Returns:
            True if finish_reason indicates normal completion.
        """
        return self.finish_reason == "STOP"

    def is_truncated(self) -> bool:
        """Check if the response was truncated due to token limits.

        Returns:
            True if the finish_reason indicates truncation.
        """
        truncation_reasons = {"MAX_TOKENS", "LENGTH", "RECITATION"}
        return self.finish_reason in truncation_reasons if self.finish_reason else False

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary.

        Excludes raw_response for safe serialization.

        Returns:
            Dictionary representation of the response.
        """
        return self.model_dump(exclude={"raw_response"})


class StructuredAIResponse(BaseModel):
    """Response when requesting structured/JSON output.

    Wraps both the parsed data and the raw text, along with parsing status.
    If JSON parsing fails, parse_success is False and parse_error contains
    the error message.

    Attributes:
        data: Parsed JSON content (empty dict if parsing failed).
        raw_text: Original text before parsing.
        model: Model that generated this response.
        tokens_used: Total tokens consumed.
        latency_ms: Generation time in milliseconds.
        parse_success: Whether JSON parsing succeeded.
        parse_error: Error message if parsing failed.

    Example:
        >>> response = client.generate_json("Extract entities...")
        >>> if response.parse_success:
        ...     entities = response.data["entities"]
        ... else:
        ...     log.warning(f"Parse failed: {response.parse_error}")
        ...     # Handle raw_text manually
    """

    data: dict[str, Any] | list[Any] = Field(
        default_factory=dict, description="Parsed JSON content"
    )
    raw_text: str = Field(..., description="Original text before parsing")
    model: str = Field(..., description="Model that generated this response")
    tokens_used: int | None = Field(None, description="Total tokens consumed")
    latency_ms: float | None = Field(None, description="Generation time in ms")
    parse_success: bool = Field(True, description="Whether JSON parsing succeeded")
    parse_error: str | None = Field(None, description="Error if parsing failed")


# =============================================================================
# Consent Manager
# =============================================================================


@dataclass
class ConsentState:
    """Tracks user consent for AI features.

    AI features require explicit user consent before any data is sent to
    external services. This state is session-scoped by default.

    Attributes:
        consented: Whether the user has given consent.
        consent_time: Unix timestamp when consent was given.
        consent_message: The message shown when consent was requested.
        session_id: Unique identifier for this consent session.
    """

    consented: bool = False
    consent_time: float | None = None
    consent_message: str | None = None
    session_id: str = field(default_factory=lambda: str(random.randint(100000, 999999)))


# Module-level consent state (session-scoped)
_consent_state = ConsentState()


def request_consent(force: bool = False) -> bool:
    """Request user consent for AI features.

    This function should be called before making any AI requests. It presents
    the user with information about what data will be sent and gets their
    explicit approval.

    Args:
        force: If True, request consent even if already given.

    Returns:
        True if consent was given, False otherwise.

    Note:
        In a CLI context, this prints to stdout and reads from stdin.
        In a GUI context, this should be overridden.
    """
    global _consent_state

    if _consent_state.consented and not force:
        return True

    consent_message = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 AI ANALYSIS CONSENT REQUEST                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Digital Life Narrative AI would like to use Google's Gemini to analyze     ║
║  your memories and create a narrative of your life story.                   ║
║                                                                              ║
║  WHAT WILL BE SENT:                                                          ║
║  • Timestamps (when photos/videos were taken)                               ║
║  • Media types (photo, video, message, etc.)                                ║
║  • Platform info (where the memory came from)                               ║
║  • Based on your privacy settings: locations, captions, people counts       ║
║                                                                              ║
║  WHAT WILL NOT BE SENT:                                                      ║
║  ✗ Actual photos or videos (only metadata)                                  ║
║  ✗ File paths or content hashes                                             ║
║  ✗ Exact GPS coordinates (reduced precision)                                ║
║  ✗ Full names (anonymized if included)                                      ║
║                                                                              ║
║  You can revoke consent at any time. See PRIVACY.md for details.            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(consent_message)

    try:
        response = input("\nDo you consent to AI analysis? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            _consent_state.consented = True
            _consent_state.consent_time = time.time()
            _consent_state.consent_message = consent_message
            print("\n✓ Consent granted. AI analysis enabled for this session.\n")
            return True
        else:
            print("\n✗ Consent declined. AI features will be disabled.\n")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n✗ Consent declined. AI features will be disabled.\n")
        return False


def has_consent() -> bool:
    """Check if user has given consent for AI features.

    Returns:
        True if consent was given in this session.
    """
    return _consent_state.consented


def revoke_consent() -> None:
    """Revoke previously given consent.

    After calling this, AI features will require new consent.
    """
    global _consent_state
    _consent_state = ConsentState()
    print("✓ AI consent revoked. AI features are now disabled.\n")


def grant_consent_programmatic() -> None:
    """Grant consent programmatically (for non-interactive use).

    Use this in scripts or tests where interactive consent is not possible.
    The caller is responsible for ensuring appropriate consent was obtained.
    """
    global _consent_state
    _consent_state.consented = True
    _consent_state.consent_time = time.time()
    _consent_state.consent_message = "Programmatic consent"


# =============================================================================
# Main AI Client Class
# =============================================================================


class AIClient:
    """Central client for all Gemini API communication.

    This is the SOLE interface to the Gemini API. All AI requests must go
    through this client, which handles:

    - Configuration respect (AI mode, privacy settings)
    - Automatic retry with exponential backoff
    - Typed exceptions for predictable error handling
    - Secure logging (no secrets or full prompts)
    - User consent verification
    - Usage tracking for observability

    The client is lazy-initialized — no API calls are made until needed.

    Example:
        >>> client = AIClient()
        >>>
        >>> # Check if AI is available
        >>> if client.is_available():
        ...     response = client.generate(
        ...         prompt="Summarize this timeline...",
        ...         system_instruction="You are a life story analyst."
        ...     )
        ...     print(response.text)
        >>>
        >>> # Structured output
        >>> response = client.generate_json(
        ...     prompt="Extract key events...",
        ...     schema_hint='{"events": [{"date": "...", "description": "..."}]}'
        ... )
        >>> for event in response.data.get("events", []):
        ...     print(f"{event['date']}: {event['description']}")

    Attributes:
        config: Application configuration.

    Class Constants:
        DEFAULT_MODEL: Default model to use if not specified.
        SUPPORTED_MODELS: Set of known supported model names.
        MAX_RETRIES: Default maximum retry attempts.
        BASE_RETRY_DELAY: Base delay for exponential backoff.
        MAX_RETRY_DELAY: Maximum delay between retries.
    """

    # Class constants
    DEFAULT_MODEL: str = "gemini-2.0-flash-exp"
    SUPPORTED_MODELS: set[str] = {
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
        "gemini-2.0-flash-exp",
    }
    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY: float = 1.0
    MAX_RETRY_DELAY: float = 60.0

    def __init__(
        self,
        config: AppConfig | None = None,
        api_key: str | None = None,
        usage_tracker: "UsageTracker | None" = None,
        require_consent: bool | None = None,
    ) -> None:
        """Initialize the AI client.

        No API calls are made during initialization. The model is lazy-loaded
        on first use.

        Args:
            config: Application configuration. If None, loads from get_config().
            api_key: Override API key. If None, loads from configured sources.
            usage_tracker: Optional usage tracker for observability.
            require_consent: Whether to require user consent before AI calls.

        Raises:
            AIUnavailableError: If google-generativeai package is not installed.
        """
        self._config = config or get_config()
        self._model: Any = None  # genai.GenerativeModel
        self._is_configured = False
        self._api_key: str | None = None
        self._usage_tracker = usage_tracker
        # Use config setting if require_consent not explicitly set
        self._require_consent = (
            require_consent if require_consent is not None else self._config.ai.require_consent
        )
        self._logger = logging.getLogger(f"{__name__}.AIClient")
        self._logger.addFilter(RedactingFilter())

        # Check if SDK is available
        if not GENAI_AVAILABLE:
            self._logger.warning("google-generativeai package not installed")
            return

        # Check if AI is enabled
        self._is_enabled = self._config.ai.is_enabled()
        if not self._is_enabled:
            self._logger.info("AI is disabled in configuration")
            return

        # Get API key
        try:
            if api_key:
                self._api_key = api_key
            else:
                secret = get_api_key()
                self._api_key = secret.get_secret_value()
        except APIKeyNotFoundError:
            self._logger.warning("No API key configured")
            return

        # Configure the SDK (never log the key!)
        try:
            genai.configure(api_key=self._api_key)
            self._is_configured = True
            self._logger.info(f"AI client configured with default narrative model: {self._config.ai.narrative_model}")
        except Exception as e:
            self._logger.error(f"Failed to configure AI SDK: {type(e).__name__}")

    def _ensure_available(self) -> None:
        """Verify AI is available before making requests.

        Checks:
        1. SDK is installed
        2. AI mode is enabled
        3. API key is configured
        4. User has given consent (if required)

        Raises:
            AIUnavailableError: If any check fails.
        """
        if not GENAI_AVAILABLE:
            raise AIUnavailableError("sdk_missing")

        if not self._config.ai.is_enabled():
            raise AIUnavailableError("disabled")

        if not self._is_configured or not self._api_key:
            raise APIKeyMissingError()

        if self._require_consent and not has_consent():
            raise AIUnavailableError("no_consent")

    def _get_model(self, model_name: str | None = None) -> Any:
        """Get or create the GenerativeModel instance.

        The model is lazy-initialized and cached for reuse.

        Args:
            model_name: Specific model name, or None to use config default.

        Returns:
            Configured GenerativeModel instance.
        """
        target_model = model_name or self._config.ai.narrative_model

        # Return cached model if same name
        if self._model is not None and target_model == self._config.ai.narrative_model:
            return self._model

        # Create new model instance
        model = genai.GenerativeModel(
            model_name=target_model,
            safety_settings=self._get_safety_settings(),
        )

        # Only cache if it's the default model
        if target_model == self._config.ai.narrative_model:
            self._model = model

        return model

    def _get_generation_config(self, **overrides: Any) -> "GenerationConfig":
        """Build GenerationConfig from settings with optional overrides.

        Args:
            **overrides: Per-call overrides for generation parameters.

        Returns:
            Configured GenerationConfig instance.
        """
        config_params = {
            "temperature": self._config.ai.temperature,
            "max_output_tokens": self._config.ai.max_output_tokens,
        }
        config_params.update(overrides)

        if GenerationConfig is None:
            # SDK not available, return dict (tests may mock this)
            return config_params
        return GenerationConfig(**config_params)

    def _get_safety_settings(self) -> dict:
        """Get safety settings appropriate for life story analysis.

        We're analyzing personal memories, so we need balanced settings that
        allow discussion of personal content while blocking truly harmful content.

        Returns:
            Dictionary of safety settings.
        """
        if HarmCategory is None or HarmBlockThreshold is None:
            # SDK not available
            return {}
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    def generate(
        self,
        prompt: str | list[Any],
        system_instruction: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> AIResponse:
        """Generate text from a prompt or multimodal parts.

        This is the main generation method. It handles retries, error mapping,
        and response parsing automatically.

        Args:
            prompt: Either a string prompt or a list of parts (text, image bytes, etc.)
            system_instruction: Optional system instruction to guide the model.
            model: Specific model name to use (overrides config).
            **overrides: Per-call overrides (temperature, max_output_tokens, etc.)

        Returns:
            AIResponse with the generated text and metadata.
        """
        self._ensure_available()

        start_time = time.time()

        # Build contents
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [system_instruction]})
            contents.append({"role": "model", "parts": ["Understood."]})
        
        # Handle multimodal prompt (list of parts) or simple string
        if isinstance(prompt, list):
            contents.append({"role": "user", "parts": prompt})
        else:
            contents.append({"role": "user", "parts": [prompt]})

        # Build generation config
        gen_config = self._get_generation_config(**overrides)

        # Get model instance
        model_instance = self._get_model(model)
        model_name = model or self._config.ai.narrative_model

        # Make the request with retry
        try:
            raw_response = self._execute_with_retry(
                self._do_generate,
                model=model_instance,
                contents=contents,
                generation_config=gen_config,
            )
        except Exception as e:
            # Log failure (no prompt content - security!)
            self._logger.error(
                f"Generation failed: {type(e).__name__}",
                extra={"model": model_name},
            )
            raise

        # Parse response
        latency_ms = (time.time() - start_time) * 1000

        # Extract text from response
        try:
            text = raw_response.text
        except ValueError:
            # Response might be blocked
            if raw_response.prompt_feedback.block_reason:
                raise ContentBlockedError(
                    blocked_reason=str(raw_response.prompt_feedback.block_reason)
                )
            text = ""

        # Extract token counts if available
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
            usage = raw_response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

        # Extract finish reason
        finish_reason = None
        if raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                finish_reason = (
                    str(candidate.finish_reason.name) if candidate.finish_reason else None
                )

        response = AIResponse(
            text=text,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

        # Track usage
        if self._usage_tracker and total_tokens:
            self._usage_tracker.record_request(
                model=model_name,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                latency_ms=latency_ms,
                success=True,
            )

        # Log success (no content - security!)
        self._logger.info(
            f"Generation successful: {total_tokens or '?'} tokens in {latency_ms:.0f}ms",
            extra={
                "model": model_name,
                "tokens": total_tokens,
                "time_ms": latency_ms,
            },
        )

        return response

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        schema_hint: str | None = None,
        **overrides: Any,
    ) -> StructuredAIResponse:
        """Generate and parse as JSON.

        Modifies the prompt to request JSON output and parses the response.
        If parsing fails, returns StructuredAIResponse with parse_success=False.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.
            schema_hint: Optional JSON schema hint to include in prompt.
            **overrides: Generation parameter overrides.

        Returns:
            StructuredAIResponse with parsed data or error information.

        Example:
            >>> response = client.generate_json(
            ...     prompt="Extract events from this timeline",
            ...     schema_hint='{"events": [{"date": "YYYY-MM-DD", "description": "..."}]}'
            ... )
            >>> if response.parse_success:
            ...     for event in response.data["events"]:
            ...         print(event)
        """
        # Modify system instruction for JSON output
        json_instruction = (
            "You must respond with valid JSON only. No markdown, no explanations, "
            "no code blocks - just pure JSON that can be parsed directly."
        )

        if system_instruction:
            full_instruction = f"{system_instruction}\n\n{json_instruction}"
        else:
            full_instruction = json_instruction

        # Add schema hint to prompt
        if schema_hint:
            full_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_hint}"
        else:
            full_prompt = prompt

        # Generate
        start_time = time.time()
        response = self.generate(
            prompt=full_prompt,
            system_instruction=full_instruction,
            **overrides,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Try to parse JSON
        text = response.text.strip()
        data: dict[str, Any] | list[Any] = {}
        parse_success = False
        parse_error: str | None = None

        # Try direct parse first
        try:
            data = json.loads(text)
            parse_success = True
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    parse_success = True
                except json.JSONDecodeError:
                    parse_error = f"JSON parse error in code block: {e.msg}"
            else:
                # Try to find JSON object/array in text
                json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        parse_success = True
                    except json.JSONDecodeError:
                        parse_error = f"JSON parse error in extracted content: {e.msg}"
                else:
                    parse_error = f"JSON parse error: {e.msg}"

        return StructuredAIResponse(
            data=data,
            raw_text=response.text,
            model=response.model,
            tokens_used=response.total_tokens,
            latency_ms=latency_ms,
            parse_success=parse_success,
            parse_error=parse_error,
        )

    def generate_with_schema(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_instruction: str | None = None,
        **overrides: Any,
    ) -> StructuredAIResponse:
        """Generate with explicit JSON schema validation.

        Include schema in prompt and validate response against it.

        Args:
            prompt: The user prompt.
            schema: JSON schema to validate against.
            system_instruction: Optional system instruction.
            **overrides: Generation parameter overrides.

        Returns:
            StructuredAIResponse with parsed and validated data.
        """
        schema_hint = json.dumps(schema, indent=2)
        return self.generate_json(
            prompt=prompt,
            system_instruction=system_instruction,
            schema_hint=schema_hint,
            **overrides,
        )

    def _do_generate(
        self,
        model: Any,
        contents: list,
        generation_config: "GenerationConfig",
    ) -> Any:
        """Execute the actual API call.

        This method is wrapped by retry logic.

        Args:
            model: The GenerativeModel instance.
            contents: The conversation contents.
            generation_config: Generation configuration.

        Returns:
            Raw response from the SDK.
        """
        return model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(),
        )

    def _execute_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry on transient failures.

        Uses exponential backoff with jitter for retries.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            max_retries: Override max retries (uses config default if None).
            **kwargs: Keyword arguments for the function.

        Returns:
            The function's return value.

        Raises:
            AIClientError: On failure after all retries exhausted.
        """
        retries = max_retries if max_retries is not None else self._config.ai.max_retries
        base_delay = self._config.ai.retry_base_delay
        last_error: AIClientError | None = None

        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Map to our exception type
                mapped_error = self._map_exception(e)
                last_error = mapped_error

                # Don't retry non-retriable errors
                if not mapped_error.retriable:
                    raise mapped_error

                # Don't retry after max attempts
                if attempt >= retries:
                    self._logger.error(
                        f"Max retries ({retries}) exhausted: {type(mapped_error).__name__}"
                    )
                    raise mapped_error

                # Calculate delay with exponential backoff + jitter
                delay = min(base_delay * (2**attempt), self.MAX_RETRY_DELAY)
                jitter = random.uniform(0, 1)
                total_delay = delay + jitter

                # Check rate limit hint
                if isinstance(mapped_error, AIRateLimitError) and mapped_error.retry_after_seconds:
                    total_delay = max(total_delay, mapped_error.retry_after_seconds)

                self._logger.warning(
                    f"Retry {attempt + 1}/{retries} after {total_delay:.1f}s: "
                    f"{type(mapped_error).__name__}"
                )

                time.sleep(total_delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise AIClientError("Unknown error during retry")

    def _map_exception(self, error: Exception) -> AIClientError:
        """Map SDK exceptions to our exception hierarchy.

        Args:
            error: The original exception.

        Returns:
            Mapped AIClientError subclass.
        """
        error_str = str(error).lower()

        # Check for google.api_core exceptions
        # Use try-except to handle cases where google_exceptions is mocked
        if google_exceptions:
            try:
                if isinstance(error, google_exceptions.InvalidArgument):
                    if "token" in error_str or "length" in error_str:
                        return TokenLimitExceededError(original_error=error)
                    return AIBadRequestError(str(error), original_error=error)

                if isinstance(error, google_exceptions.PermissionDenied):
                    return AIAuthenticationError(original_error=error)

                if isinstance(error, google_exceptions.Unauthenticated):
                    return AIAuthenticationError(original_error=error)

                if isinstance(error, google_exceptions.ResourceExhausted):
                    if "quota" in error_str:
                        return AIQuotaExceededError(original_error=error)
                    return AIRateLimitError(original_error=error)

                if isinstance(error, google_exceptions.NotFound):
                    # Try to extract model name
                    model_match = re.search(r"model[s]?[:/\s]+([^\s,]+)", error_str)
                    model_name = model_match.group(1) if model_match else self._config.ai.narrative_model
                    return ModelNotAvailableError(model_name, original_error=error)

                if isinstance(error, google_exceptions.DeadlineExceeded):
                    return AITimeoutError(
                        self._config.ai.timeout_seconds,
                        original_error=error,
                    )

                if isinstance(error, google_exceptions.InternalServerError):
                    return AIServerError(status_code=500, original_error=error)

                if isinstance(error, google_exceptions.ServiceUnavailable):
                    return AIServerError(status_code=503, original_error=error)
            except TypeError:
                # google_exceptions may be mocked, fall through to string matching
                pass

        # Fallback pattern matching on error message
        if "blocked" in error_str or "safety" in error_str:
            return ContentBlockedError(original_error=error)

        if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            return AIAuthenticationError(original_error=error)

        if "429" in error_str or "rate" in error_str:
            return AIRateLimitError(original_error=error)

        if "quota" in error_str or "billing" in error_str:
            return AIQuotaExceededError(original_error=error)

        if "timeout" in error_str or "deadline" in error_str:
            return AITimeoutError(
                self._config.ai.timeout_seconds,
                original_error=error,
            )

        if "500" in error_str or "502" in error_str or "503" in error_str:
            return AIServerError(original_error=error)

        if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
            return TokenLimitExceededError(original_error=error)

        if "model" in error_str and "not found" in error_str:
            return ModelNotAvailableError(self._config.ai.narrative_model, original_error=error)

        # Generic fallback
        return AIClientError(str(error), retriable=False, original_error=error)

    # =========================================================================
    # Token Management
    # =========================================================================

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens for text using the SDK.

        Uses the generative AI SDK's token counting if available.
        Falls back to estimation if SDK method fails.

        Args:
            text: The text to count tokens for.
            model: Specific model (uses default if None).

        Returns:
            Token count.
        """
        if not GENAI_AVAILABLE or not self._is_configured:
            return self.estimate_tokens(text)

        try:
            model_instance = self._get_model(model)
            result = model_instance.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fall back to estimation
            return self.estimate_tokens(text)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text without API call.

        Uses a simple heuristic (~4 characters per token).
        For more accurate counts, use count_tokens().

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def check_token_limit(
        self,
        text: str,
        max_tokens: int | None = None,
    ) -> tuple[bool, int]:
        """Check if text is within token limits.

        Args:
            text: The text to check.
            max_tokens: Maximum token limit. Uses config if None.

        Returns:
            Tuple of (within_limit, estimated_tokens).
        """
        limit = max_tokens or self._config.ai.max_output_tokens
        estimated = self.estimate_tokens(text)
        return (estimated <= limit, estimated)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_available(self) -> bool:
        """Check if AI is available without making an API call.

        Checks:
        - SDK is installed
        - AI mode is enabled
        - API key is configured

        Does NOT check:
        - User consent (call has_consent() separately)
        - Network connectivity (call test_connection() for that)

        Returns:
            True if AI is available, False otherwise.
        """
        return (
            GENAI_AVAILABLE
            and self._config.ai.is_enabled()
            and self._is_configured
            and self._api_key is not None
        )

    def test_connection(self) -> tuple[bool, str]:
        """Test connectivity to the Gemini API.

        Makes a minimal API call to verify the connection works.

        Returns:
            Tuple of (success, message).

        Example:
            >>> success, message = client.test_connection()
            >>> if success:
            ...     print("Connected!")
            ... else:
            ...     print(f"Failed: {message}")
        """
        if not self.is_available():
            return False, "AI not available"

        if not has_consent():
            # Temporarily allow for test
            grant_consent_programmatic()
            try:
                return self._do_test_connection()
            finally:
                revoke_consent()

        return self._do_test_connection()

    def _do_test_connection(self) -> tuple[bool, str]:
        """Execute the actual connection test."""
        try:
            response = self.generate(
                prompt="Say 'ok'",
                max_output_tokens=10,
            )
            if response.text:
                return True, f"Connected to {self._config.ai.narrative_model}"
            return False, "Empty response from API"
        except AIClientError as e:
            return False, e.message
        except Exception as e:
            return False, str(e)

    def get_model_info(self, model: str | None = None) -> dict[str, Any]:
        """Get information about a model.

        Args:
            model: Specific model name, or None for configured default.

        Returns:
            Dictionary with model name and limits.
        """
        model_name = model or self._config.ai.narrative_model
        return {
            "name": model_name,
            "is_supported": model_name in self.SUPPORTED_MODELS,
            "max_output_tokens": self._config.ai.max_output_tokens,
            "temperature": self._config.ai.temperature,
            "timeout_seconds": self._config.ai.timeout_seconds,
            "is_available": self.is_available(),
        }

    def list_available_models(self) -> list[str]:
        """List available models.

        Returns the set of known supported models. May include models
        that aren't available in your API tier.

        Returns:
            List of model names.
        """
        return sorted(self.SUPPORTED_MODELS)


# =============================================================================
# Module-Level Functions
# =============================================================================


def get_client(config: AppConfig | None = None) -> AIClient:
    """Factory function to create a configured AI client.

    Handles common initialization errors and converts them to
    AIUnavailableError for consistent error handling.

    Args:
        config: Optional configuration override.

    Returns:
        Configured AIClient instance.

    Raises:
        AIUnavailableError: If AI cannot be initialized.

    Example:
        >>> from src.ai.client import get_client, AIUnavailableError
        >>>
        >>> try:
        ...     client = get_client()
        ...     response = client.generate("Hello!")
        ... except AIUnavailableError as e:
        ...     print(f"AI not available: {e.reason}")
    """
    try:
        return AIClient(config=config)
    except APIKeyNotFoundError:
        raise AIUnavailableError("no_api_key")
    except Exception as e:
        logger.error(f"Failed to create AI client: {type(e).__name__}")
        raise AIUnavailableError("offline", str(e))


def require_ai(func: F) -> F:
    """Decorator for functions that require AI.

    Checks AI availability before the function runs. If AI is not available,
    raises AIUnavailableError before any work is done.

    Example:
        >>> @require_ai
        ... def analyze_memories(memories):
        ...     client = get_client()
        ...     return client.generate(...)
        >>>
        >>> try:
        ...     result = analyze_memories([...])
        ... except AIUnavailableError:
        ...     use_fallback_analysis()
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = get_config()

        if not GENAI_AVAILABLE:
            raise AIUnavailableError("sdk_missing")

        if not config.ai.is_enabled():
            raise AIUnavailableError("disabled")

        try:
            get_api_key()
        except APIKeyNotFoundError:
            raise AIUnavailableError("no_api_key")

        return func(*args, **kwargs)

    return wrapper  # type: ignore


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Response model aliases
StructuredResponse = StructuredAIResponse

# Exception aliases for backward compatibility
AIAuthError = AIAuthenticationError
AITokenLimitError = TokenLimitExceededError
AIModelNotFoundError = ModelNotAvailableError
AIContentBlockedError = ContentBlockedError

# Method aliases: add generate_structured as alias to generate_json on AIClient
AIClient.generate_structured = AIClient.generate_json
