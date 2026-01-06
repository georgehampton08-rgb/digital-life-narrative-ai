"""Gemini AI Client for Digital Life Narrative AI.

This module is the SOLE INTERFACE to the Gemini API. All AI communication flows
through this client. No other file in the codebase should import google-generativeai.

The client provides:
- Robust retry logic with exponential backoff
- Typed exceptions for predictable error handling
- Structured response models for consistent outputs
- Full respect for configuration (AI mode, privacy settings)
- Security-first logging (never logs secrets or full prompts)

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
"""

from __future__ import annotations

import functools
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeVar

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


# Configure module logger - NEVER log secrets
logger = logging.getLogger(__name__)

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
        message: Human-readable error description.
        retriable: Whether the operation can be retried.
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
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retriable = retriable
        self.original_error = original_error
    
    def __str__(self) -> str:
        return self.message


class AIUnavailableError(AIClientError):
    """AI is disabled by configuration or no API key is available.
    
    This error signals to higher layers that they should enter fallback mode.
    The application should continue to function without AI features.
    
    Attributes:
        reason: Why AI is unavailable ("disabled", "no_key", "offline", "no_consent").
    
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
        reason: Literal["disabled", "no_key", "offline", "no_consent", "sdk_missing"],
        message: str | None = None,
    ) -> None:
        self.reason = reason
        
        default_messages = {
            "disabled": "AI features are disabled in configuration",
            "no_key": "No Gemini API key configured",
            "offline": "Cannot reach Gemini API (network offline)",
            "no_consent": "User consent required before AI analysis",
            "sdk_missing": "google-generativeai package not installed",
        }
        
        msg = message or default_messages.get(reason, f"AI unavailable: {reason}")
        super().__init__(msg, retriable=False)


class AIAuthError(AIClientError):
    """Authentication failed — invalid or expired API key.
    
    This error is never retriable. The user needs to update their API key.
    
    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIAuthError:
        ...     prompt_user_to_update_key()
    """
    
    def __init__(
        self,
        message: str = "API authentication failed. Please check your API key.",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)


class AIRateLimitError(AIClientError):
    """Rate limit exceeded (HTTP 429).
    
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
    
    Example:
        >>> try:
        ...     response = client.generate(prompt)
        ... except AIQuotaExceededError:
        ...     notify_user_quota_exceeded()
    """
    
    def __init__(
        self,
        message: str = "API quota exceeded. Check your billing and usage limits.",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)


class AIServerError(AIClientError):
    """Server-side error (HTTP 5xx).
    
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
    """Invalid request (HTTP 4xx except auth/rate limit).
    
    This error is not retriable — the request itself is malformed.
    
    Example:
        >>> try:
        ...     response = client.generate(invalid_prompt)
        ... except AIBadRequestError as e:
        ...     log.error(f"Bad request: {e.message}")
    """
    
    def __init__(
        self,
        message: str = "Invalid request to AI service.",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)


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


class AITokenLimitError(AIClientError):
    """Input or output exceeded token limits.
    
    This error is not retriable with the same input — the caller needs to
    reduce the input size or chunk the data.
    
    Attributes:
        limit: The token limit that was exceeded (if known).
        actual: The actual token count (if known).
    
    Example:
        >>> try:
        ...     response = client.generate(long_prompt)
        ... except AITokenLimitError as e:
        ...     chunks = split_prompt(long_prompt)
        ...     for chunk in chunks:
        ...         client.generate(chunk)
    """
    
    def __init__(
        self,
        message: str = "Token limit exceeded. Please reduce input size.",
        limit: int | None = None,
        actual: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.limit = limit
        self.actual = actual


class AIModelNotFoundError(AIClientError):
    """Requested model doesn't exist or isn't available.
    
    This error is not retriable — the configuration needs to be fixed.
    
    Attributes:
        model_name: The model that was requested.
    
    Example:
        >>> try:
        ...     client = AIClient()
        ... except AIModelNotFoundError as e:
        ...     log.error(f"Model '{e.model_name}' not found, check config")
    """
    
    def __init__(
        self,
        model_name: str,
        message: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        msg = message or f"Model '{model_name}' not found. Check model name in configuration."
        super().__init__(msg, retriable=False, original_error=original_error)
        self.model_name = model_name


class AIContentBlockedError(AIClientError):
    """Content was blocked by safety filters.
    
    This error is not retriable with the same content.
    
    Attributes:
        block_reason: The reason for blocking if available.
    """
    
    def __init__(
        self,
        message: str = "Content blocked by safety filters.",
        block_reason: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, retriable=False, original_error=original_error)
        self.block_reason = block_reason


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
        generation_time_ms: Time taken for generation in milliseconds.
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
    generation_time_ms: float | None = Field(None, description="Generation time in ms")
    raw_response: Any = Field(None, exclude=True, description="Original SDK response")
    
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


class StructuredResponse(BaseModel):
    """Response when requesting structured/JSON output.
    
    Wraps both the parsed data and the raw text, along with parsing status.
    If JSON parsing fails, parse_success is False and parse_error contains
    the error message.
    
    Attributes:
        data: Parsed JSON content (empty dict if parsing failed).
        raw_text: Original text before parsing.
        model: Model that generated this response.
        tokens: Total tokens used.
        parse_success: Whether JSON parsing succeeded.
        parse_error: Error message if parsing failed.
    
    Example:
        >>> response = client.generate_structured("Extract entities...")
        >>> if response.parse_success:
        ...     entities = response.data["entities"]
        ... else:
        ...     log.warning(f"Parse failed: {response.parse_error}")
        ...     # Handle raw_text manually
    """
    
    data: dict[str, Any] = Field(default_factory=dict, description="Parsed JSON content")
    raw_text: str = Field(..., description="Original text before parsing")
    model: str = Field(..., description="Model that generated this response")
    tokens: int | None = Field(None, description="Total tokens used")
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
        >>> response = client.generate_structured(
        ...     prompt="Extract key events...",
        ...     schema_hint='{"events": [{"date": "...", "description": "..."}]}'
        ... )
        >>> for event in response.data.get("events", []):
        ...     print(f"{event['date']}: {event['description']}")
    
    Attributes:
        config: Application configuration.
    """
    
    def __init__(
        self,
        config: AppConfig | None = None,
        api_key: str | None = None,
        require_consent: bool | None = None,
    ) -> None:
        """Initialize the AI client.
        
        No API calls are made during initialization. The model is lazy-loaded
        on first use.
        
        Args:
            config: Application configuration. If None, loads from get_config().
            api_key: Override API key. If None, loads from configured sources.
            require_consent: Whether to require user consent before AI calls.
        
        Raises:
            AIUnavailableError: If google-generativeai package is not installed.
        """
        self._config = config or get_config()
        self._model: Any = None  # genai.GenerativeModel
        self._is_configured = False
        self._api_key: str | None = None
        # Use config setting if require_consent not explicitly set
        self._require_consent = require_consent if require_consent is not None else self._config.ai.require_consent
        self._logger = logging.getLogger(f"{__name__}.AIClient")
        
        # Check if SDK is available
        if not GENAI_AVAILABLE:
            self._logger.warning("google-generativeai package not installed")
            return
        
        # Don't configure if AI is disabled
        if not self._config.ai.is_enabled():
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
        
        # Configure the SDK
        try:
            genai.configure(api_key=self._api_key)
            self._is_configured = True
            self._logger.info(f"AI client configured with model: {self._config.ai.model_name}")
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
            raise AIUnavailableError("no_key")
        
        if self._require_consent and not has_consent():
            raise AIUnavailableError("no_consent")
    
    def _get_model(self) -> Any:
        """Get or create the GenerativeModel instance.
        
        The model is lazy-initialized and cached for reuse.
        
        Returns:
            Configured GenerativeModel instance.
        """
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self._config.ai.model_name,
                safety_settings=self._get_safety_settings(),
            )
        return self._model
    
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
        
        return GenerationConfig(**config_params)
    
    def _get_safety_settings(self) -> dict:
        """Get safety settings appropriate for life story analysis.
        
        We're analyzing personal memories, so we need balanced settings that
        allow discussion of personal content while blocking truly harmful content.
        
        Returns:
            Dictionary of safety settings.
        """
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        **overrides: Any,
    ) -> AIResponse:
        """Generate text from a prompt.
        
        This is the main generation method. It handles retries, error mapping,
        and response parsing automatically.
        
        Args:
            prompt: The user prompt to send to the model.
            system_instruction: Optional system instruction to guide the model.
            **overrides: Per-call overrides (temperature, max_output_tokens, etc.)
        
        Returns:
            AIResponse with the generated text and metadata.
        
        Raises:
            AIUnavailableError: If AI is disabled or not configured.
            AIAuthError: If authentication fails.
            AIRateLimitError: If rate limit is exceeded.
            AIServerError: If the server returns an error.
            AITimeoutError: If the request times out.
            AITokenLimitError: If token limits are exceeded.
        
        Example:
            >>> response = client.generate(
            ...     prompt="What patterns do you see?",
            ...     system_instruction="You are a life story analyst.",
            ...     temperature=0.5
            ... )
            >>> print(response.text)
        """
        self._ensure_available()
        
        start_time = time.time()
        
        # Build contents
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [system_instruction]})
            contents.append({"role": "model", "parts": ["Understood."]})
        contents.append({"role": "user", "parts": [prompt]})
        
        # Build generation config
        gen_config = self._get_generation_config(**overrides)
        
        # Make the request with retry
        try:
            raw_response = self._with_retry(
                self._do_generate,
                contents=contents,
                generation_config=gen_config,
            )
        except Exception as e:
            # Log failure (no prompt content)
            self._logger.error(
                f"Generation failed: {type(e).__name__}",
                extra={"model": self._config.ai.model_name},
            )
            raise
        
        # Parse response
        generation_time_ms = (time.time() - start_time) * 1000
        
        # Extract text from response
        try:
            text = raw_response.text
        except ValueError:
            # Response might be blocked
            if raw_response.prompt_feedback.block_reason:
                raise AIContentBlockedError(
                    block_reason=str(raw_response.prompt_feedback.block_reason)
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
                finish_reason = str(candidate.finish_reason.name) if candidate.finish_reason else None
        
        response = AIResponse(
            text=text,
            model=self._config.ai.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            generation_time_ms=generation_time_ms,
            raw_response=raw_response,
        )
        
        # Log success (no content)
        self._logger.info(
            f"Generation successful: {total_tokens or '?'} tokens in {generation_time_ms:.0f}ms",
            extra={
                "model": self._config.ai.model_name,
                "tokens": total_tokens,
                "time_ms": generation_time_ms,
            },
        )
        
        return response
    
    def generate_structured(
        self,
        prompt: str,
        system_instruction: str | None = None,
        schema_hint: str | None = None,
    ) -> StructuredResponse:
        """Generate structured JSON output.
        
        Modifies the prompt to request JSON output and parses the response.
        If parsing fails, returns StructuredResponse with parse_success=False.
        
        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.
            schema_hint: Optional JSON schema hint to include in prompt.
        
        Returns:
            StructuredResponse with parsed data or error information.
        
        Example:
            >>> response = client.generate_structured(
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
        response = self.generate(
            prompt=full_prompt,
            system_instruction=full_instruction,
        )
        
        # Try to parse JSON
        text = response.text.strip()
        data: dict[str, Any] = {}
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
                    parse_error = f"JSON parse error: {e.msg}"
            else:
                # Try to find JSON object/array in text
                json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        parse_success = True
                    except json.JSONDecodeError:
                        parse_error = f"JSON parse error: {e.msg}"
                else:
                    parse_error = f"JSON parse error: {e.msg}"
        
        return StructuredResponse(
            data=data,
            raw_text=response.text,
            model=response.model,
            tokens=response.total_tokens,
            parse_success=parse_success,
            parse_error=parse_error,
        )
    
    def _do_generate(
        self,
        contents: list,
        generation_config: "GenerationConfig",
    ) -> Any:
        """Execute the actual API call.
        
        This method is wrapped by retry logic.
        
        Args:
            contents: The conversation contents.
            generation_config: Generation configuration.
        
        Returns:
            Raw response from the SDK.
        """
        model = self._get_model()
        return model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(),
        )
    
    def _with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry on transient failures.
        
        Uses exponential backoff with jitter for retries.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The function's return value.
        
        Raises:
            AIClientError: On failure after all retries exhausted.
        """
        max_retries = self._config.ai.max_retries
        base_delay = self._config.ai.retry_base_delay
        last_error: AIClientError | None = None
        
        for attempt in range(max_retries + 1):
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
                if attempt >= max_retries:
                    self._logger.error(
                        f"Max retries ({max_retries}) exhausted: {type(mapped_error).__name__}"
                    )
                    raise mapped_error
                
                # Calculate delay with jitter
                delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                
                # Check rate limit hint
                if isinstance(mapped_error, AIRateLimitError) and mapped_error.retry_after_seconds:
                    total_delay = max(total_delay, mapped_error.retry_after_seconds)
                
                self._logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {total_delay:.1f}s: "
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
        error_type = type(error).__name__
        
        # Check for google.api_core exceptions
        if google_exceptions:
            if isinstance(error, google_exceptions.InvalidArgument):
                if "token" in error_str or "length" in error_str:
                    return AITokenLimitError(original_error=error)
                return AIBadRequestError(str(error), original_error=error)
            
            if isinstance(error, google_exceptions.PermissionDenied):
                return AIAuthError(original_error=error)
            
            if isinstance(error, google_exceptions.Unauthenticated):
                return AIAuthError(original_error=error)
            
            if isinstance(error, google_exceptions.ResourceExhausted):
                if "quota" in error_str:
                    return AIQuotaExceededError(original_error=error)
                return AIRateLimitError(original_error=error)
            
            if isinstance(error, google_exceptions.NotFound):
                # Try to extract model name
                model_match = re.search(r"model[s]?[:/\s]+([^\s,]+)", error_str)
                model_name = model_match.group(1) if model_match else self._config.ai.model_name
                return AIModelNotFoundError(model_name, original_error=error)
            
            if isinstance(error, google_exceptions.DeadlineExceeded):
                return AITimeoutError(
                    self._config.ai.timeout_seconds,
                    original_error=error,
                )
            
            if isinstance(error, google_exceptions.InternalServerError):
                return AIServerError(status_code=500, original_error=error)
            
            if isinstance(error, google_exceptions.ServiceUnavailable):
                return AIServerError(status_code=503, original_error=error)
        
        # Fallback pattern matching on error message
        if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            return AIAuthError(original_error=error)
        
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
            return AITokenLimitError(original_error=error)
        
        if "model" in error_str and "not found" in error_str:
            return AIModelNotFoundError(self._config.ai.model_name, original_error=error)
        
        # Generic fallback
        return AIClientError(str(error), retriable=False, original_error=error)
    
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
                return True, f"Connected to {self._config.ai.model_name}"
            return False, "Empty response from API"
        except AIClientError as e:
            return False, e.message
        except Exception as e:
            return False, str(e)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a simple heuristic (characters / 4). For more accurate counts,
        the model's count_tokens method could be used, but that requires
        an API call.
        
        Args:
            text: The text to estimate tokens for.
        
        Returns:
            Estimated token count.
        """
        # Simple heuristic: ~4 characters per token on average
        return len(text) // 4
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model.
        
        Returns:
            Dictionary with model name and limits.
        """
        return {
            "name": self._config.ai.model_name,
            "max_output_tokens": self._config.ai.max_output_tokens,
            "temperature": self._config.ai.temperature,
            "timeout_seconds": self._config.ai.timeout_seconds,
            "is_available": self.is_available(),
        }


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
        raise AIUnavailableError("no_key")
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
            raise AIUnavailableError("no_key")
        
        return func(*args, **kwargs)
    
    return wrapper  # type: ignore
