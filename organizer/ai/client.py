"""Low-level Gemini API client for Digital Life Narrative AI.

This module provides the AIClient class for interacting with Google's Gemini API.
Handles authentication, request/response formatting, rate limiting, and retries.

The higher-level LifeStoryAnalyzer will use this client for AI-powered analysis.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import ClientError as AIClientError
from google.api_core import exceptions as google_exceptions

from organizer.config import (
    AISettings,
    APIKeyManager,
    APIKeyNotFoundError,
    KeyStorageBackend,
    get_config,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class AIClientError(Exception):
    """Base exception for AI client errors."""

    pass


class APIKeyMissingError(AIClientError):
    """Raised when API key is not configured."""

    pass


class RateLimitError(AIClientError):
    """Raised when rate limit is exceeded.

    Attributes:
        retry_after: Suggested wait time in seconds, if provided by API.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelNotAvailableError(AIClientError):
    """Raised when the requested model is not available."""

    pass


class TokenLimitExceededError(AIClientError):
    """Raised when the request exceeds token limits."""

    pass


class AIRequestError(AIClientError):
    """Generic AI request failure."""

    pass


# =============================================================================
# Response Data Class
# =============================================================================


@dataclass
class AIResponse:
    """Response from an AI generation request.

    Attributes:
        text: The generated text response.
        model: The model that generated the response.
        prompt_tokens: Number of tokens in the prompt (if available).
        completion_tokens: Number of tokens in the completion (if available).
        finish_reason: Why generation stopped (if available).
        raw_response: The original response object from the API.
    """

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw_response: Any = field(default=None, repr=False)


# =============================================================================
# AI Client
# =============================================================================


class AIClient:
    """Client for interacting with Google's Gemini API.

    Handles authentication, request/response formatting, rate limiting,
    and retries with exponential backoff.

    Attributes:
        settings: AI configuration settings.
        model_name: Name of the Gemini model to use.

    Example:
        ```python
        client = AIClient()
        response = client.generate("Tell me a story about...")
        print(response.text)
        ```
    """

    # Default safety settings for life story analysis
    # Allows personal content while maintaining reasonable safety
    # Default safety settings for life story analysis
    # Allows personal content while maintaining reasonable safety
    DEFAULT_SAFETY_SETTINGS = [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_ONLY_HIGH",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_MEDIUM_AND_ABOVE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        settings: AISettings | None = None,
    ) -> None:
        """Initialize the AI client.

        Args:
            api_key: Gemini API key. If None, attempts to retrieve from APIKeyManager.
            settings: AI configuration settings. Uses defaults if None.

        Raises:
            APIKeyMissingError: If no API key is available.
        """
        self.settings = settings or AISettings()
        self._api_key = api_key or self._retrieve_api_key()

        if not self._api_key:
            raise APIKeyMissingError("No API key available. Configure with 'organizer configure'.")

        # Initialize the new genai client
        self._client = genai.Client(api_key=self._api_key)

        logger.debug(f"AI client initialized with model: {self.settings.model_name}")

    def _retrieve_api_key(self) -> str | None:
        """Attempt to retrieve API key from configured storage.

        Returns:
            API key string or None if not found.
        """
        try:
            config = get_config()
            manager = APIKeyManager(
                config.key_storage_backend,
                config.encrypted_key_file_path,
            )
            return manager.retrieve_key()
        except (APIKeyNotFoundError, Exception) as e:
            logger.debug(f"Could not retrieve API key: {e}")
            return None

    def _get_config(self, system_instruction: str | None = None, temperature: float | None = None, max_tokens: int | None = None) -> types.GenerateContentConfig:
        """Get generation config."""
        return types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.settings.max_tokens,
            safety_settings=self.DEFAULT_SAFETY_SETTINGS,
            system_instruction=system_instruction,
        )

    # =========================================================================
    # Public Methods
    # =========================================================================

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIResponse:
        """Generate a response from the AI model.

        Args:
            prompt: The user prompt to send to the model.
            system_instruction: Optional system instruction to guide the model.
            temperature: Override default temperature (0.0-2.0).
            max_tokens: Override default max tokens.

        Returns:
            AIResponse containing the generated text and metadata.

        Raises:
            AIRequestError: If the request fails after retries.
            TokenLimitExceededError: If the prompt exceeds token limits.
            RateLimitError: If rate limit is exceeded and retries exhausted.
        """
        # Get config with overrides
        config = self._get_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        def do_generate():
            return self._client.models.generate_content(
                model=self.settings.model_name,
                contents=prompt,
                config=config,
            )

        try:
            response = self._with_retry(do_generate)
            return self._parse_response(response)

        except AIClientError:
            raise
        except Exception as e:
            self._handle_api_error(e)
            raise  # _handle_api_error always raises, but this satisfies type checker

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response from the AI model.

        Instructs the model to output valid JSON and parses the response.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            AIRequestError: If generation fails or JSON parsing fails.
        """
        # Add JSON instruction to system prompt
        json_instruction = (
            "You must respond with valid JSON only. "
            "Do not include any text before or after the JSON. "
            "Do not use markdown code blocks."
        )

        if system_instruction:
            full_instruction = f"{system_instruction}\n\n{json_instruction}"
        else:
            full_instruction = json_instruction

        response = self.generate(
            prompt=prompt,
            system_instruction=full_instruction,
            temperature=0.3,  # Lower temperature for structured output
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            text = response.text.strip()

            # Handle markdown code blocks if present despite instruction
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response.text}")
            raise AIRequestError(f"Failed to parse JSON response: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        try:
            result = self._client.models.count_tokens(
                model=self.settings.model_name,
                contents=text
            )
            return result.total_tokens
        except Exception as e:
            logger.debug(f"Token counting failed, using estimate: {e}")
            return estimate_tokens(text)

    def is_available(self) -> bool:
        """Check if the AI client is available and configured.

        Makes a minimal test to verify API connectivity.

        Returns:
            True if client is ready to use.
        """
        try:
            # Try to count tokens as a minimal API test
            self.count_tokens("test")
            return True
        except Exception as e:
            logger.debug(f"AI client availability check failed: {e}")
            return False

    @property
    def model_name(self) -> str:
        """Get the name of the configured model."""
        return self.settings.model_name

    # =========================================================================
    # Private Methods
    # =========================================================================

    # _get_model_with_overrides removed in favor of _get_config

    def _with_retry(
        self,
        func: Callable[[], T],
        max_retries: int | None = None,
    ) -> T:
        """Execute a function with retry logic and exponential backoff.

        Retries on:
        - Rate limits (429)
        - Temporary server errors (500-503)

        Does NOT retry on:
        - Auth errors (401/403)
        - Bad request (400)
        - Token limit exceeded

        Args:
            func: Function to execute.
            max_retries: Maximum retry attempts. Uses settings default if None.

        Returns:
            Result of the function.

        Raises:
            Appropriate AIClientError subclass on failure.
        """
        retries = max_retries if max_retries is not None else self.settings.max_retries
        last_exception: Exception | None = None

        for attempt in range(retries + 1):
            try:
                return func()

            except google_exceptions.ResourceExhausted as e:
                # Rate limit - retry with backoff
                last_exception = e
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{retries + 1})"
                )

                if attempt < retries:
                    time.sleep(wait_time)
                    continue

                raise RateLimitError(
                    f"Rate limit exceeded after {retries + 1} attempts",
                    retry_after=wait_time,
                )

            except google_exceptions.ServiceUnavailable as e:
                # Temporary server error - retry
                last_exception = e
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"Service unavailable, waiting {wait_time:.1f}s (attempt {attempt + 1}/{retries + 1})"
                )

                if attempt < retries:
                    time.sleep(wait_time)
                    continue

                raise AIRequestError(f"Service unavailable after {retries + 1} attempts: {e}")

            except google_exceptions.InternalServerError as e:
                # Server error - retry
                last_exception = e
                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"Server error, waiting {wait_time:.1f}s (attempt {attempt + 1}/{retries + 1})"
                )

                if attempt < retries:
                    time.sleep(wait_time)
                    continue

                raise AIRequestError(f"Server error after {retries + 1} attempts: {e}")

            except google_exceptions.InvalidArgument as e:
                # Bad request - don't retry
                error_msg = str(e).lower()
                if "token" in error_msg or "limit" in error_msg:
                    raise TokenLimitExceededError(f"Token limit exceeded: {e}")
                raise AIRequestError(f"Invalid request: {e}")

            except google_exceptions.PermissionDenied as e:
                # Auth error - don't retry
                raise APIKeyMissingError(f"Permission denied - check API key: {e}")

            except google_exceptions.Unauthenticated as e:
                # Auth error - don't retry
                raise APIKeyMissingError(f"Authentication failed - check API key: {e}")

            except google_exceptions.NotFound as e:
                # Model not found - don't retry
                raise ModelNotAvailableError(f"Model not available: {e}")

            except Exception as e:
                # Unknown error - convert and raise
                self._handle_api_error(e)

        # Should not reach here, but just in case
        if last_exception:
            raise AIRequestError(f"Request failed: {last_exception}")
        raise AIRequestError("Request failed for unknown reason")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Wait time in seconds.
        """
        base_delay = 1.0
        max_delay = 60.0

        # Exponential backoff: 1s, 2s, 4s, 8s, ...
        delay = base_delay * (2**attempt)

        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay = delay + jitter

        return min(delay, max_delay)

    def _parse_response(self, response: Any) -> AIResponse:
        """Parse response."""
        if not response.candidates:
            raise AIRequestError("Empty response from API")

        candidate = response.candidates[0]
        
        # Check for safety blocks or other errors
        if candidate.finish_reason == "SAFETY":
            raise AIRequestError("Response blocked due to safety settings")

        # Extract text
        text = response.text or ""
        
        return AIResponse(
            text=text,
            model=self.settings.model_name,
            prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else None,
            completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else None,
            finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
            raw_response=response,
        )

    def _handle_api_error(self, error: Exception) -> None:
        """Map API exceptions to our error hierarchy."""
        error_str = str(error).lower()

        if "quota" in error_str or "429" in error_str:
            raise RateLimitError(f"Rate limit or quota exceeded: {error}")

        if "token" in error_str and "limit" in error_str:
            raise TokenLimitExceededError(f"Token limit exceeded: {error}")

        if "401" in error_str or "unauthenticated" in error_str:
            raise APIKeyMissingError(f"API key error: {error}")

        if "404" in error_str or "not found" in error_str:
            raise ModelNotAvailableError(f"Model not available: {error}")

        raise AIRequestError(f"AI request failed: {error}")


# =============================================================================
# Module-Level Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a simple heuristic: approximately 4 characters per token.
    This is a rough estimate and should only be used when the
    actual token counting API is unavailable.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    # Gemini uses similar tokenization to GPT models
    # Rough estimate: ~4 chars per token for English text
    return max(1, len(text) // 4)


def get_client(
    api_key: str | None = None,
    settings: AISettings | None = None,
) -> AIClient:
    """Get a configured AI client instance.

    Convenience function for obtaining an AIClient with default or
    custom configuration.

    Args:
        api_key: Optional API key override.
        settings: Optional settings override.

    Returns:
        Configured AIClient instance.

    Raises:
        APIKeyMissingError: If no API key is available.
    """
    return AIClient(api_key=api_key, settings=settings)


def check_api_key_configured() -> bool:
    """Check if an API key is configured without initializing a client.

    Returns:
        True if an API key is configured.
    """
    try:
        config = get_config()
        manager = APIKeyManager(
            config.key_storage_backend,
            config.encrypted_key_file_path,
        )
        return manager.is_key_configured()
    except Exception:
        return False
