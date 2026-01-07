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

import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentResponse,
    HarmBlockThreshold,
    HarmCategory,
)
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
    DEFAULT_SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

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

        # Configure the genai library
        genai.configure(api_key=self._api_key)

        # Initialize the model
        self._model = self._configure_model()

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

    def _configure_model(self) -> genai.GenerativeModel:
        """Configure and return the Gemini model.

        Returns:
            Configured GenerativeModel instance.
        """
        generation_config = genai.GenerationConfig(
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=self.settings.model_name,
            generation_config=generation_config,
            safety_settings=self.DEFAULT_SAFETY_SETTINGS,
        )

        return model

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
        # Create model with overrides if specified
        model = self._get_model_with_overrides(
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        def do_generate() -> GenerateContentResponse:
            return model.generate_content(prompt)

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
        """Count tokens in a text string.

        Uses Gemini's token counting API if available.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        try:
            result = self._model.count_tokens(text)
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

    def _get_model_with_overrides(
        self,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> genai.GenerativeModel:
        """Get a model instance with parameter overrides.

        Args:
            system_instruction: Optional system instruction.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.

        Returns:
            Configured GenerativeModel.
        """
        # Determine generation config
        gen_config = genai.GenerationConfig(
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.settings.max_tokens,
        )

        return genai.GenerativeModel(
            model_name=self.settings.model_name,
            generation_config=gen_config,
            safety_settings=self.DEFAULT_SAFETY_SETTINGS,
            system_instruction=system_instruction,
        )

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

    def _parse_response(self, response: GenerateContentResponse) -> AIResponse:
        """Parse a Gemini response into our AIResponse format.

        Args:
            response: Raw Gemini response.

        Returns:
            Parsed AIResponse.

        Raises:
            AIRequestError: If response is blocked or empty.
        """
        # Check for blocked response
        if not response.candidates:
            if response.prompt_feedback:
                raise AIRequestError(f"Response blocked: {response.prompt_feedback}")
            raise AIRequestError("Empty response from API")

        candidate = response.candidates[0]

        # Check finish reason
        finish_reason = None
        if hasattr(candidate, "finish_reason"):
            finish_reason = str(candidate.finish_reason.name) if candidate.finish_reason else None

            # Check for safety blocks
            if finish_reason == "SAFETY":
                raise AIRequestError("Response blocked due to safety settings")

        # Extract text
        try:
            text = response.text
        except ValueError as e:
            raise AIRequestError(f"Could not extract text from response: {e}")

        # Extract token counts if available
        prompt_tokens = None
        completion_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            prompt_tokens = getattr(metadata, "prompt_token_count", None)
            completion_tokens = getattr(metadata, "candidates_token_count", None)

        return AIResponse(
            text=text,
            model=self.settings.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            raw_response=response,
        )

    def _handle_api_error(self, error: Exception) -> None:
        """Map API exceptions to our error hierarchy.

        Args:
            error: The caught exception.

        Raises:
            Appropriate AIClientError subclass.
        """
        error_str = str(error).lower()

        # Check for common error patterns
        if "quota" in error_str or "rate" in error_str:
            raise RateLimitError(f"Rate limit or quota exceeded: {error}")

        if "token" in error_str and "limit" in error_str:
            raise TokenLimitExceededError(f"Token limit exceeded: {error}")

        if "api key" in error_str or "authentication" in error_str:
            raise APIKeyMissingError(f"API key error: {error}")

        if "model" in error_str and ("not found" in error_str or "unavailable" in error_str):
            raise ModelNotAvailableError(f"Model not available: {error}")

        # Generic error
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
