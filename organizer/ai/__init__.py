"""AI module for Digital Life Narrative AI.

This package provides AI-powered analysis capabilities using Google's Gemini API.

Exports:
    AIClient: Low-level Gemini API client
    AIResponse: Response data class
    LifeStoryAnalyzer: Core life story analysis engine
    FallbackAnalyzer: Statistics-only analyzer when AI unavailable
    Exception classes: AIClientError, RateLimitError, etc.
    get_client: Convenience function to get configured client
"""

from organizer.ai.client import (
    AIClient,
    AIClientError,
    AIRequestError,
    AIResponse,
    APIKeyMissingError,
    ModelNotAvailableError,
    RateLimitError,
    TokenLimitExceededError,
    check_api_key_configured,
    estimate_tokens,
    get_client,
)
from organizer.ai.fallback import (
    FallbackAnalyzer,
    generate_fallback_report,
    is_fallback_mode,
)
from organizer.ai.life_analyzer import (
    AINotAvailableError,
    AnalysisError,
    InsufficientDataError,
    LifeStoryAnalyzer,
)

__all__ = [
    # Client
    "AIClient",
    "AIResponse",
    "get_client",
    # Analyzers
    "LifeStoryAnalyzer",
    "FallbackAnalyzer",
    # Exceptions
    "AIClientError",
    "APIKeyMissingError",
    "RateLimitError",
    "ModelNotAvailableError",
    "TokenLimitExceededError",
    "AIRequestError",
    "AnalysisError",
    "InsufficientDataError",
    "AINotAvailableError",
    # Utilities
    "estimate_tokens",
    "check_api_key_configured",
    "generate_fallback_report",
    "is_fallback_mode",
]
