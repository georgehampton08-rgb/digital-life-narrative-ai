"""AI module for Digital Life Narrative AI.

This module provides the interface to Google's Gemini LLM for AI-powered
analysis of life memories. The client.py module is the SOLE interface to
the Gemini API — no other file should import google-generativeai.

Exports:
    - AIClient: Main client for Gemini API communication
    - LifeStoryAnalyzer: Orchestrates Memory → AI → Report flow
    - get_client: Factory function to create configured client
    - require_ai: Decorator for functions requiring AI
    - AIResponse: Standardized response model
    - StructuredResponse: Response model for JSON outputs
    - LifeStoryReport: Complete analysis report
    - LifeChapter: Individual life chapter
    - Exception hierarchy for typed error handling
"""

from src.ai.analyzer import (
    # Analyzer
    LifeStoryAnalyzer,
    AnalysisProgress,
    # Report models
    LifeStoryReport,
    LifeChapter,
    PlatformBehaviorInsight,
    DataGap,
    # Config
    AnalysisConfig,
    # Exceptions
    AnalysisError,
    InsufficientDataError,
)
from src.ai.client import (
    # Main client
    AIClient,
    get_client,
    require_ai,
    # Consent management
    request_consent,
    has_consent,
    revoke_consent,
    grant_consent_programmatic,
    # Response models
    AIResponse,
    StructuredResponse,
    # Exceptions
    AIClientError,
    AIUnavailableError,
    AIAuthError,
    AIRateLimitError,
    AIQuotaExceededError,
    AIServerError,
    AIBadRequestError,
    AITimeoutError,
    AITokenLimitError,
    AIModelNotFoundError,
    AIContentBlockedError,
)

__all__ = [
    # Analyzer
    "LifeStoryAnalyzer",
    "AnalysisProgress",
    "LifeStoryReport",
    "LifeChapter",
    "PlatformBehaviorInsight",
    "DataGap",
    "AnalysisConfig",
    "AnalysisError",
    "InsufficientDataError",
    # Client
    "AIClient",
    "get_client",
    "require_ai",
    # Consent
    "request_consent",
    "has_consent",
    "revoke_consent",
    "grant_consent_programmatic",
    # Responses
    "AIResponse",
    "StructuredResponse",
    # Exceptions
    "AIClientError",
    "AIUnavailableError",
    "AIAuthError",
    "AIRateLimitError",
    "AIQuotaExceededError",
    "AIServerError",
    "AIBadRequestError",
    "AITimeoutError",
    "AITokenLimitError",
    "AIModelNotFoundError",
    "AIContentBlockedError",
]
