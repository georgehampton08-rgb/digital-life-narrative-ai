"""Digital Life Narrative AI - Reconstruct your life story from media exports.

An AI-first universal data dump organizer that weaves your scattered photos
and videos into a meaningful narrative using Google's Gemini AI.

Quick Start:
    >>> from organizer import analyze
    >>> report = analyze(["~/Downloads/takeout"], "./my_story")
    >>> print(report.executive_summary)

CLI Usage:
    $ organizer config set-key      # Configure Gemini API key
    $ organizer analyze -i ~/Photos -o ./my_story
    $ organizer scan ~/Downloads/takeout
"""

__version__ = "0.1.0"
__author__ = "George Hampton"
__email__ = "georgehampton08@gmail.com"

from organizer.models import (
    AnalysisConfig,
    Confidence,
    DataGap,
    GeoLocation,
    LifeChapter,
    LifeStoryReport,
    MediaItem,
    MediaType,
    ParseResult,
    PlatformBehaviorInsight,
    SourcePlatform,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    # Models
    "MediaItem",
    "MediaType",
    "SourcePlatform",
    "Confidence",
    "GeoLocation",
    "ParseResult",
    "LifeChapter",
    "LifeStoryReport",
    "PlatformBehaviorInsight",
    "DataGap",
    "AnalysisConfig",
]
