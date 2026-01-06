"""Parser package for Digital Life Narrative AI.

This package contains platform-specific parsers that normalize
export data into MediaItem objects.

Exports:
    BaseParser: Abstract base class for all parsers
    ParserRegistry: Registry for parser lookup
    ParserError: Parser-specific exception
    parse_all_sources: High-level multi-source parsing
    parse_single_path: Single path parsing convenience function
"""

from organizer.parsers.base import (
    BaseParser,
    ParserError,
    ParserRegistry,
    parse_all_sources,
    parse_single_path,
)

__all__ = [
    "BaseParser",
    "ParserError",
    "ParserRegistry",
    "parse_all_sources",
    "parse_single_path",
]

# Import parsers to trigger registration
# These imports must be after the base imports to avoid circular dependencies
from organizer.parsers import snapchat  # noqa: F401, E402
from organizer.parsers import google_photos  # noqa: F401, E402
from organizer.parsers import local  # noqa: F401, E402
