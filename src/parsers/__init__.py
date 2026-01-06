"""Parsers package for Digital Life Narrative AI.

This package contains parsers for various platform exports (Snapchat, Google Photos, etc.).
All parsers inherit from BaseParser and register with ParserRegistry.

Exports:
    - BaseParser: Abstract base class for all parsers
    - ParserRegistry: Registry for parser discovery
    - ParseResult: Result of parsing operation
    - ParseStatus: Status enum
    - register_parser: Decorator for registering parsers
"""

from src.parsers.base import (
    # Base class
    BaseParser,
    # Registry
    ParserRegistry,
    register_parser,
    parse_directory,
    # Data models
    ParseResult,
    ParseStatus,
    ParseWarning,
    ParseError,
    ParseProgress,
    # Type aliases
    ProgressCallback,
    # Exceptions
    ParserError,
    ParserNotFoundError,
    ParseAbortedError,
)

__all__ = [
    "BaseParser",
    "ParserRegistry",
    "register_parser",
    "parse_directory",
    "ParseResult",
    "ParseStatus",
    "ParseWarning",
    "ParseError",
    "ParseProgress",
    "ProgressCallback",
    "ParserError",
    "ParserNotFoundError",
    "ParseAbortedError",
]
