"""Logging configuration for Digital Life Narrative AI.

Provides centralized logging setup with Rich console formatting
and optional file logging.

Example:
    >>> from organizer.utils.logging import setup_logging, get_logger, LogContext
    >>> setup_logging(level="DEBUG")
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting analysis")
    >>> with LogContext("Parsing export"):
    ...     # do work
    ... # Logs: "Parsing export completed in 2.3s"
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from rich.console import Console
from rich.logging import RichHandler


# =============================================================================
# Constants
# =============================================================================

# Package logger name
PACKAGE_NAME = "organizer"

# Noisy third-party loggers to filter
NOISY_LOGGERS = [
    "google",
    "google.auth",
    "google.api_core",
    "google.generativeai",
    "urllib3",
    "httpx",
    "httpcore",
    "PIL",
    "asyncio",
]

# Log format for file handler
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console for Rich handler
_console = Console(stderr=True)


# =============================================================================
# Setup Functions
# =============================================================================


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    quiet_third_party: bool = True,
) -> None:
    """Configure logging for the organizer package.

    Sets up Rich console handler for pretty output and optionally
    a file handler for persistent logs.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        quiet_third_party: If True, suppress noisy third-party loggers.

    Example:
        >>> setup_logging(level="DEBUG", log_file=Path("./organizer.log"))
    """
    # Parse level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger for our package
    root_logger = logging.getLogger(PACKAGE_NAME)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Rich console handler
    console_handler = RichHandler(
        console=_console,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=numeric_level == logging.DEBUG,
        markup=True,
    )
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers
    if quiet_third_party:
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Don't propagate to root logger
    root_logger.propagate = False

    # Log startup message at debug level
    root_logger.debug(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Uses the package logger as parent to inherit configuration.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing file")
    """
    # Ensure it's under our package namespace
    if not name.startswith(PACKAGE_NAME):
        name = f"{PACKAGE_NAME}.{name}"
    return logging.getLogger(name)


# =============================================================================
# Log Context Manager
# =============================================================================


class LogContext:
    """Context manager for timing and logging operations.

    Logs the start and completion of an operation with elapsed time.

    Attributes:
        message: Description of the operation.
        level: Log level for messages.
        logger: Logger instance to use.
        elapsed: Elapsed time in seconds (after exit).

    Example:
        >>> with LogContext("Parsing Snapchat export") as ctx:
        ...     # do work
        ...     pass
        # Logs: "Parsing Snapchat export..."
        # Logs: "Parsing Snapchat export completed in 2.34s"
    """

    def __init__(
        self,
        message: str,
        level: int = logging.INFO,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the log context.

        Args:
            message: Description of the operation.
            level: Log level (default: INFO).
            logger: Logger to use (default: package logger).
        """
        self.message = message
        self.level = level
        self.logger = logger or logging.getLogger(PACKAGE_NAME)
        self.elapsed: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> "LogContext":
        """Enter the context and log start message."""
        self._start_time = time.perf_counter()
        self.logger.log(self.level, f"{self.message}...")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context and log completion/error."""
        self.elapsed = time.perf_counter() - self._start_time

        if exc_type is not None:
            self.logger.error(
                f"{self.message} failed after {self.elapsed:.2f}s: {exc_val}"
            )
        else:
            self.logger.log(
                self.level,
                f"{self.message} completed in {self.elapsed:.2f}s"
            )


@contextmanager
def log_context(
    message: str,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
) -> Generator[LogContext, None, None]:
    """Functional context manager for timing operations.

    This is an alternative to the LogContext class for those who
    prefer the @contextmanager style.

    Args:
        message: Description of the operation.
        level: Log level (default: INFO).
        logger: Logger to use (default: package logger).

    Yields:
        LogContext instance with elapsed time.

    Example:
        >>> with log_context("Processing files") as ctx:
        ...     # do work
        ...     pass
        >>> print(f"Took {ctx.elapsed:.2f}s")
    """
    ctx = LogContext(message, level, logger)
    with ctx:
        yield ctx


# =============================================================================
# Utility Functions
# =============================================================================


def set_level(level: str) -> None:
    """Change the log level at runtime.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def add_file_handler(log_file: Path) -> None:
    """Add a file handler to the existing logger.

    Args:
        log_file: Path to log file.
    """
    logger = logging.getLogger(PACKAGE_NAME)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logger.level)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
