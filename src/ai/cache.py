"""Machine-Local AI Analysis Cache.

This module provides a local caching layer for AI analysis results.
Cached results are:
- Bound to the specific machine (via machine_id)
- Invalidated when media content changes
- Invalidated when analysis configuration changes
- Safe to delete at any time (triggers recomputation)
- Never committed to version control

Cache entries are stored as JSON files with metadata for validation.
The cache is a pure optimization — the system works identically without it.

Security notes:
- No raw images or videos are cached
- No API keys or secrets are cached
- No full file paths are stored
- Captions are not cached in full

Example:
    >>> from src.ai.cache import AICache, fingerprint_media_set, fingerprint_analysis_config
    >>>
    >>> # Create cache instance
    >>> cache = AICache()
    >>>
    >>> # Compute fingerprints
    >>> media_fp = fingerprint_media_set(memories)
    >>> config_fp = fingerprint_analysis_config(analysis_config)
    >>>
    >>> # Check for cached result
    >>> cache_key = cache.build_cache_key(media_fp, config_fp, "full_report")
    >>> cached = cache.load(cache_key, media_fp, config_fp)
    >>>
    >>> if cached is not None:
    ...     print("Cache hit!")
    ... else:
    ...     print("Cache miss, running analysis...")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.ai.analyzer import AnalysisConfig
    from src.config import PrivacyConfig
    from src.core.memory import Memory

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class CacheMeta(BaseModel):
    """Metadata stored alongside each cached result.

    This metadata is used to validate cache entries and ensure they
    match the current machine, media set, and configuration.

    Attributes:
        created_at: Unix timestamp when cache entry was created.
        machine_id: Identifies the machine that created this cache.
        media_set_fingerprint: Hash of the input media set.
        analysis_config_fingerprint: Hash of analysis configuration.
        version: Cache schema version (e.g., "1.0").
        item_count: Number of media items, for quick sanity check.
    """

    created_at: float = Field(description="Unix timestamp when cache entry was created")
    machine_id: str = Field(description="Machine identifier hash")
    media_set_fingerprint: str = Field(description="Hash of input media set")
    analysis_config_fingerprint: str = Field(description="Hash of analysis config")
    version: str = Field(default="1.0", description="Cache schema version")
    item_count: int = Field(default=0, description="Number of media items")


class CacheEntry(BaseModel):
    """The complete structure written to disk.

    Contains both the validation metadata and the actual cached payload.

    Attributes:
        meta: Cache metadata for validation.
        payload: The cached AI results as a dictionary.
    """

    meta: CacheMeta
    payload: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Helper Functions
# =============================================================================


def get_machine_id() -> str:
    """Generate a stable identifier unique to this machine/user combination.

    Combines hostname and user home directory, then hashes to create
    a stable identifier. This ensures cache entries are tied to the
    specific machine where they were created.

    Returns:
        First 16 hex characters of SHA-256 hash.

    Security:
        Never logs or stores the raw hostname/path, only the hash.

    Example:
        >>> machine_id = get_machine_id()
        >>> len(machine_id)
        16
    """
    try:
        hostname = platform.node() or "unknown"
        home_path = str(Path.home())

        # Combine into a single string
        combined = f"{hostname}:{home_path}"

        # Hash with SHA-256
        hash_digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        # Return first 16 characters
        return hash_digest[:16]
    except Exception as e:
        # Fallback for edge cases
        logger.debug(f"Machine ID generation fallback: {type(e).__name__}")
        return hashlib.sha256(b"fallback-machine-id").hexdigest()[:16]


def fingerprint_media_set(items: list["Memory"]) -> str:
    """Create a deterministic fingerprint of the media collection.

    Sorts items by stable keys and hashes the result to create a
    fingerprint that changes when the media set changes.

    Args:
        items: List of Memory objects to fingerprint.

    Returns:
        Full SHA-256 hex digest (64 characters).

    Critical rules:
        - Does NOT include full file paths (privacy)
        - Does NOT include full captions (privacy)
        - DOES include content hashes if available (detect changes)
        - Sorting ensures same items in different order = same fingerprint

    Example:
        >>> fingerprint = fingerprint_media_set(memories)
        >>> len(fingerprint)
        64
    """
    if not items:
        return hashlib.sha256(b"empty-media-set").hexdigest()

    # Build list of tuples for each item
    item_tuples = []

    for item in items:
        # Use created_at isoformat if available
        if item.created_at:
            timestamp_str = item.created_at.isoformat()
        else:
            timestamp_str = ""

        # Platform value
        platform_str = item.source_platform.value if item.source_platform else ""

        # Content hash or fallback
        if item.content_hash:
            content_id = item.content_hash
        else:
            # Fallback: truncated filename + item id
            filename = item.source_filename or ""
            content_id = f"{filename[-20:]}"

        item_tuples.append((timestamp_str, platform_str, content_id))

    # Sort for deterministic ordering
    item_tuples.sort()

    # Join all tuples into a single string
    combined = "|".join(f"{ts}:{plat}:{cid}" for ts, plat, cid in item_tuples)

    # Hash with SHA-256
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def fingerprint_analysis_config(
    config: "AnalysisConfig",
    privacy: "PrivacyConfig | None" = None,
) -> str:
    """Create a deterministic fingerprint of analysis settings.

    Extracts relevant fields and hashes them to create a fingerprint
    that changes when configuration changes.

    Args:
        config: Analysis configuration to fingerprint.
        privacy: Optional privacy settings (affects AI prompts).

    Returns:
        Full SHA-256 hex digest (64 characters).

    Note:
        Privacy settings are included because different privacy levels
        may produce different AI prompts/results.

    Example:
        >>> fingerprint = fingerprint_analysis_config(config, privacy)
        >>> len(fingerprint)
        64
    """
    # Extract relevant fields into a dict
    config_dict = {
        "min_chapters": config.min_chapters,
        "max_chapters": config.max_chapters,
        "min_chapter_duration_days": config.min_chapter_duration_days,
        "include_platform_analysis": config.include_platform_analysis,
        "include_gap_analysis": config.include_gap_analysis,
        "narrative_style": config.narrative_style,
        "privacy_level": config.privacy_level,
    }

    # Add privacy mode if provided
    if privacy is not None:
        config_dict["privacy_mode"] = (
            privacy.mode.value if hasattr(privacy.mode, "value") else str(privacy.mode)
        )

    # Serialize to JSON with sorted keys for determinism
    json_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))

    # Hash with SHA-256
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def _get_default_cache_dir() -> Path:
    """Get the platform-appropriate default cache directory.

    Returns:
        Path to the default cache directory:
        - macOS: ~/Library/Caches/life-story-reconstructor
        - Linux: ~/.cache/life-story-reconstructor
        - Windows: %LOCALAPPDATA%/life-story-reconstructor/cache
    """
    system = platform.system().lower()

    if system == "darwin":  # macOS
        return Path.home() / "Library" / "Caches" / "life-story-reconstructor"
    elif system == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "life-story-reconstructor" / "cache"
        else:
            return Path.home() / "AppData" / "Local" / "life-story-reconstructor" / "cache"
    else:  # Linux and others
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "life-story-reconstructor"
        else:
            return Path.home() / ".cache" / "life-story-reconstructor"


# =============================================================================
# Main Cache Class
# =============================================================================


class AICache:
    """Local file-based cache for AI analysis results.

    Provides persistent caching of AI analysis results to avoid
    redundant API calls. The cache is machine-specific and includes
    validation to ensure cached results match current inputs.

    Attributes:
        _cache_dir: Directory where cache files are stored.
        _enabled: Whether caching is active.
        _version: Cache schema version.
        _machine_id: Stable identifier for this machine.
        _logger: Logger instance.

    Example:
        >>> cache = AICache()
        >>>
        >>> # Build cache key
        >>> key = cache.build_cache_key(media_fp, config_fp, "full_report")
        >>>
        >>> # Try to load cached result
        >>> result = cache.load(key, media_fp, config_fp)
        >>>
        >>> if result is None:
        ...     # Cache miss - run analysis
        ...     result = run_analysis()
        ...     # Store for next time
        ...     cache.store(key, meta, result)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        enabled: bool = True,
        version: str = "1.0",
    ) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. If None, uses platform default.
            enabled: Whether to enable caching. If False, all operations no-op.
            version: Cache schema version. Changes invalidate old entries.
        """
        self._enabled = enabled
        self._version = version
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Pre-compute machine ID
        self._machine_id = get_machine_id()

        # Determine cache directory
        if cache_dir is None:
            self._cache_dir = _get_default_cache_dir()
        else:
            self._cache_dir = Path(cache_dir).expanduser().resolve()

        # Try to create cache directory
        if self._enabled:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)

                # Set restrictive permissions on Unix
                if sys.platform != "win32":
                    try:
                        self._cache_dir.chmod(0o700)
                    except OSError:
                        pass  # Best effort

                self._logger.debug(f"Cache initialized at: {self._cache_dir}")

            except Exception as e:
                self._logger.warning(
                    f"Failed to create cache directory, caching disabled: {type(e).__name__}"
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled."""
        return self._enabled

    @property
    def cache_dir(self) -> Path:
        """The cache directory path."""
        return self._cache_dir

    def build_cache_key(
        self,
        media_fingerprint: str,
        config_fingerprint: str,
        purpose: str,
    ) -> str:
        """Create a unique cache key for a specific analysis result.

        Combines machine ID, fingerprints, and purpose into a single key.

        Args:
            media_fingerprint: Hash of the media set.
            config_fingerprint: Hash of the analysis configuration.
            purpose: Analysis stage identifier (e.g., "full_report").

        Returns:
            First 32 hex characters of SHA-256 hash.

        Example:
            >>> key = cache.build_cache_key(media_fp, config_fp, "full_report")
            >>> len(key)
            32
        """
        combined = f"{self._machine_id}:{media_fingerprint}:{config_fingerprint}:{purpose}"
        hash_digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        return hash_digest[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Return the file path for a cache key.

        Args:
            key: The cache key.

        Returns:
            Path to the cache file.
        """
        return self._cache_dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        """Check if a cache entry exists (file exists, not validated).

        Args:
            key: The cache key to check.

        Returns:
            True if cache file exists, False otherwise.
        """
        if not self._enabled:
            return False
        return self._get_cache_path(key).exists()

    def load(
        self,
        key: str,
        media_fingerprint: str,
        config_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Load and validate a cache entry.

        Attempts to load a cached result and validates it matches
        the current machine, media set, and configuration.

        Args:
            key: The cache key.
            media_fingerprint: Expected media set fingerprint.
            config_fingerprint: Expected config fingerprint.

        Returns:
            The cached payload dict if valid, None otherwise.

        Note:
            This method NEVER raises exceptions. All errors result in None.
        """
        if not self._enabled:
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._logger.debug(f"Cache miss: file not found for key {key[:16]}...")
            return None

        try:
            # Read and parse JSON
            content = cache_path.read_text(encoding="utf-8")
            data = json.loads(content)

            # Parse as CacheEntry
            entry = CacheEntry.model_validate(data)

            # Validate machine ID
            if entry.meta.machine_id != self._machine_id:
                self._logger.debug(f"Cache miss: machine ID mismatch for key {key[:16]}...")
                return None

            # Validate media fingerprint
            if entry.meta.media_set_fingerprint != media_fingerprint:
                self._logger.debug(f"Cache miss: media fingerprint mismatch for key {key[:16]}...")
                return None

            # Validate config fingerprint
            if entry.meta.analysis_config_fingerprint != config_fingerprint:
                self._logger.debug(f"Cache miss: config fingerprint mismatch for key {key[:16]}...")
                return None

            # Validate version
            if entry.meta.version != self._version:
                self._logger.debug(f"Cache miss: version mismatch for key {key[:16]}...")
                return None

            self._logger.debug(f"Cache hit for key {key[:16]}...")
            return entry.payload

        except json.JSONDecodeError as e:
            self._logger.warning(f"Cache corrupted (JSON error): {type(e).__name__}")
            return None
        except Exception as e:
            self._logger.warning(f"Cache load failed: {type(e).__name__}")
            return None

    def store(
        self,
        key: str,
        meta: CacheMeta,
        payload: dict[str, Any],
    ) -> bool:
        """Store a cache entry atomically.

        Writes the cache entry to a temp file first, then atomically
        renames it to ensure write integrity.

        Args:
            key: The cache key.
            meta: Cache metadata.
            payload: The data to cache.

        Returns:
            True if storage was successful, False otherwise.

        Note:
            This method NEVER raises exceptions. All errors result in False.
        """
        if not self._enabled:
            return False

        cache_path = self._get_cache_path(key)

        try:
            # Build cache entry
            entry = CacheEntry(meta=meta, payload=payload)

            # Serialize to JSON
            json_content = entry.model_dump_json(indent=2)

            # Write to temp file in same directory (ensures same filesystem)
            fd, temp_path = tempfile.mkstemp(
                dir=self._cache_dir,
                suffix=".tmp",
                prefix=".cache_",
            )

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json_content)

                # Atomic rename
                Path(temp_path).replace(cache_path)

                self._logger.debug(f"Cached result: {key[:16]}...")
                return True

            except Exception:
                # Cleanup temp file on error
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                raise

        except Exception as e:
            self._logger.warning(f"Cache store failed: {type(e).__name__}")
            return False

    def invalidate(self, key: str) -> bool:
        """Delete a specific cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        if not self._enabled:
            return False

        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
                self._logger.debug(f"Invalidated cache: {key[:16]}...")
                return True
            return False
        except Exception as e:
            self._logger.warning(f"Cache invalidate failed: {type(e).__name__}")
            return False

    def clear(self) -> int:
        """Delete all cache entries.

        Returns:
            Number of cache files deleted.
        """
        if not self._enabled:
            return 0

        count = 0

        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass

            if count > 0:
                self._logger.info(f"Cleared {count} cache entries")

            return count

        except Exception as e:
            self._logger.warning(f"Cache clear failed: {type(e).__name__}")
            return count

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics for debugging.

        Returns:
            Dictionary with cache statistics:
            - enabled: Whether caching is active
            - cache_dir: Path to cache directory
            - entry_count: Number of cache files
            - total_size_bytes: Total size of cache files
            - oldest_entry: Timestamp of oldest entry (or None)
            - newest_entry: Timestamp of newest entry (or None)
        """
        stats: dict[str, Any] = {
            "enabled": self._enabled,
            "cache_dir": str(self._cache_dir),
            "entry_count": 0,
            "total_size_bytes": 0,
            "oldest_entry": None,
            "newest_entry": None,
        }

        if not self._enabled:
            return stats

        try:
            cache_files = list(self._cache_dir.glob("*.json"))
            stats["entry_count"] = len(cache_files)

            oldest = None
            newest = None
            total_size = 0

            for cache_file in cache_files:
                try:
                    file_stat = cache_file.stat()
                    total_size += file_stat.st_size
                    mtime = file_stat.st_mtime

                    if oldest is None or mtime < oldest:
                        oldest = mtime
                    if newest is None or mtime > newest:
                        newest = mtime

                except Exception:
                    pass

            stats["total_size_bytes"] = total_size
            stats["oldest_entry"] = oldest
            stats["newest_entry"] = newest

        except Exception as e:
            self._logger.debug(f"Cache stats failed: {type(e).__name__}")

        return stats
