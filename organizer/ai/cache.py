"""Machine-Local AI Analysis Cache for Digital Life Narrative AI.

This module provides a local caching layer for AI analysis results, 
ensuring efficiency and cost control.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def get_machine_id() -> str:
    """Generate a stable identifier unique to this machine/user combination."""
    try:
        hostname = platform.node() or "unknown"
        home_path = str(Path.home())
        combined = f"{hostname}:{home_path}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(b"fallback-machine-id").hexdigest()[:16]

class AICache:
    """Local file-based cache for AI analysis results."""

    def __init__(self, cache_dir: Path | None = None, enabled: bool = True) -> None:
        self._enabled = enabled
        self._machine_id = get_machine_id()
        self._version = "1.1" # Version bump for multimodal changes

        if cache_dir is None:
            from organizer.config import get_app_data_dir
            self._cache_dir = get_app_data_dir() / "cache"
        else:
            self._cache_dir = cache_dir

        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def build_cache_key(self, media_fingerprint: str, config_fingerprint: str, purpose: str) -> str:
        """Create a unique cache key."""
        combined = f"{self._machine_id}:{media_fingerprint}:{config_fingerprint}:{purpose}:{self._version}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def load(self, key: str) -> dict[str, Any] | None:
        """Load a cache entry."""
        if not self._enabled:
            return None

        cache_path = self._cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("payload")
        except Exception as e:
            logger.warning(f"Failed to load cache entry: {e}")
            return None

    def store(self, key: str, payload: dict[str, Any]) -> bool:
        """Store a cache entry."""
        if not self._enabled:
            return False

        cache_path = self._cache_dir / f"{key}.json"
        try:
            # Atomic write
            with tempfile.NamedTemporaryFile("w", dir=self._cache_dir, delete=False, encoding="utf-8") as tf:
                json.dump({"payload": payload, "machine_id": self._machine_id, "version": self._version}, tf, indent=2)
                temp_name = tf.name
            
            Path(temp_name).replace(cache_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to store cache entry: {e}")
            return False

    def clear(self) -> int:
        """Delete all cache entries."""
        count = 0
        for f in self._cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        return count
