"""Legacy proxy for src.core.models.

This module is retained for backward compatibility with existing imports.
All core data models have been moved to src.core.models.
"""

from dlnai.core.models import (
    ConfidenceLevel,
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)

# Re-export everything for compatibility
__all__ = [
    "Memory",
    "MediaType",
    "SourcePlatform",
    "ConfidenceLevel",
    "GeoPoint",
    "Location",
    "PersonTag",
]
