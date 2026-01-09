"""Timeline - Aggregation and Analysis Layer for Memories.

This module provides the Timeline class that aggregates Memory objects into
structured, analyzable timelines. This is the data preparation layer that feeds
the AI analyzer.

A Timeline is NOT just a chronologically sorted list. It provides:
- Time period aggregation (days, weeks, months, quarters, years)
- Pattern detection (gaps, bursts, trends)
- Statistical analysis (coverage, density, quality)
- Cross-platform deduplication
- AI-ready summaries

The Timeline is what makes AI analysis possible by transforming raw memories
into patterns the AI can interpret.

Example:
    >>> from src.core import Timeline, Memory
    >>>
    >>> timeline = Timeline(memories)
    >>> stats = timeline.compute_statistics()
    >>> print(f"Coverage: {stats.coverage_percentage:.1f}%")
    >>> print(f"Quality: {stats.quality_score.value}")
    >>>
    >>> # Prepare for AI
    >>> ai_summary = timeline.to_ai_summary(max_items=500)
    >>> send_to_gemini(ai_summary)
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, Field, computed_field

from src.core.models import (
    ConfidenceLevel,
    Location,
    MediaType,
    Memory,
    SourcePlatform,
)


# =============================================================================
# Helper Models
# =============================================================================


class DateRange(BaseModel):
    """Simple date range model.

    Attributes:
        start: Start date (inclusive)
        end: End date (inclusive)
    """

    start: date
    end: date

    @computed_field
    @property
    def days(self) -> int:
        """Duration in days."""
        return (self.end - self.start).days + 1

    def contains(self, d: date) -> bool:
        """Check if date is within range."""
        return self.start <= d <= self.end

    def overlaps(self, other: DateRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and self.end >= other.start

    def merge(self, other: DateRange) -> DateRange:
        """Merge two ranges into one spanning both."""
        return DateRange(start=min(self.start, other.start), end=max(self.end, other.end))


# =============================================================================
# Enums
# =============================================================================


class TimeGranularity(str, Enum):
    """Granularity for time period aggregation.

    Different analyses need different granularity levels.

    Attributes:
        DAY: Daily buckets
        WEEK: Weekly buckets (Monday-Sunday)
        MONTH: Monthly buckets
        QUARTER: Quarterly buckets (Q1-Q4)
        YEAR: Yearly buckets
    """

    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TimelineQuality(str, Enum):
    """Overall quality assessment of timeline data.

    AI analysis quality depends heavily on timeline quality.

    Attributes:
        EXCELLENT: >80% coverage, high confidence timestamps
        GOOD: >50% coverage, mostly reliable timestamps
        FAIR: >25% coverage, mixed confidence
        SPARSE: <25% coverage, significant gaps
        POOR: <10% coverage or major data issues
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    SPARSE = "sparse"
    POOR = "poor"


# =============================================================================
# Supporting Models
# =============================================================================


class TimePeriod(BaseModel):
    """A bucket of memories within a time range.

    Represents an aggregated time period at a specific granularity.

    Attributes:
        period_start: Start date of period
        period_end: End date of period
        granularity: Time granularity of this period
        memory_count: Number of memories in period
        memory_ids: List of memory IDs in this period
        platform_counts: Count by platform
        media_type_counts: Count by media type
        has_location_data: True if any memory has location
        has_people_data: True if any memory has people
        average_confidence: Average confidence level (0-1 scale)

    Example:
        >>> period = TimePeriod(
        ...     period_start=date(2020, 7, 1),
        ...     period_end=date(2020, 7, 31),
        ...     granularity=TimeGranularity.MONTH,
        ...     memory_count=145
        ... )
        >>> period.density()  # memories per day
    """

    period_start: date
    period_end: date
    granularity: TimeGranularity
    memory_count: int = 0
    memory_ids: list[str] = Field(default_factory=list)
    platform_counts: dict[str, int] = Field(default_factory=dict)
    media_type_counts: dict[str, int] = Field(default_factory=dict)
    has_location_data: bool = False
    has_people_data: bool = False
    average_confidence: float = 0.0

    def is_empty(self) -> bool:
        """Check if period has no memories.

        Returns:
            True if memory_count is 0
        """
        return self.memory_count == 0

    def density(self) -> float:
        """Calculate memories per day in this period.

        Returns:
            Average memories per day
        """
        days = (self.period_end - self.period_start).days + 1
        if days == 0:
            return 0.0
        return self.memory_count / days

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization/AI.

        Returns:
            Dictionary representation
        """
        return {
            "start": self.period_start.isoformat(),
            "end": self.period_end.isoformat(),
            "granularity": self.granularity.value,
            "memory_count": self.memory_count,
            "density": round(self.density(), 2),
            "platforms": self.platform_counts,
            "media_types": self.media_type_counts,
            "has_location": self.has_location_data,
            "has_people": self.has_people_data,
            "avg_confidence": round(self.average_confidence, 2),
        }


class TimelineGap(BaseModel):
    """Detected gap in the timeline.

    Represents a period with no or very little data.

    Attributes:
        start_date: When gap starts
        end_date: When gap ends
        duration_days: Length of gap in days
        preceding_period: Period before gap (if any)
        following_period: Period after gap (if any)
        severity: Classification of gap importance
        possible_causes: Inferred reasons for gap

    Example:
        >>> gap = TimelineGap(
        ...     start_date=date(2019, 1, 1),
        ...     end_date=date(2019, 12, 31),
        ...     duration_days=365,
        ...     severity="critical",
        ...     possible_causes=["Platform not used", "Lost data"]
        ... )
    """

    start_date: date
    end_date: date
    duration_days: int
    preceding_period: TimePeriod | None = None
    following_period: TimePeriod | None = None
    severity: Literal["minor", "moderate", "major", "critical"] = "moderate"
    possible_causes: list[str] = Field(default_factory=list)

    def to_ai_description(self) -> str:
        """Generate AI-readable description of gap.

        Returns:
            Human-readable gap description
        """
        duration_desc = f"{self.duration_days} days"
        if self.duration_days >= 365:
            years = self.duration_days / 365
            duration_desc = f"{years:.1f} years"
        elif self.duration_days >= 30:
            months = self.duration_days / 30
            duration_desc = f"{months:.1f} months"

        desc = (
            f"{self.severity.upper()} gap: {duration_desc} "
            f"({self.start_date} to {self.end_date})"
        )

        if self.possible_causes:
            desc += f". Possible causes: {', '.join(self.possible_causes)}"

        return desc


class PlatformActivity(BaseModel):
    """Aggregated platform usage statistics.

    Detailed activity breakdown for a single platform.

    Attributes:
        platform: Source platform
        total_memories: Total count from this platform
        first_memory_date: Earliest memory date
        last_memory_date: Latest memory date
        peak_year: Year with most activity
        peak_month: Month with most activity (YYYY-MM)
        active_months: Months with at least 1 memory
        average_per_active_month: Average memories per active month
        common_media_types: Most frequent media types
        unique_locations_count: Distinct locations
        unique_people_count: Distinct people
    """

    platform: SourcePlatform
    total_memories: int
    first_memory_date: date | None = None
    last_memory_date: date | None = None
    peak_year: int | None = None
    peak_month: str | None = None
    active_months: int = 0
    average_per_active_month: float = 0.0
    common_media_types: list[MediaType] = Field(default_factory=list)
    unique_locations_count: int = 0
    unique_people_count: int = 0


class TimelineStatistics(BaseModel):
    """Comprehensive timeline statistics.

    Complete statistical summary of the timeline.

    Attributes:
        total_memories: Total memory count
        dated_memories: Memories with timestamps
        undated_memories: Memories without timestamps
        date_range: Overall date range
        total_days_span: Days from first to last
        days_with_memories: Days that have at least 1 memory
        coverage_percentage: Percentage of days covered
        memories_per_day_average: Average memories per day
        memories_per_month_average: Average memories per month
        busiest_day: Day with most memories
        busiest_day_count: Memory count on busiest day
        busiest_month: Month with most memories (YYYY-MM)
        busiest_month_count: Memory count in busiest month
        quietest_year: Year with fewest memories
        platform_breakdown: Per-platform statistics
        media_type_breakdown: Count by media type
        location_coverage: Percentage with location data
        people_coverage: Percentage with people tags
        average_confidence: Overall confidence level
        quality_score: Overall quality assessment
        gaps: Detected gaps in timeline
    """

    total_memories: int
    dated_memories: int
    undated_memories: int
    date_range: DateRange | None = None
    total_days_span: int = 0
    days_with_memories: int = 0
    coverage_percentage: float = 0.0
    memories_per_day_average: float = 0.0
    memories_per_month_average: float = 0.0
    busiest_day: date | None = None
    busiest_day_count: int = 0
    busiest_month: str | None = None
    busiest_month_count: int = 0
    quietest_year: int | None = None
    platform_breakdown: list[PlatformActivity] = Field(default_factory=list)
    media_type_breakdown: dict[str, int] = Field(default_factory=dict)
    location_coverage: float = 0.0
    people_coverage: float = 0.0
    average_confidence: float = 0.0
    quality_score: TimelineQuality = TimelineQuality.POOR
    gaps: list[TimelineGap] = Field(default_factory=list)


# =============================================================================
# Main Timeline Class
# =============================================================================


class Timeline:
    """Timeline aggregator and analyzer for Memory objects.

    The main data preparation layer that transforms raw memories into
    analyzable patterns for AI consumption.

    Attributes:
        _memories: Dictionary mapping memory IDs to Memory objects
        _by_date: Dictionary mapping dates to memory ID lists
        _by_platform: Dictionary mapping platforms to memory ID lists
        _by_year: Dictionary mapping years to memory ID lists
        _by_month: Dictionary mapping YYYY-MM to memory ID lists
        _undated: List of memory IDs without timestamps
        _statistics: Cached statistics (invalidated on mutation)
        _periods: Cached period aggregations

    Example:
        >>> timeline = Timeline()
        >>> timeline.add_memories(parsed_memories)
        >>>
        >>> # Query
        >>> july_2020 = timeline.get_memories_for_month(2020, 7)
        >>>
        >>> # Analysis
        >>> stats = timeline.compute_statistics()
        >>> gaps = timeline.detect_gaps(threshold_days=30)
        >>>
        >>> # AI preparation
        >>> summary = timeline.to_ai_summary(max_items=500)
    """

    def __init__(self, memories: list[Memory] | None = None):
        """Initialize timeline with optional memories.

        Args:
            memories: Optional list of memories to add
        """
        # Storage
        self._memories: dict[str, Memory] = {}

        # Indexes
        self._by_date: dict[date, list[str]] = defaultdict(list)
        self._by_platform: dict[SourcePlatform, list[str]] = defaultdict(list)
        self._by_year: dict[int, list[str]] = defaultdict(list)
        self._by_month: dict[str, list[str]] = defaultdict(list)
        self._undated: list[str] = []

        # Caches
        self._statistics: TimelineStatistics | None = None
        self._periods: dict[TimeGranularity, list[TimePeriod]] = {}

        # Add initial memories
        if memories:
            self.add_memories(memories)

    # =========================================================================
    # Memory Management Methods
    # =========================================================================

    def add_memory(self, memory: Memory) -> None:
        """Add a single memory to timeline.

        Updates all indexes and invalidates caches.

        Args:
            memory: Memory to add
        """
        # Add to main storage
        self._memories[memory.id] = memory

        # Update platform index
        self._by_platform[memory.source_platform].append(memory.id)

        # Update date indexes
        if memory.created_at:
            mem_date = memory.created_at.date()
            self._by_date[mem_date].append(memory.id)

            # Year index
            year = memory.created_at.year
            self._by_year[year].append(memory.id)

            # Month index (YYYY-MM format)
            month_key = f"{year:04d}-{memory.created_at.month:02d}"
            self._by_month[month_key].append(memory.id)
        else:
            # No timestamp
            self._undated.append(memory.id)

        # Invalidate caches
        self._invalidate_caches()

    def add_memories(self, memories: list[Memory]) -> int:
        """Bulk add memories.

        More efficient than individual adds as caches are invalidated once.

        Args:
            memories: List of memories to add

        Returns:
            Count of memories added
        """
        for memory in memories:
            # Direct add without cache invalidation
            self._memories[memory.id] = memory
            self._by_platform[memory.source_platform].append(memory.id)

            if memory.created_at:
                mem_date = memory.created_at.date()
                self._by_date[mem_date].append(memory.id)
                year = memory.created_at.year
                self._by_year[year].append(memory.id)
                month_key = f"{year:04d}-{memory.created_at.month:02d}"
                self._by_month[month_key].append(memory.id)
            else:
                self._undated.append(memory.id)

        # Single cache invalidation
        self._invalidate_caches()

        return len(memories)

    def remove_memory(self, memory_id: str) -> bool:
        """Remove memory by ID.

        Args:
            memory_id: ID of memory to remove

        Returns:
            True if memory was found and removed
        """
        if memory_id not in self._memories:
            return False

        memory = self._memories[memory_id]

        # Remove from main storage
        del self._memories[memory_id]

        # Remove from platform index
        if memory_id in self._by_platform[memory.source_platform]:
            self._by_platform[memory.source_platform].remove(memory_id)

        # Remove from date indexes
        if memory.created_at:
            mem_date = memory.created_at.date()
            if memory_id in self._by_date[mem_date]:
                self._by_date[mem_date].remove(memory_id)

            year = memory.created_at.year
            if memory_id in self._by_year[year]:
                self._by_year[year].remove(memory_id)

            month_key = f"{year:04d}-{memory.created_at.month:02d}"
            if memory_id in self._by_month[month_key]:
                self._by_month[month_key].remove(memory_id)
        else:
            if memory_id in self._undated:
                self._undated.remove(memory_id)

        self._invalidate_caches()
        return True

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None if not found
        """
        return self._memories.get(memory_id)

    def get_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Bulk retrieve memories, preserving order.

        Args:
            memory_ids: List of memory IDs

        Returns:
            List of Memory objects (skips missing IDs)
        """
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def __len__(self) -> int:
        """Total memory count.

        Returns:
            Number of memories in timeline
        """
        return len(self._memories)

    def __iter__(self) -> Iterator[Memory]:
        """Iterate all memories in chronological order.

        Yields dated memories first (sorted), then undated.

        Yields:
            Memory objects in chronological order
        """
        # Get all dated memories sorted by date
        dated_pairs = []
        for mem_id, memory in self._memories.items():
            if memory.created_at:
                dated_pairs.append((memory.created_at, memory))

        dated_pairs.sort(key=lambda x: x[0])

        # Yield dated memories
        for _, memory in dated_pairs:
            yield memory

        # Yield undated memories
        for mem_id in self._undated:
            if mem_id in self._memories:
                yield self._memories[mem_id]

    def _invalidate_caches(self) -> None:
        """Invalidate all cached computations."""
        self._statistics = None
        self._periods = {}

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_memories_in_range(self, start: date, end=date) -> list[Memory]:
        """Return memories within date range (inclusive).

        Args:
            start: Start date
            end: End date

        Returns:
            List of memories in range, chronologically sorted
        """
        result_ids = []
        current = start

        while current <= end:
            if current in self._by_date:
                result_ids.extend(self._by_date[current])
            current += timedelta(days=1)

        # Get memories and sort by timestamp
        memories = self.get_memories(result_ids)
        memories.sort(key=lambda m: m.created_at if m.created_at else datetime.max)

        return memories

    def get_memories_for_date(self, d: date) -> list[Memory]:
        """Return memories for a specific date.

        Args:
            d: Date to query

        Returns:
            List of memories from that date
        """
        return self.get_memories(self._by_date.get(d, []))

    def get_memories_for_month(self, year: int, month: int) -> list[Memory]:
        """Return memories for a specific month.

        Args:
            year: Year (e.g., 2020)
            month: Month (1-12)

        Returns:
            List of memories from that month
        """
        month_key = f"{year:04d}-{month:02d}"
        return self.get_memories(self._by_month.get(month_key, []))

    def get_memories_for_year(self, year: int) -> list[Memory]:
        """Return memories for a specific year.

        Args:
            year: Year (e.g., 2020)

        Returns:
            List of memories from that year
        """
        return self.get_memories(self._by_year.get(year, []))

    def get_memories_by_platform(self, platform: SourcePlatform) -> list[Memory]:
        """Return all memories from a platform.

        Args:
            platform: Source platform

        Returns:
            List of memories from that platform
        """
        return self.get_memories(self._by_platform.get(platform, []))

    def get_memories_with_location(self) -> list[Memory]:
        """Return memories that have location data.

        Returns:
            List of memories with non-empty location
        """
        return [m for m in self._memories.values() if m.location and not m.location.is_empty()]

    def get_memories_with_people(self) -> list[Memory]:
        """Return memories that have people tags.

        Returns:
            List of memories with at least one person tag
        """
        return [m for m in self._memories.values() if len(m.people) > 0]

    def get_undated_memories(self) -> list[Memory]:
        """Return memories without timestamps.

        Returns:
            List of undated memories
        """
        return self.get_memories(self._undated)

    # =========================================================================
    # Aggregation Methods
    # =========================================================================

    def get_periods(self, granularity: TimeGranularity) -> list[TimePeriod]:
        """Divide timeline into periods of specified granularity.

        Results are cached for efficiency.

        Args:
            granularity: Time period granularity

        Returns:
            List of TimePeriod objects
        """
        if granularity in self._periods:
            return self._periods[granularity]

        if not self._by_date:
            return []

        # Get date range
        all_dates = sorted(self._by_date.keys())
        if not all_dates:
            return []

        start_date = all_dates[0]
        end_date = all_dates[-1]

        periods = []
        current = start_date

        while current <= end_date:
            # Determine period boundaries
            if granularity == TimeGranularity.DAY:
                period_start = current
                period_end = current
                current += timedelta(days=1)
            elif granularity == TimeGranularity.WEEK:
                # Week starts Monday
                period_start = current - timedelta(days=current.weekday())
                period_end = period_start + timedelta(days=6)
                current = period_end + timedelta(days=1)
            elif granularity == TimeGranularity.MONTH:
                period_start = current.replace(day=1)
                # Last day of month
                if current.month == 12:
                    period_end = current.replace(month=12, day=31)
                else:
                    next_month = current.replace(month=current.month + 1, day=1)
                    period_end = next_month - timedelta(days=1)
                current = period_end + timedelta(days=1)
            elif granularity == TimeGranularity.QUARTER:
                quarter = (current.month - 1) // 3 + 1
                period_start = current.replace(month=(quarter - 1) * 3 + 1, day=1)
                quarter_end_month = quarter * 3
                if quarter_end_month == 12:
                    period_end = current.replace(month=12, day=31)
                else:
                    next_quarter = current.replace(month=quarter_end_month + 1, day=1)
                    period_end = next_quarter - timedelta(days=1)
                current = period_end + timedelta(days=1)
            elif granularity == TimeGranularity.YEAR:
                period_start = current.replace(month=1, day=1)
                period_end = current.replace(month=12, day=31)
                current = period_end + timedelta(days=1)
            else:
                break

            # Get memories in this period
            period_memories = self.get_memories_in_range(period_start, min(period_end, end_date))

            if period_memories:
                # Aggregate statistics
                platform_counts: dict[str, int] = Counter()
                media_type_counts: dict[str, int] = Counter()
                has_location = False
                has_people = False
                confidence_values = []

                for mem in period_memories:
                    platform_counts[mem.source_platform.value] += 1
                    media_type_counts[mem.media_type.value] += 1

                    if mem.location and not mem.location.is_empty():
                        has_location = True
                    if mem.people:
                        has_people = True

                    # Map confidence to 0-1 scale
                    conf_map = {
                        ConfidenceLevel.VERIFIED: 1.0,
                        ConfidenceLevel.HIGH: 0.8,
                        ConfidenceLevel.MEDIUM: 0.6,
                        ConfidenceLevel.LOW: 0.4,
                        ConfidenceLevel.INFERRED: 0.2,
                    }
                    confidence_values.append(conf_map.get(mem.created_at_confidence, 0.5))

                avg_confidence = mean(confidence_values) if confidence_values else 0.0

                period = TimePeriod(
                    period_start=period_start,
                    period_end=min(period_end, end_date),
                    granularity=granularity,
                    memory_count=len(period_memories),
                    memory_ids=[m.id for m in period_memories],
                    platform_counts=dict(platform_counts),
                    media_type_counts=dict(media_type_counts),
                    has_location_data=has_location,
                    has_people_data=has_people,
                    average_confidence=avg_confidence,
                )

                periods.append(period)

        # Cache and return
        self._periods[granularity] = periods
        return periods

    def get_activity_by_year(self) -> dict[int, int]:
        """Return memory count by year.

        Returns:
            Dictionary mapping year to count
        """
        return {year: len(ids) for year, ids in self._by_year.items()}

    def get_activity_by_month(self) -> dict[str, int]:
        """Return memory count by month.

        Returns:
            Dictionary mapping YYYY-MM to count
        """
        return {month: len(ids) for month, ids in self._by_month.items()}

    def get_activity_by_day_of_week(self) -> dict[int, int]:
        """Return memory count by day of week.

        Returns:
            Dictionary mapping 0-6 (Monday-Sunday) to count
        """
        day_counts: dict[int, int] = Counter()

        for d, ids in self._by_date.items():
            day_counts[d.weekday()] += len(ids)

        return dict(day_counts)

    def get_activity_by_hour(self) -> dict[int, int]:
        """Return memory count by hour of day.

        Only considers memories with time component.

        Returns:
            Dictionary mapping 0-23 to count
        """
        hour_counts: dict[int, int] = Counter()

        for memory in self._memories.values():
            if memory.created_at and isinstance(memory.created_at, datetime):
                hour_counts[memory.created_at.hour] += 1

        return dict(hour_counts)

    def get_platform_activity(self, platform: SourcePlatform) -> PlatformActivity:
        """Compute detailed activity for a platform.

        Args:
            platform: Platform to analyze

        Returns:
            PlatformActivity statistics
        """
        memories = self.get_memories_by_platform(platform)

        if not memories:
            return PlatformActivity(
                platform=platform,
                total_memories=0,
            )

        # Filter dated memories
        dated = [m for m in memories if m.created_at]

        # Dates
        first_date = min(m.created_at.date() for m in dated) if dated else None
        last_date = max(m.created_at.date() for m in dated) if dated else None

        # Activity by year/month
        year_counts: Counter = Counter()
        month_counts: Counter = Counter()

        for m in dated:
            year_counts[m.created_at.year] += 1
            month_key = f"{m.created_at.year:04d}-{m.created_at.month:02d}"
            month_counts[month_key] += 1

        peak_year = year_counts.most_common(1)[0][0] if year_counts else None
        peak_month = month_counts.most_common(1)[0][0] if month_counts else None
        active_months = len(month_counts)
        avg_per_month = len(memories) / active_months if active_months > 0 else 0.0

        # Media types
        type_counts = Counter(m.media_type for m in memories)
        common_types = [mt for mt, _ in type_counts.most_common(3)]

        # Unique locations and people
        unique_locations = set()
        unique_people = set()

        for m in memories:
            if m.location and not m.location.is_empty():
                # Simple location deduping by place name
                if m.location.place_name:
                    unique_locations.add(m.location.place_name)

            for person in m.people:
                unique_people.add(person.normalized_name or person.name)

        return PlatformActivity(
            platform=platform,
            total_memories=len(memories),
            first_memory_date=first_date,
            last_memory_date=last_date,
            peak_year=peak_year,
            peak_month=peak_month,
            active_months=active_months,
            average_per_active_month=avg_per_month,
            common_media_types=common_types,
            unique_locations_count=len(unique_locations),
            unique_people_count=len(unique_people),
        )

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def compute_statistics(self) -> TimelineStatistics:
        """Compute comprehensive statistics.

        Result is cached until timeline is modified.

        Returns:
            TimelineStatistics object
        """
        if self._statistics is not None:
            return self._statistics

        total = len(self._memories)
        dated = total - len(self._undated)
        undated = len(self._undated)

        # Date range
        date_range = None
        total_days_span = 0
        days_with_memories = len(self._by_date)
        coverage_pct = 0.0
        memories_per_day = 0.0
        memories_per_month = 0.0

        if self._by_date:
            all_dates = sorted(self._by_date.keys())
            start = all_dates[0]
            end = all_dates[-1]
            date_range = DateRange(start=start, end=end)
            total_days_span = (end - start).days + 1

            if total_days_span > 0:
                coverage_pct = (days_with_memories / total_days_span) * 100
                memories_per_day = dated / total_days_span

            if len(self._by_month) > 0:
                memories_per_month = dated / len(self._by_month)

        # Busiest day/month
        busiest_day = None
        busiest_day_count = 0

        if self._by_date:
            busiest_day, day_ids = max(self._by_date.items(), key=lambda x: len(x[1]))
            busiest_day_count = len(day_ids)

        busiest_month = None
        busiest_month_count = 0

        if self._by_month:
            busiest_month, month_ids = max(self._by_month.items(), key=lambda x: len(x[1]))
            busiest_month_count = len(month_ids)

        # Quietest year
        quietest_year = None
        if len(self._by_year) > 1:
            quietest_year, _ = min(self._by_year.items(), key=lambda x: len(x[1]))

        # Platform breakdown
        platform_breakdown = [
            self.get_platform_activity(platform) for platform in self._by_platform.keys()
        ]

        # Media type breakdown
        media_type_counts: Counter = Counter()
        for memory in self._memories.values():
            media_type_counts[memory.media_type.value] += 1

        # Coverage percentages
        location_count = len(self.get_memories_with_location())
        people_count = len(self.get_memories_with_people())

        location_coverage = (location_count / total * 100) if total > 0 else 0.0
        people_coverage = (people_count / total * 100) if total > 0 else 0.0

        # Average confidence
        conf_map = {
            ConfidenceLevel.VERIFIED: 1.0,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.INFERRED: 0.2,
        }

        confidence_values = [
            conf_map.get(m.created_at_confidence, 0.5)
            for m in self._memories.values()
            if m.created_at
        ]

        avg_confidence = mean(confidence_values) if confidence_values else 0.0

        # Quality assessment
        quality = self.assess_quality()

        # Detect gaps
        gaps = self.detect_gaps(threshold_days=30)

        stats = TimelineStatistics(
            total_memories=total,
            dated_memories=dated,
            undated_memories=undated,
            date_range=date_range,
            total_days_span=total_days_span,
            days_with_memories=days_with_memories,
            coverage_percentage=coverage_pct,
            memories_per_day_average=memories_per_day,
            memories_per_month_average=memories_per_month,
            busiest_day=busiest_day,
            busiest_day_count=busiest_day_count,
            busiest_month=busiest_month,
            busiest_month_count=busiest_month_count,
            quietest_year=quietest_year,
            platform_breakdown=platform_breakdown,
            media_type_breakdown=dict(media_type_counts),
            location_coverage=location_coverage,
            people_coverage=people_coverage,
            average_confidence=avg_confidence,
            quality_score=quality,
            gaps=gaps,
        )

        # Cache
        self._statistics = stats
        return stats

    def detect_gaps(self, threshold_days: int = 30) -> list[TimelineGap]:
        """Find gaps larger than threshold.

        Args:
            threshold_days: Minimum gap size to detect

        Returns:
            List of TimelineGap objects
        """
        if not self._by_date:
            return []

        gaps = []
        all_dates = sorted(self._by_date.keys())

        for i in range(len(all_dates) - 1):
            current_date = all_dates[i]
            next_date = all_dates[i + 1]

            gap_days = (next_date - current_date).days - 1

            if gap_days >= threshold_days:
                # Classify severity
                if gap_days < 60:
                    severity = "minor"
                elif gap_days < 180:
                    severity = "moderate"
                elif gap_days < 365:
                    severity = "major"
                else:
                    severity = "critical"

                # Infer possible causes
                possible_causes = []
                if gap_days >= 365:
                    possible_causes.append("Year-long gap - possible life event or lost data")
                elif gap_days >= 90:
                    possible_causes.append("Extended period without platform usage")

                gap = TimelineGap(
                    start_date=current_date + timedelta(days=1),
                    end_date=next_date - timedelta(days=1),
                    duration_days=gap_days,
                    severity=severity,
                    possible_causes=possible_causes,
                )

                gaps.append(gap)

        return gaps

    def detect_bursts(self, threshold_multiplier: float = 3.0) -> list[TimePeriod]:
        """Find periods with unusually high activity.

        Args:
            threshold_multiplier: Activity must exceed average by this factor

        Returns:
            List of TimePeriod objects with high activity
        """
        periods = self.get_periods(TimeGranularity.WEEK)

        if not periods:
            return []

        # Calculate average
        avg_count = mean([p.memory_count for p in periods])
        threshold = avg_count * threshold_multiplier

        # Find bursts
        bursts = [p for p in periods if p.memory_count > threshold]

        return bursts

    def detect_trends(self) -> dict:
        """Analyze overall trends.

        Returns:
            Dictionary with trend analysis
        """
        year_activity = self.get_activity_by_year()

        if len(year_activity) < 2:
            return {"trend": "insufficient_data"}

        years = sorted(year_activity.keys())
        counts = [year_activity[y] for y in years]

        # Simple linear trend
        first_half = mean(counts[: len(counts) // 2])
        second_half = mean(counts[len(counts) // 2 :])

        if second_half > first_half * 1.2:
            trend = "increasing"
        elif second_half < first_half * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "year_range": f"{years[0]}-{years[-1]}",
            "total_years": len(years),
            "avg_first_half": round(first_half),
            "avg_second_half": round(second_half),
        }

    def find_duplicates(self, tolerance_seconds: int = 60) -> list[tuple[str, str]]:
        """Find potential duplicate memories.

        Uses is_same_moment() from Memory class.

        Args:
            tolerance_seconds: Time window for considering memories as duplicates

        Returns:
            List of (memory_id, memory_id) pairs
        """
        duplicates = []
        memories_list = list(self._memories.values())

        # Only check memories with timestamps
        dated = [m for m in memories_list if m.created_at]

        # Sort by timestamp for efficiency
        dated.sort(key=lambda m: m.created_at)

        # Check adjacent memories (most duplicates will be close in time)
        for i in range(len(dated) - 1):
            mem1 = dated[i]

            # Only check next few memories (within tolerance window)
            for j in range(i + 1, min(i + 10, len(dated))):
                mem2 = dated[j]

                # Time difference check first (fast)
                time_diff = abs((mem1.created_at - mem2.created_at).total_seconds())
                if time_diff > tolerance_seconds:
                    break  # No need to check further

                # Detailed check
                if mem1.is_same_moment(mem2, tolerance_seconds=tolerance_seconds):
                    duplicates.append((mem1.id, mem2.id))

        return duplicates

    def assess_quality(self) -> TimelineQuality:
        """Determine overall timeline quality.

        Based on coverage, confidence, and gaps.

        Returns:
            TimelineQuality enum value
        """
        if not self._memories:
            return TimelineQuality.POOR

        dated_count = len(self._memories) - len(self._undated)
        if dated_count == 0:
            return TimelineQuality.POOR

        # Calculate coverage
        coverage = 0.0
        if self._by_date:
            all_dates = sorted(self._by_date.keys())
            if len(all_dates) >= 2:
                span = (all_dates[-1] - all_dates[0]).days + 1
                coverage = len(self._by_date) / span if span > 0 else 0

        # Average confidence
        conf_map = {
            ConfidenceLevel.VERIFIED: 1.0,
            ConfidenceLevel.HIGH: 0.8,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.INFERRED: 0.2,
        }

        confidence_values = [
            conf_map.get(m.created_at_confidence, 0.5)
            for m in self._memories.values()
            if m.created_at
        ]

        avg_confidence = mean(confidence_values) if confidence_values else 0.0

        # Quality decision
        if coverage > 0.8 and avg_confidence > 0.7:
            return TimelineQuality.EXCELLENT
        elif coverage > 0.5 and avg_confidence > 0.5:
            return TimelineQuality.GOOD
        elif coverage > 0.25:
            return TimelineQuality.FAIR
        elif coverage > 0.1:
            return TimelineQuality.SPARSE
        else:
            return TimelineQuality.POOR

    # =========================================================================
    # AI Preparation Methods
    # =========================================================================

    def to_ai_summary(self, max_items: int = 500) -> dict:
        """Prepare timeline summary for Gemini.

        **CRITICAL METHOD** for AI analysis.

        Args:
            max_items: Maximum memories to include in sample

        Returns:
            Dictionary with timeline summary for AI
        """
        stats = self.compute_statistics()

        # Sample memories
        sample = self.sample_memories(max_items, strategy="rich")

        # Temporal distribution
        year_activity = self.get_activity_by_year()
        month_activity = self.get_activity_by_month()

        # Platform breakdown summary
        platform_summary = {
            pa.platform.value: {
                "count": pa.total_memories,
                "first_date": pa.first_memory_date.isoformat() if pa.first_memory_date else None,
                "last_date": pa.last_memory_date.isoformat() if pa.last_memory_date else None,
                "peak_year": pa.peak_year,
            }
            for pa in stats.platform_breakdown
        }

        # Gaps summary
        gaps_summary = [
            {
                "start": gap.start_date.isoformat(),
                "end": gap.end_date.isoformat(),
                "days": gap.duration_days,
                "severity": gap.severity,
                "description": gap.to_ai_description(),
            }
            for gap in stats.gaps[:10]  # Top 10 gaps
        ]

        # Detected patterns
        trends = self.detect_trends()
        bursts = self.detect_bursts()

        return {
            "statistics": {
                "total_memories": stats.total_memories,
                "date_range": {
                    "start": stats.date_range.start.isoformat() if stats.date_range else None,
                    "end": stats.date_range.end.isoformat() if stats.date_range else None,
                    "days_span": stats.total_days_span,
                },
                "coverage_percentage": round(stats.coverage_percentage, 1),
                "quality": stats.quality_score.value,
                "avg_memories_per_day": round(stats.memories_per_day_average, 2),
                "location_coverage": round(stats.location_coverage, 1),
                "people_coverage": round(stats.people_coverage, 1),
            },
            "temporal_distribution": {
                "by_year": year_activity,
                "by_month": dict(sorted(month_activity.items())[:24]),  # Recent 24 months
                "busiest_month": stats.busiest_month,
                "busiest_month_count": stats.busiest_month_count,
            },
            "platform_breakdown": platform_summary,
            "media_types": stats.media_type_breakdown,
            "sample_memories": [
                {
                    "id": m.id,
                    "date": m.created_at.date().isoformat() if m.created_at else None,
                    "platform": m.source_platform.value,
                    "type": m.media_type.value,
                    "has_location": m.location is not None and not m.location.is_empty(),
                    "has_people": len(m.people) > 0,
                    "has_caption": m.caption is not None,
                }
                for m in sample
            ],
            "gaps": gaps_summary,
            "patterns": {
                "trends": trends,
                "burst_count": len(bursts),
                "duplicate_pairs": len(self.find_duplicates()),
            },
        }

    def sample_memories(self, n: int, strategy: str = "stratified") -> list[Memory]:
        """Select representative memories using sampling strategy.

        Args:
            n: Number of memories to sample
            strategy: Sampling strategy
                - "random": Pure random sample
                - "stratified": Proportional by year
                - "coverage": Ensure coverage across time
                - "diverse": Maximize platform/type diversity
                - "rich": Prefer memories with location/people/captions

        Returns:
            List of sampled Memory objects
        """
        memories = list(self._memories.values())
        dated = [m for m in memories if m.created_at]

        if len(dated) <= n:
            return dated

        if strategy == "random":
            return random.sample(dated, n)

        elif strategy == "stratified":
            # Sample proportionally by year
            by_year = {}
            for m in dated:
                year = m.created_at.year
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(m)

            # Calculate samples per year
            total_years = len(by_year)
            samples_per_year = {
                year: max(1, int(n * len(mems) / len(dated))) for year, mems in by_year.items()
            }

            # Sample from each year
            sample = []
            for year, mems in by_year.items():
                k = min(samples_per_year[year], len(mems))
                sample.extend(random.sample(mems, k))

            # Trim to n
            return sample[:n]

        elif strategy == "coverage":
            # Ensure even coverage across timeline
            dated.sort(key=lambda m: m.created_at)
            step = len(dated) // n
            return [dated[i * step] for i in range(n)]

        elif strategy == "diverse":
            # Maximize platform and type diversity
            sample = []
            remaining = dated.copy()

            # First pass: one from each platform/type combo
            combos = set()
            for m in remaining:
                combo = (m.source_platform, m.media_type)
                if combo not in combos:
                    sample.append(m)
                    combos.add(combo)
                if len(sample) >= n:
                    break

            # Fill remainder randomly
            if len(sample) < n:
                remaining = [m for m in remaining if m not in sample]
                sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))

            return sample[:n]

        elif strategy == "rich":
            # Prefer memories with location, people, captions
            def richness_score(m: Memory) -> int:
                score = 0
                if m.location and not m.location.is_empty():
                    score += 2
                if m.people:
                    score += 2
                if m.caption:
                    score += 1
                return score

            # Sort by richness
            dated.sort(key=richness_score, reverse=True)

            # Take top n, but ensure some temporal diversity
            rich = dated[: n * 2]  # Get more candidates
            rich.sort(key=lambda m: m.created_at)

            # Sample evenly from rich set
            step = len(rich) // n
            return [rich[i * step] for i in range(n)]

        else:
            # Default to random
            return random.sample(dated, n)

    def get_chapter_candidates(self) -> list[DateRange]:
        """Use heuristics to suggest potential chapter boundaries.

        AI will refine these, but this gives starting points.

        Returns:
            List of DateRange objects representing candidate chapters
        """
        if not self._by_date:
            return []

        # Get gaps
        gaps = self.detect_gaps(threshold_days=60)

        # Split timeline at major gaps
        all_dates = sorted(self._by_date.keys())
        chapter_boundaries = [all_dates[0]]

        for gap in gaps:
            if gap.severity in ["major", "critical"]:
                # End current chapter before gap, start new after
                chapter_boundaries.append(gap.start_date - timedelta(days=1))
                chapter_boundaries.append(gap.end_date + timedelta(days=1))

        chapter_boundaries.append(all_dates[-1])

        # Create chapters from boundaries
        candidates = []
        for i in range(0, len(chapter_boundaries) - 1, 2):
            start = chapter_boundaries[i]
            end = chapter_boundaries[i + 1]

            if start <= end:
                candidates.append(DateRange(start=start, end=end))

        return candidates

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_json(self, path: Path | None = None) -> str:
        """Export timeline to JSON.

        Args:
            path: Optional file path to write to

        Returns:
            JSON string
        """
        data = {
            "memories": [m.model_dump(mode="json") for m in self._memories.values()],
            "statistics": self.compute_statistics().model_dump(mode="json"),
        }

        json_str = json.dumps(data, indent=2, default=str)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, data: str | Path) -> "Timeline":
        """Load timeline from JSON.

        Args:
            data: JSON string or path to JSON file

        Returns:
            Timeline object
        """
        if isinstance(data, Path):
            with open(data, "r") as f:
                json_data = json.load(f)
        else:
            json_data = json.loads(data)

        memories = [Memory(**m) for m in json_data["memories"]]
        return cls(memories)

    def to_dataframe(self) -> Any:
        """Export to pandas DataFrame.

        Requires pandas to be installed.

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas not available
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for memory in self._memories.values():
            row = {
                "id": memory.id,
                "date": memory.created_at.date() if memory.created_at else None,
                "timestamp": memory.created_at,
                "platform": memory.source_platform.value,
                "media_type": memory.media_type.value,
                "has_location": memory.location is not None and not memory.location.is_empty(),
                "has_people": len(memory.people) > 0,
                "has_caption": memory.caption is not None,
                "confidence": memory.created_at_confidence.value,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def merge(self, other: "Timeline") -> "Timeline":
        """Combine two timelines into one.

        Handles duplicates by memory ID.

        Args:
            other: Another timeline to merge

        Returns:
            New Timeline with combined memories
        """
        # Combine all memories (dict merge handles duplicates by ID)
        combined = {**self._memories, **other._memories}

        return Timeline(list(combined.values()))

    def filter(self, predicate: Callable[[Memory], bool]) -> "Timeline":
        """Return new timeline with only matching memories.

        Args:
            predicate: Function that returns True for memories to keep

        Returns:
            New Timeline with filtered memories
        """
        matching = [m for m in self._memories.values() if predicate(m)]
        return Timeline(matching)

    def slice(self, start: date, end: date) -> "Timeline":
        """Return new timeline with only memories in date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            New Timeline with memories in range
        """
        memories_in_range = self.get_memories_in_range(start, end)
        return Timeline(memories_in_range)
