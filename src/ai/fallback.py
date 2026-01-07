"""Fallback Analyzer — Statistics-Only Report When AI Unavailable.

This module provides the FallbackAnalyzer that produces a degraded,
statistics-only report when the Gemini AI client is unavailable.

The fallback is INTENTIONALLY LIMITED to demonstrate the AI's value:
- No meaningful chapter detection (just years)
- No narratives (just placeholders)
- No thematic analysis
- No pattern detection
- No executive summary (just statistics)

This is the "degraded mode" that proves AI is essential.

What fallback DOES provide:
- Memory counts by platform, year, month
- Date range statistics
- Location/people coverage percentages
- Media type breakdown
- Time-based gap detection
- Year-based "chapters" (not semantic)

What fallback does NOT provide:
- Meaningful chapter titles
- AI-written narratives
- Cross-platform behavior insights
- Pattern detection
- Life story interpretation

Example:
    >>> from src.ai.fallback import create_fallback_report, is_fallback_mode
    >>>
    >>> if is_fallback_mode():
    ...     report = create_fallback_report(memories)
    ...     print(report.executive_summary)  # Stats only
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal

from src.ai.life_analyzer import (
    AnalysisProgress,
    DataGap,
    LifeChapter,
    LifeStoryReport,
    PlatformBehaviorInsight,
)
from src.core.memory import MediaType, Memory, SourcePlatform
from src.core.timeline import DateRange, Timeline

if TYPE_CHECKING:
    pass

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Placeholder Constants
# =============================================================================


FALLBACK_NARRATIVE_PLACEHOLDER: str = (
    """
[AI-generated narrative unavailable]

This chapter contains {memory_count} memories from {date_range}.

To unlock AI-powered life story analysis with meaningful narratives,
themes, and insights, please configure your Gemini API key:

    organizer config set-key

**Statistics for this period:**
{stats_summary}
""".strip()
)


FALLBACK_SUMMARY_TEMPLATE: str = (
    """
📊 **STATISTICS-ONLY REPORT** (AI Unavailable)

This report was generated without AI analysis. You're seeing
statistics only — the meaningful life story reconstruction requires
Gemini AI.

---

## Your Collection

- **{total_memories}** total memories
- Spanning **{years} years** ({date_range})
- Across **{platform_count} platforms**

---

## Platform Breakdown

{platform_breakdown}

---

## Activity by Year

{yearly_breakdown}

---

## To unlock the full Life Story experience:

```
organizer config set-key
```

## What you're missing:

- ✗ Meaningful chapter detection (life phases, transitions)
- ✗ AI-written narratives for each chapter
- ✗ Cross-platform behavior analysis
- ✗ Pattern and theme detection
- ✗ Executive life story summary
- ✗ Insights about your life journey
""".strip()
)


FALLBACK_CHAPTER_TITLE_TEMPLATE: str = "Year {year}"


FALLBACK_DATA_QUALITY_NOTE: str = (
    "⚠️ This report was generated in FALLBACK MODE without AI analysis. "
    "Chapter boundaries are based on calendar years, not life events. "
    "Narratives, insights, and thematic analysis are unavailable. "
    "To unlock the full experience, configure your Gemini API key with: "
    "organizer config set-key"
)


FALLBACK_INSIGHT_UNAVAILABLE: str = "[Insight unavailable - AI analysis required]"


FALLBACK_PLATFORM_USAGE_PATTERN: str = (
    "Statistics only - AI analysis unavailable. "
    "Configure Gemini API key for behavioral insights."
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FallbackConfig:
    """Configuration for fallback analysis.

    Controls how the statistics-only report is generated.

    Attributes:
        group_by: How to group memories into chapters
                  ("year", "quarter", or "month").
        min_gap_days: Minimum days to consider a gap significant.
        include_basic_stats: Whether to compute full statistics.

    Example:
        >>> config = FallbackConfig(group_by="quarter", min_gap_days=30)
    """

    group_by: Literal["year", "quarter", "month"] = "year"
    min_gap_days: int = 60
    include_basic_stats: bool = True


# =============================================================================
# Fallback Analyzer
# =============================================================================


class FallbackAnalyzer:
    """Statistics-only analyzer for when AI is unavailable.

    Produces a valid but obviously limited report that demonstrates
    the value of AI analysis by showing what's missing.

    The fallback analyzer:
    - Creates year-based chapters (not semantic)
    - Provides statistical summaries only
    - Uses placeholders for narratives
    - Clearly communicates AI is unavailable

    Attributes:
        _config: Fallback configuration.
        _logger: Logger instance.

    Example:
        >>> analyzer = FallbackAnalyzer()
        >>> report = analyzer.analyze(memories)
        >>> assert report.is_fallback is True
    """

    def __init__(self, config: FallbackConfig | None = None) -> None:
        """Initialize the fallback analyzer.

        Args:
            config: Fallback configuration (uses defaults if not provided).
        """
        self._config = config or FallbackConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze(
        self,
        memories: list[Memory],
        progress_callback: Callable[[AnalysisProgress], None] | None = None,
    ) -> LifeStoryReport:
        """Produce a fallback statistics-only report.

        Creates a valid LifeStoryReport with is_fallback=True,
        containing statistics and year-based chapters but no
        AI-generated content.

        Args:
            memories: List of Memory objects to analyze.
            progress_callback: Optional callback for progress updates.

        Returns:
            LifeStoryReport with is_fallback=True.

        Example:
            >>> def on_progress(p):
            ...     print(f"{p.percentage():.0f}%")
            >>>
            >>> report = analyzer.analyze(memories, progress_callback=on_progress)
            >>> print(report.executive_summary)
        """
        start_time = datetime.now(timezone.utc)

        def emit_progress(stage: str, step: int, total: int, message: str) -> None:
            """Emit progress update."""
            if progress_callback:
                progress = AnalysisProgress(
                    stage=stage,
                    current_step=step,
                    total_steps=total,
                    message=message,
                    elapsed_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )
                try:
                    progress_callback(progress)
                except Exception as e:
                    self._logger.warning(f"Progress callback failed: {e}")

        # Progress tracking
        total_steps = 5

        # -----------------------------------------------------------------
        # Step 1: Build Timeline
        # -----------------------------------------------------------------
        emit_progress("preparation", 1, total_steps, "Building timeline")

        timeline = Timeline(memories=memories)

        if len(timeline) == 0:
            return self._create_empty_report()

        # -----------------------------------------------------------------
        # Step 2: Compute Statistics
        # -----------------------------------------------------------------
        emit_progress("statistics", 2, total_steps, "Computing statistics")

        stats = self._compute_statistics(timeline)

        # -----------------------------------------------------------------
        # Step 3: Detect Gaps
        # -----------------------------------------------------------------
        emit_progress("gaps", 3, total_steps, "Detecting gaps")

        gaps = self._detect_gaps_simple(timeline)

        # -----------------------------------------------------------------
        # Step 4: Create Year-Based Chapters
        # -----------------------------------------------------------------
        emit_progress("chapters", 4, total_steps, "Creating chapters")

        if self._config.group_by == "year":
            chapters = self._create_year_chapters(timeline)
        elif self._config.group_by == "month":
            chapters = self._create_month_chapters(timeline)
        else:
            chapters = self._create_quarter_chapters(timeline)

        # -----------------------------------------------------------------
        # Step 5: Assemble Report
        # -----------------------------------------------------------------
        emit_progress("finalization", 5, total_steps, "Assembling report")

        platform_insights = self._compute_platform_stats(list(timeline))

        report = self._create_fallback_report(
            timeline=timeline,
            chapters=chapters,
            stats=stats,
            gaps=gaps,
            platform_insights=platform_insights,
        )

        report.generation_time_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        self._logger.info(
            f"Fallback analysis complete: {len(chapters)} chapters, " f"{len(timeline)} memories"
        )

        return report

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    def _compute_statistics(self, timeline: Timeline) -> dict[str, Any]:
        """Compute comprehensive statistics.

        Args:
            timeline: The timeline to analyze.

        Returns:
            Dictionary of computed statistics.
        """
        memories = list(timeline)
        stats: dict[str, Any] = {}

        # Basic counts
        stats["total_memories"] = len(memories)

        # Memories per year
        by_year: Counter[int] = Counter()
        for m in memories:
            if m.created_at:
                by_year[m.created_at.year] += 1
        stats["by_year"] = dict(sorted(by_year.items()))

        # Memories per month
        by_month: Counter[str] = Counter()
        for m in memories:
            if m.created_at:
                key = f"{m.created_at.year}-{m.created_at.month:02d}"
                by_month[key] += 1
        stats["by_month"] = dict(sorted(by_month.items()))

        # Memories per platform
        by_platform: Counter[str] = Counter()
        for m in memories:
            if m.source_platform:
                by_platform[m.source_platform.value] += 1
        stats["by_platform"] = dict(by_platform.most_common())

        # Memories per media type
        by_type: Counter[str] = Counter()
        for m in memories:
            if m.media_type:
                by_type[m.media_type.value] += 1
        stats["by_media_type"] = dict(by_type.most_common())

        # Date range
        dated = [m for m in memories if m.created_at]
        if dated:
            dates = [m.created_at for m in dated]
            stats["date_range"] = {
                "start": min(dates).date().isoformat(),
                "end": max(dates).date().isoformat(),
            }
            years = max(dates).year - min(dates).year + 1
            stats["years_spanned"] = years
        else:
            stats["date_range"] = {"start": "unknown", "end": "unknown"}
            stats["years_spanned"] = 0

        # Location coverage
        with_location = sum(1 for m in memories if m.location and not m.location.is_empty())
        stats["location_coverage"] = with_location / len(memories) if memories else 0

        # People coverage
        with_people = sum(1 for m in memories if m.people)
        stats["people_coverage"] = with_people / len(memories) if memories else 0

        # Average memories per month
        if by_month:
            stats["avg_per_month"] = len(memories) / len(by_month)
        else:
            stats["avg_per_month"] = 0

        # Most/least active periods
        if by_month:
            most_active = max(by_month.items(), key=lambda x: x[1])
            least_active = min(by_month.items(), key=lambda x: x[1])
            stats["most_active_month"] = {"month": most_active[0], "count": most_active[1]}
            stats["least_active_month"] = {"month": least_active[0], "count": least_active[1]}

        return stats

    def _compute_platform_stats(
        self,
        memories: list[Memory],
    ) -> list[PlatformBehaviorInsight]:
        """Create placeholder platform insights.

        Provides statistics only, no behavioral analysis.

        Args:
            memories: All memories.

        Returns:
            List of stats-only platform insights.
        """
        total = len(memories)
        if total == 0:
            return []

        # Count by platform
        by_platform: Counter[SourcePlatform] = Counter()
        for m in memories:
            if m.source_platform:
                by_platform[m.source_platform] += 1

        insights: list[PlatformBehaviorInsight] = []

        for platform, count in by_platform.most_common():
            insight = PlatformBehaviorInsight(
                platform=platform,
                usage_pattern=FALLBACK_PLATFORM_USAGE_PATTERN,
                peak_period=None,
                unique_characteristics=[],
                memory_count=count,
                percentage_of_total=(count / total) * 100,
            )
            insights.append(insight)

        return insights

    # =========================================================================
    # Chapter Methods
    # =========================================================================

    def _create_year_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Create simple year-based chapters.

        NOT meaningful chapters — just time buckets by calendar year.

        Args:
            timeline: The timeline.

        Returns:
            List of year-based chapters with placeholder narratives.
        """
        chapters: list[LifeChapter] = []

        # Get activity by year
        by_year = timeline.get_activity_by_year()

        for year in sorted(by_year.keys()):
            count = by_year[year]

            # Get memories for this year for IDs
            year_memories = timeline.get_memories_for_year(year)
            memory_ids = [m.id for m in year_memories]

            # Determine actual date range
            dated = [m for m in year_memories if m.created_at]
            if dated:
                dates = [m.created_at for m in dated]
                start = min(dates).date() if hasattr(min(dates), "date") else min(dates)
                end = max(dates).date() if hasattr(max(dates), "date") else max(dates)
            else:
                start = date(year, 1, 1)
                end = date(year, 12, 31)

            # Create stats summary for narrative placeholder
            stats_summary = self._create_chapter_stats_summary(year_memories)

            # Create placeholder narrative
            narrative = FALLBACK_NARRATIVE_PLACEHOLDER.format(
                memory_count=count,
                date_range=f"{start} to {end}",
                stats_summary=stats_summary,
            )

            chapter = LifeChapter(
                title=FALLBACK_CHAPTER_TITLE_TEMPLATE.format(year=year),
                start_date=start,
                end_date=end,
                themes=[],  # No themes in fallback
                narrative=narrative,
                opening_line=f"[{count} memories from {year}]",
                key_events=[FALLBACK_INSIGHT_UNAVAILABLE],
                insights=[FALLBACK_INSIGHT_UNAVAILABLE],
                memory_ids=memory_ids,
                memory_count=count,
                reasoning="Calendar year grouping (fallback mode)",
                confidence=0.1,  # Low confidence - not semantic
            )
            chapters.append(chapter)

        return chapters

    def _create_month_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Create month-based chapters.

        Alternative to year-based for more granular view.

        Args:
            timeline: The timeline.

        Returns:
            List of month-based chapters.
        """
        chapters: list[LifeChapter] = []

        by_month = timeline.get_activity_by_month()

        for month_key in sorted(by_month.keys()):
            count = by_month[month_key]
            if count == 0:
                continue

            # Parse year-month
            year, month = map(int, month_key.split("-"))

            # Get memories for this month
            month_memories = timeline.get_memories_for_month(year, month)
            memory_ids = [m.id for m in month_memories]

            # Month date range
            start = date(year, month, 1)
            if month == 12:
                end = date(year, 12, 31)
            else:
                end = date(year, month + 1, 1) - timedelta(days=1)

            stats_summary = self._create_chapter_stats_summary(month_memories)

            narrative = FALLBACK_NARRATIVE_PLACEHOLDER.format(
                memory_count=count,
                date_range=f"{start} to {end}",
                stats_summary=stats_summary,
            )

            month_name = start.strftime("%B %Y")

            chapter = LifeChapter(
                title=month_name,
                start_date=start,
                end_date=end,
                themes=[],
                narrative=narrative,
                opening_line=f"[{count} memories from {month_name}]",
                key_events=[FALLBACK_INSIGHT_UNAVAILABLE],
                insights=[FALLBACK_INSIGHT_UNAVAILABLE],
                memory_ids=memory_ids,
                memory_count=count,
                reasoning="Calendar month grouping (fallback mode)",
                confidence=0.1,
            )
            chapters.append(chapter)

        return chapters

    def _create_quarter_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Create quarter-based chapters.

        Args:
            timeline: The timeline.

        Returns:
            List of quarter-based chapters.
        """
        chapters: list[LifeChapter] = []

        # Group by quarter
        by_quarter: dict[str, list[Memory]] = {}

        for memory in timeline:
            if memory.created_at:
                year = memory.created_at.year
                quarter = (memory.created_at.month - 1) // 3 + 1
                key = f"{year}-Q{quarter}"
                if key not in by_quarter:
                    by_quarter[key] = []
                by_quarter[key].append(memory)

        for quarter_key in sorted(by_quarter.keys()):
            memories = by_quarter[quarter_key]
            count = len(memories)

            # Parse quarter
            year, q = quarter_key.split("-Q")
            year = int(year)
            quarter = int(q)

            # Quarter date range
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3
            start = date(year, start_month, 1)
            if end_month == 12:
                end = date(year, 12, 31)
            else:
                end = date(year, end_month + 1, 1) - timedelta(days=1)

            memory_ids = [m.id for m in memories]
            stats_summary = self._create_chapter_stats_summary(memories)

            narrative = FALLBACK_NARRATIVE_PLACEHOLDER.format(
                memory_count=count,
                date_range=f"{start} to {end}",
                stats_summary=stats_summary,
            )

            chapter = LifeChapter(
                title=f"Q{quarter} {year}",
                start_date=start,
                end_date=end,
                themes=[],
                narrative=narrative,
                opening_line=f"[{count} memories from Q{quarter} {year}]",
                key_events=[FALLBACK_INSIGHT_UNAVAILABLE],
                insights=[FALLBACK_INSIGHT_UNAVAILABLE],
                memory_ids=memory_ids,
                memory_count=count,
                reasoning="Calendar quarter grouping (fallback mode)",
                confidence=0.1,
            )
            chapters.append(chapter)

        return chapters

    def _create_chapter_stats_summary(self, memories: list[Memory]) -> str:
        """Create stats summary for chapter narrative.

        Args:
            memories: Memories in the chapter.

        Returns:
            Formatted stats string.
        """
        if not memories:
            return "No memories in this period."

        lines = []

        # Media type breakdown
        by_type: Counter[str] = Counter()
        for m in memories:
            if m.media_type:
                by_type[m.media_type.value] += 1

        if by_type:
            lines.append(
                "Media types: " + ", ".join(f"{t}: {c}" for t, c in by_type.most_common(5))
            )

        # Platform breakdown
        by_platform: Counter[str] = Counter()
        for m in memories:
            if m.source_platform:
                by_platform[m.source_platform.value] += 1

        if by_platform:
            lines.append(
                "Platforms: " + ", ".join(f"{p}: {c}" for p, c in by_platform.most_common(5))
            )

        # Location count
        with_location = sum(1 for m in memories if m.location and not m.location.is_empty())
        if with_location:
            lines.append(f"Memories with location: {with_location}")

        return "\n".join(lines) if lines else "Basic statistics only."

    # =========================================================================
    # Gap Detection
    # =========================================================================

    def _detect_gaps_simple(self, timeline: Timeline) -> list[DataGap]:
        """Detect gaps using simple time analysis.

        Finds periods with no data exceeding min_gap_days.

        Args:
            timeline: The timeline.

        Returns:
            List of detected gaps with placeholder explanations.
        """
        # Use timeline's built-in gap detection
        timeline_gaps = timeline.detect_gaps(threshold_days=self._config.min_gap_days)

        data_gaps: list[DataGap] = []

        for gap in timeline_gaps:
            # Determine severity based on duration only
            if gap.duration_days > 365:
                severity: Literal["minor", "moderate", "significant", "major"] = "major"
            elif gap.duration_days > 180:
                severity = "significant"
            elif gap.duration_days > 90:
                severity = "moderate"
            else:
                severity = "minor"

            data_gap = DataGap(
                start_date=gap.start_date,
                end_date=gap.end_date,
                duration_days=gap.duration_days,
                possible_explanations=["Unable to determine - AI analysis unavailable"],
                severity=severity,
                impacts_narrative=False,  # No narrative to impact
            )
            data_gaps.append(data_gap)

        return data_gaps

    # =========================================================================
    # Report Assembly
    # =========================================================================

    def _create_fallback_report(
        self,
        timeline: Timeline,
        chapters: list[LifeChapter],
        stats: dict[str, Any],
        gaps: list[DataGap],
        platform_insights: list[PlatformBehaviorInsight],
    ) -> LifeStoryReport:
        """Assemble the fallback report.

        Args:
            timeline: The timeline.
            chapters: Year-based chapters.
            stats: Computed statistics.
            gaps: Detected gaps.
            platform_insights: Platform stats.

        Returns:
            Complete LifeStoryReport with is_fallback=True.
        """
        # Build platform breakdown for summary
        platform_breakdown_lines = []
        for p, count in stats.get("by_platform", {}).items():
            pct = (count / stats["total_memories"]) * 100 if stats["total_memories"] else 0
            platform_breakdown_lines.append(f"- **{p}**: {count} ({pct:.1f}%)")

        # Build yearly breakdown for summary
        yearly_breakdown_lines = []
        for year, count in stats.get("by_year", {}).items():
            yearly_breakdown_lines.append(f"- **{year}**: {count} memories")

        # Get date range
        date_range = stats.get("date_range", {})
        date_range_str = (
            f"{date_range.get('start', 'unknown')} to {date_range.get('end', 'unknown')}"
        )

        # Create executive summary
        executive_summary = FALLBACK_SUMMARY_TEMPLATE.format(
            total_memories=stats["total_memories"],
            years=stats.get("years_spanned", 0),
            date_range=date_range_str,
            platform_count=len(stats.get("by_platform", {})),
            platform_breakdown="\n".join(platform_breakdown_lines) or "No platform data",
            yearly_breakdown="\n".join(yearly_breakdown_lines) or "No yearly data",
        )

        # Build date range object
        report_date_range = None
        if date_range.get("start") and date_range.get("start") != "unknown":
            try:
                report_date_range = DateRange(
                    start=date.fromisoformat(date_range["start"]),
                    end=date.fromisoformat(date_range["end"]),
                )
            except (ValueError, KeyError):
                pass

        # Create report
        report = LifeStoryReport(
            ai_model="none (fallback mode)",
            analysis_config={"fallback": True, "group_by": self._config.group_by},
            date_range=report_date_range,
            total_memories_analyzed=stats["total_memories"],
            executive_summary=executive_summary,
            chapters=chapters,
            platform_insights=platform_insights,
            detected_patterns=[],  # No patterns in fallback
            data_gaps=gaps,
            data_quality_notes=[FALLBACK_DATA_QUALITY_NOTE],
            is_partial=False,
            partial_failures=[],
            is_fallback=True,  # KEY: Mark as fallback
            tokens_used=0,  # No AI tokens used
        )

        return report

    def _create_empty_report(self) -> LifeStoryReport:
        """Create report for empty input.

        Returns:
            Empty fallback report.
        """
        return LifeStoryReport(
            ai_model="none (fallback mode)",
            analysis_config={"fallback": True},
            total_memories_analyzed=0,
            executive_summary=(
                "📊 **NO MEMORIES FOUND**\n\n"
                "No memories were provided for analysis. "
                "Please ensure you have imported data from at least one source."
            ),
            chapters=[],
            is_fallback=True,
            data_quality_notes=["No memories to analyze"],
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_fallback_report(
    memories: list[Memory],
    config: FallbackConfig | None = None,
) -> LifeStoryReport:
    """Convenience function to create a fallback report.

    Args:
        memories: List of Memory objects.
        config: Optional fallback configuration.

    Returns:
        LifeStoryReport with is_fallback=True.

    Example:
        >>> report = create_fallback_report(memories)
        >>> print(report.is_fallback)  # True
    """
    analyzer = FallbackAnalyzer(config=config)
    return analyzer.analyze(memories)


def is_fallback_mode() -> bool:
    """Check if AI is unavailable (would trigger fallback).

    Returns True if:
    - google.generativeai SDK is not installed
    - AI is disabled in configuration
    - No API key is configured

    Returns:
        True if fallback mode is necessary.

    Example:
        >>> if is_fallback_mode():
        ...     print("Running in statistics-only mode")
        ...     report = create_fallback_report(memories)
        ... else:
        ...     report = analyze_memories(memories)
    """
    try:
        from src.ai.client import GENAI_AVAILABLE, get_client
        from src.config import get_config

        # Check SDK availability
        if not GENAI_AVAILABLE:
            return True

        # Check if AI is enabled
        config = get_config()
        if not config.ai.is_enabled():
            return True

        # Try to get client (checks API key)
        try:
            client = get_client()
            return not client.is_available()
        except Exception:
            return True

    except ImportError:
        return True
