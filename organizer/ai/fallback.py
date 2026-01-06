"""Fallback analyzer for Digital Life Narrative AI.

Provides statistics-only analysis when Gemini AI is unavailable (no API key,
rate limited, errors). The fallback MUST be clearly marked as degraded.

IMPORTANT: The fallback provides statistics only, NOT the rich narrative
analysis. This is intentional – the product value is in the AI.

What fallback provides:
- Total media count by platform
- Date range and temporal distribution
- Location and people frequency
- Media type distribution
- Basic calendar-year "chapters"
- Data gap detection

What fallback does NOT provide:
- Meaningful chapter titles
- Narrative paragraphs
- Thematic analysis
- Cross-platform behavior insights
- Life story interpretation
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from organizer.config import AnalysisConfig
from organizer.models import (
    Confidence,
    DataGap,
    LifeChapter,
    LifeStoryReport,
    MediaItem,
    MediaType,
    SourcePlatform,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fallback Analyzer
# =============================================================================


class FallbackAnalyzer:
    """Statistics-only analyzer when AI is unavailable.

    Generates a degraded LifeStoryReport with is_fallback_mode=True.
    Provides useful statistics but no AI-powered narrative analysis.

    This analyzer is used when:
    - No API key is configured
    - API rate limits are exceeded
    - AI service is unavailable
    - User has enabled local_only_mode

    Attributes:
        config: Analysis configuration settings.

    Example:
        ```python
        fallback = FallbackAnalyzer()
        report = fallback.analyze(media_items)
        assert report.is_fallback_mode == True
        ```
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize the fallback analyzer.

        Args:
            config: Analysis configuration. Uses defaults if None.
        """
        self.config = config or AnalysisConfig()
        logger.debug("FallbackAnalyzer initialized")

    def analyze(self, items: list[MediaItem]) -> LifeStoryReport:
        """Generate a statistics-only report.

        Args:
            items: List of parsed MediaItem objects.

        Returns:
            LifeStoryReport with is_fallback_mode=True.
        """
        logger.info(f"Generating fallback report for {len(items)} items")

        # Sort items by timestamp
        sorted_items = self._sort_items(items)

        # Calculate statistics
        stats = self._calculate_statistics(sorted_items)

        # Generate calendar-year chapters
        chapters = self._generate_yearly_chapters(sorted_items)

        # Detect data gaps
        data_gaps = self._detect_data_gaps(sorted_items)

        # Generate placeholder summary
        executive_summary = self._generate_fallback_summary(sorted_items, stats)

        # Calculate date range
        date_range = None
        if stats["date_range"]["start"] and stats["date_range"]["end"]:
            date_range = (
                stats["date_range"]["start"],
                stats["date_range"]["end"],
            )

        # Compile quality notes
        quality_notes = self._generate_quality_notes(items, stats)

        # Generate statistical patterns (basic, non-AI)
        patterns = self._detect_simple_patterns(stats)

        report = LifeStoryReport(
            generated_at=datetime.now(tz=timezone.utc),
            ai_model_used="none (fallback mode)",
            total_media_analyzed=len(items),
            date_range=date_range,
            executive_summary=executive_summary,
            chapters=chapters,
            platform_insights=[],  # No AI insights available
            detected_patterns=patterns,
            data_gaps=data_gaps,
            data_quality_notes=quality_notes,
            raw_ai_response=None,
            is_fallback_mode=True,
        )

        logger.info(f"Fallback report generated: {len(chapters)} chapters")

        return report

    # =========================================================================
    # Statistics Calculation
    # =========================================================================

    def _calculate_statistics(self, items: list[MediaItem]) -> dict[str, Any]:
        """Calculate comprehensive statistics from media items.

        Args:
            items: Sorted media items.

        Returns:
            Dictionary of statistics.
        """
        stats: dict[str, Any] = {
            "total_count": len(items),
            "by_platform": defaultdict(int),
            "by_type": defaultdict(int),
            "by_year": defaultdict(int),
            "by_month": defaultdict(int),
            "locations": defaultdict(int),
            "people": defaultdict(int),
            "date_range": {"start": None, "end": None},
            "items_with_timestamp": 0,
            "items_with_location": 0,
            "items_with_people": 0,
        }

        dates: list[date] = []

        for item in items:
            # Platform distribution
            stats["by_platform"][item.source_platform.value] += 1

            # Media type distribution
            stats["by_type"][item.media_type.value] += 1

            # Temporal distribution
            if item.timestamp:
                stats["items_with_timestamp"] += 1
                item_date = item.timestamp.date()
                dates.append(item_date)

                stats["by_year"][item.timestamp.year] += 1
                month_key = f"{item.timestamp.year}-{item.timestamp.month:02d}"
                stats["by_month"][month_key] += 1

            # Location frequency
            if item.location:
                stats["items_with_location"] += 1
                loc_name = (
                    item.location.place_name
                    or item.location.country
                    or item.location.raw_location_string
                )
                if loc_name:
                    stats["locations"][loc_name] += 1

            # People frequency
            if item.people:
                stats["items_with_people"] += 1
                for person in item.people:
                    stats["people"][person] += 1

        # Convert defaultdicts
        stats["by_platform"] = dict(stats["by_platform"])
        stats["by_type"] = dict(stats["by_type"])
        stats["by_year"] = dict(stats["by_year"])
        stats["by_month"] = dict(stats["by_month"])
        stats["locations"] = dict(stats["locations"])
        stats["people"] = dict(stats["people"])

        # Date range
        if dates:
            stats["date_range"]["start"] = min(dates)
            stats["date_range"]["end"] = max(dates)

        return stats

    # =========================================================================
    # Chapter Generation
    # =========================================================================

    def _generate_yearly_chapters(
        self,
        items: list[MediaItem],
    ) -> list[LifeChapter]:
        """Generate placeholder chapters by calendar year.

        NO semantic analysis – just calendar year divisions.

        Args:
            items: Sorted media items.

        Returns:
            List of yearly LifeChapter placeholders.
        """
        # Group items by year
        by_year: dict[int, list[MediaItem]] = defaultdict(list)

        for item in items:
            if item.timestamp:
                by_year[item.timestamp.year].append(item)

        chapters = []

        for year in sorted(by_year.keys()):
            year_items = by_year[year]
            chapter = self._generate_fallback_chapter(year, year_items)
            chapters.append(chapter)

        return chapters

    def _generate_fallback_chapter(
        self,
        year: int,
        items: list[MediaItem],
    ) -> LifeChapter:
        """Create a placeholder chapter for a calendar year.

        Args:
            year: The calendar year.
            items: Media items from that year.

        Returns:
            Placeholder LifeChapter with LOW confidence.
        """
        # Calculate some basic stats for the narrative
        platform_counts = defaultdict(int)
        type_counts = defaultdict(int)
        locations: set[str] = set()

        for item in items:
            platform_counts[item.source_platform.value] += 1
            type_counts[item.media_type.value] += 1
            if item.location and item.location.place_name:
                locations.add(item.location.place_name)

        # Build a basic statistical narrative
        narrative_parts = [
            f"⚠️ AI narrative generation unavailable.",
            "",
            f"This year contains {len(items)} media items.",
        ]

        # Add platform breakdown
        if platform_counts:
            platform_str = ", ".join(
                f"{v} from {k}" for k, v in sorted(
                    platform_counts.items(), key=lambda x: x[1], reverse=True
                )
            )
            narrative_parts.append(f"Sources: {platform_str}.")

        # Add type breakdown
        if type_counts:
            type_str = ", ".join(
                f"{v} {k}s" for k, v in sorted(
                    type_counts.items(), key=lambda x: x[1], reverse=True
                )
            )
            narrative_parts.append(f"Content: {type_str}.")

        # Add locations
        if locations:
            loc_list = list(locations)[:5]
            loc_str = ", ".join(loc_list)
            if len(locations) > 5:
                loc_str += f", and {len(locations) - 5} more"
            narrative_parts.append(f"Locations: {loc_str}.")

        narrative_parts.append("")
        narrative_parts.append(
            "To unlock rich narrative analysis, configure your Gemini API key."
        )

        # Get sample media IDs
        sample_ids = [item.id for item in items[:5]]

        # Determine location summary
        location_summary = None
        if locations:
            top_locations = sorted(
                [(loc, sum(1 for i in items if i.location and i.location.place_name == loc))
                 for loc in locations],
                key=lambda x: x[1],
                reverse=True
            )
            if top_locations:
                location_summary = top_locations[0][0]
                if len(top_locations) > 1:
                    location_summary += f" and {len(top_locations) - 1} other locations"

        return LifeChapter(
            title=f"Year {year}",
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            themes=[],  # No thematic analysis without AI
            narrative="\n".join(narrative_parts),
            key_events=[],  # Cannot identify events without AI
            location_summary=location_summary,
            media_count=len(items),
            representative_media_ids=sample_ids,
            confidence=Confidence.LOW,
        )

    # =========================================================================
    # Summary Generation
    # =========================================================================

    def _generate_fallback_summary(
        self,
        items: list[MediaItem],
        stats: dict[str, Any],
    ) -> str:
        """Generate a placeholder executive summary.

        Clearly indicates that AI analysis is unavailable.

        Args:
            items: All media items.
            stats: Calculated statistics.

        Returns:
            Fallback summary string.
        """
        total = stats["total_count"]
        platforms = list(stats["by_platform"].keys())
        platform_count = len(platforms)

        # Date range info
        start_year = "unknown"
        end_year = "unknown"
        if stats["date_range"]["start"] and stats["date_range"]["end"]:
            start_year = stats["date_range"]["start"].year
            end_year = stats["date_range"]["end"].year

        # Platform list
        if platform_count == 1:
            platform_str = platforms[0]
        elif platform_count == 2:
            platform_str = f"{platforms[0]} and {platforms[1]}"
        else:
            platform_str = f"{', '.join(platforms[:-1])}, and {platforms[-1]}"

        summary = f"""⚠️ **Fallback Mode: AI Analysis Unavailable**

This report was generated without AI-powered narrative analysis. Your collection contains **{total:,} media items** spanning **{start_year} to {end_year}** across **{platform_count} platform{'s' if platform_count != 1 else ''}** ({platform_str}).

**What you're missing without AI:**
- Rich, narrative life story chapters with meaningful titles
- Thematic analysis identifying life patterns and transitions  
- Cross-platform behavior insights
- Intelligent gap analysis explaining quiet periods
- A cohesive executive summary of your life journey

**Statistical Summary (available in fallback mode):**
"""

        # Add key statistics
        summary += f"\n- Total items: {total:,}"

        if stats["items_with_timestamp"] < total:
            ts_pct = (stats["items_with_timestamp"] / total) * 100
            summary += f" ({ts_pct:.0f}% with timestamps)"

        if stats["locations"]:
            top_locs = sorted(
                stats["locations"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            loc_str = ", ".join(f"{loc}" for loc, _ in top_locs)
            summary += f"\n- Top locations: {loc_str}"

        if stats["people"]:
            top_people = sorted(
                stats["people"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            people_str = ", ".join(f"{person}" for person, _ in top_people)
            summary += f"\n- Frequently tagged: {people_str}"

        # Add call to action
        summary += """

---

**To unlock AI-powered life story analysis:**
```
organizer configure --set-key
```

Then re-run your analysis to experience the full narrative reconstruction."""

        return summary

    # =========================================================================
    # Data Gap Detection
    # =========================================================================

    def _detect_data_gaps(self, items: list[MediaItem]) -> list[DataGap]:
        """Find significant gaps in the timeline.

        Args:
            items: Sorted media items.

        Returns:
            List of detected data gaps.
        """
        gaps = []
        threshold = timedelta(days=self.config.detect_gaps_threshold_days)

        items_with_dates = [i for i in items if i.timestamp]
        if len(items_with_dates) < 2:
            return gaps

        prev_date: date | None = None

        for item in items_with_dates:
            current_date = item.timestamp.date()

            if prev_date:
                gap_duration = current_date - prev_date

                if gap_duration >= threshold:
                    # In fallback mode, we can't speculate on reasons
                    gap = DataGap(
                        start_date=prev_date,
                        end_date=current_date,
                        gap_days=gap_duration.days,
                        possible_reasons=[
                            "AI analysis unavailable - enable Gemini to get gap insights"
                        ],
                    )
                    gaps.append(gap)

            prev_date = current_date

        return gaps

    # =========================================================================
    # Pattern Detection (Simple)
    # =========================================================================

    def _detect_simple_patterns(self, stats: dict[str, Any]) -> list[str]:
        """Detect basic patterns from statistics.

        NO semantic analysis – just obvious statistical patterns.

        Args:
            stats: Calculated statistics.

        Returns:
            List of detected patterns.
        """
        patterns = []

        # Platform dominance
        if stats["by_platform"]:
            total = stats["total_count"]
            for platform, count in stats["by_platform"].items():
                if count > total * 0.7:
                    patterns.append(f"Primary platform: {platform} ({count}/{total} items)")
                    break

        # Seasonal patterns (very basic)
        if stats["by_month"]:
            months = list(stats["by_month"].values())
            if len(months) >= 12:
                avg = sum(months) / len(months)
                high_months = sum(1 for m in months if m > avg * 1.5)
                if high_months >= 2:
                    patterns.append("Seasonal variation detected in activity levels")

        # Media type preference
        if stats["by_type"]:
            total = stats["total_count"]
            for media_type, count in stats["by_type"].items():
                if count > total * 0.8:
                    patterns.append(f"Predominantly {media_type} content")
                    break

        # Location diversity
        loc_count = len(stats["locations"])
        if loc_count > 20:
            patterns.append(f"Geographically diverse: {loc_count} distinct locations")
        elif loc_count == 1:
            patterns.append("Single primary location")

        # Add fallback notice
        patterns.append("ℹ️ Advanced pattern recognition requires AI analysis")

        return patterns

    # =========================================================================
    # Quality Notes
    # =========================================================================

    def _generate_quality_notes(
        self,
        items: list[MediaItem],
        stats: dict[str, Any],
    ) -> list[str]:
        """Generate data quality notes.

        Args:
            items: All media items.
            stats: Calculated statistics.

        Returns:
            List of quality notes.
        """
        notes = []
        total = len(items)

        if total == 0:
            return ["No media items to analyze"]

        # Timestamp coverage
        ts_count = stats["items_with_timestamp"]
        ts_pct = (ts_count / total) * 100
        if ts_pct < 50:
            notes.append(f"⚠️ Low timestamp coverage: {ts_pct:.0f}% of items have dates")
        elif ts_pct < 80:
            notes.append(f"Partial timestamp coverage: {ts_pct:.0f}%")
        else:
            notes.append(f"Good timestamp coverage: {ts_pct:.0f}%")

        # Location coverage
        loc_count = stats["items_with_location"]
        loc_pct = (loc_count / total) * 100
        if loc_pct < 10:
            notes.append(f"Limited location data: {loc_pct:.0f}%")
        elif loc_pct > 40:
            notes.append(f"Good location coverage: {loc_pct:.0f}%")

        # People tags
        people_count = stats["items_with_people"]
        if people_count > 0:
            people_pct = (people_count / total) * 100
            notes.append(f"People tagged in {people_pct:.0f}% of items")

        # Confidence levels
        low_conf = sum(
            1 for i in items if i.timestamp_confidence == Confidence.LOW
        )
        if low_conf > total * 0.3:
            notes.append("Many items have low-confidence timestamps")

        # Fallback mode notice
        notes.append("ℹ️ Detailed quality analysis available with AI enabled")

        return notes

    # =========================================================================
    # Utilities
    # =========================================================================

    def _sort_items(self, items: list[MediaItem]) -> list[MediaItem]:
        """Sort items by timestamp.

        Args:
            items: Unsorted items.

        Returns:
            Sorted items (items without timestamps at end).
        """
        def sort_key(item: MediaItem) -> tuple[int, datetime]:
            if item.timestamp:
                return (0, item.timestamp)
            return (1, datetime.min.replace(tzinfo=timezone.utc))

        return sorted(items, key=sort_key)


# =============================================================================
# Module-Level Functions
# =============================================================================


def generate_fallback_report(
    items: list[MediaItem],
    config: AnalysisConfig | None = None,
) -> LifeStoryReport:
    """Convenience function to generate a fallback report.

    Args:
        items: Media items to analyze.
        config: Optional analysis configuration.

    Returns:
        Fallback LifeStoryReport.
    """
    analyzer = FallbackAnalyzer(config)
    return analyzer.analyze(items)


def is_fallback_mode(report: LifeStoryReport) -> bool:
    """Check if a report was generated in fallback mode.

    Args:
        report: The life story report.

    Returns:
        True if report is from fallback mode.
    """
    return report.is_fallback_mode
