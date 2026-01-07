"""Life Story Analyzer for Digital Life Narrative AI.

This is the CORE PRODUCT — connects Memory → Timeline → PrivacyGate → AI → Report.

The analyzer takes normalized Memory objects and uses Gemini to generate a
comprehensive life story with chapters, narratives, and insights.

Flow:
1. Validate memories (sufficient data, privacy checks)
2. Build Timeline (temporal analysis, gap detection)
3. Privacy filtering (PrivacyGate controls what gets sent)
4. AI analysis (chapter detection, narrative generation)
5. Report generation (executive summary, quality assessment)

Example:
    >>> from src.ai import LifeStoryAnalyzer
    >>> from src.core import Memory
    >>>
    >>> analyzer = LifeStoryAnalyzer()
    >>> memories = [...]  # List of Memory objects
    >>> report = analyzer.analyze(memories)
    >>> print(report.executive_summary)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.ai.client import (
    AIClient,
    AIResponse,
    AIUnavailableError,
    StructuredResponse,
    get_client,
    request_consent,
)
from src.config import get_config
from src.core import (
    Memory,
    PrivacyGate,
    Timeline,
    TimelineStatistics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class LifeChapter(BaseModel):
    """A distinct chapter or phase in someone's life.

    Attributes:
        title: Creative chapter title (e.g., "The Chicago Years")
        start_date: Chapter start date
        end_date: Chapter end date
        themes: Key themes identified in this chapter
        narrative: Generated narrative (2-3 paragraphs)
        key_events: Notable events/moments
        location_summary: Primary location(s) during this period
        media_count: Number of memories in this chapter
        confidence: AI confidence in this chapter detection
    """

    title: str
    start_date: date
    end_date: date
    themes: list[str] = Field(default_factory=list)
    narrative: str = ""
    key_events: list[str] = Field(default_factory=list)
    location_summary: str | None = None
    media_count: int = 0
    confidence: str = "medium"  # low, medium, high


class LifeStoryReport(BaseModel):
    """Complete life story analysis report.

    Attributes:
        generated_at: When this report was created
        ai_model: Model used for generation
        total_memories: Total memories analyzed
        date_range: Earliest to latest memory
        executive_summary: Opening narrative of the life story
        chapters: Detected life chapters with narratives
        timeline_stats: Statistical timeline summary
        data_quality_notes: Issues detected in data
        is_fallback: Whether this used AI or fallback mode
    """

    generated_at: datetime
    ai_model: str
    total_memories: int
    date_range: tuple[date, date] | None = None
    executive_summary: str
    chapters: list[LifeChapter]
    timeline_stats: dict[str, Any] = Field(default_factory=dict)
    data_quality_notes: list[str] = Field(default_factory=list)
    is_fallback: bool = False


# =============================================================================
# Exceptions
# =============================================================================


class AnalysisError(Exception):
    """Base exception for analysis errors."""

    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's not enough data for meaningful analysis."""

    pass


# =============================================================================
# Analyzer
# =============================================================================


class LifeStoryAnalyzer:
    """Generates life stories from Memory objects using AI.

    This is the main product component that orchestrates:
    - Timeline construction
    - Privacy filtering
    - AI-powered chapter detection
    - Narrative generation
    - Report assembly

    Example:
        >>> analyzer = LifeStoryAnalyzer()
        >>> memories = load_memories()
        >>> report = analyzer.analyze(memories, progress_callback=print_progress)
        >>> save_report(report)
    """

    MIN_MEMORIES_FOR_ANALYSIS = 10
    MAX_MEMORIES_FOR_SAMPLE = 200

    def __init__(
        self,
        client: AIClient | None = None,
        privacy_gate: PrivacyGate | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            client: AI client. If None, creates one from config.
            privacy_gate: Privacy gate. If None, uses default.

        Raises:
            AIUnavailableError: If AI is disabled and no fallback.
        """
        self.config = get_config()
        self.client = client or get_client()
        self.privacy_gate = privacy_gate or PrivacyGate()
        self._logger = logging.getLogger(f"{__name__}.LifeStoryAnalyzer")

    def analyze(
        self,
        memories: list[Memory],
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> LifeStoryReport:
        """Analyze memories and generate a life story report.

        Main entry point for life story generation.

        Args:
            memories: List of Memory objects to analyze.
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, percent_complete).

        Returns:
            Complete LifeStoryReport with chapters and narratives.

        Raises:
            InsufficientDataError: If not enough memories for analysis.
            AIUnavailableError: If AI is required but unavailable.

        Example:
            >>> def on_progress(stage, percent):
            ...     print(f"{stage}: {percent:.0f}%")
            >>>
            >>> report = analyzer.analyze(memories, on_progress)
        """

        def report_progress(stage: str, percent: float) -> None:
            if progress_callback:
                progress_callback(stage, percent)

        report_progress("Initializing", 0.0)

        # Validate input
        if len(memories) < self.MIN_MEMORIES_FOR_ANALYSIS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_MEMORIES_FOR_ANALYSIS} memories, " f"got {len(memories)}"
            )

        # Check if AI is available
        if not self.client.is_available():
            self._logger.warning("AI unavailable, generating fallback report")
            return self._create_fallback_report(memories)

        # Check consent
        if self.config.ai.require_consent:
            if not request_consent():
                self._logger.info("User declined AI consent, generating fallback")
                return self._create_fallback_report(memories)

        # Build timeline
        report_progress("Building Timeline", 10.0)
        timeline = Timeline(memories)
        stats = timeline.compute_statistics()

        # Detect chapters
        report_progress("Detecting Life Chapters", 30.0)
        try:
            chapters = self._detect_chapters(timeline)
        except Exception as e:
            self._logger.error(f"Chapter detection failed: {e}")
            chapters = self._create_fallback_chapters(timeline)

        # Generate narratives
        report_progress("Writing Narratives", 60.0)
        chapters = self._generate_chapter_narratives(chapters, timeline)

        # Generate executive summary
        report_progress("Creating Summary", 85.0)
        try:
            exec_summary = self._generate_executive_summary(chapters, stats)
        except Exception as e:
            self._logger.error(f"Executive summary failed: {e}")
            exec_summary = self._create_fallback_summary(chapters)

        # Assess data quality
        report_progress("Finalizing", 95.0)
        quality_notes = self._assess_data_quality(timeline)

        # Assemble report
        date_range = None
        if stats.earliest_memory and stats.latest_memory:
            date_range = (stats.earliest_memory.date(), stats.latest_memory.date())

        report = LifeStoryReport(
            generated_at=datetime.now(timezone.utc),
            ai_model=self.config.ai.model_name,
            total_memories=len(memories),
            date_range=date_range,
            executive_summary=exec_summary,
            chapters=chapters,
            timeline_stats=self._stats_to_dict(stats),
            data_quality_notes=quality_notes,
            is_fallback=False,
        )

        report_progress("Complete", 100.0)
        self._logger.info(f"Analysis complete: {len(chapters)} chapters, {len(memories)} memories")

        return report

    def _detect_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Detect life chapters using AI.

        Args:
            timeline: Timeline object with memories.

        Returns:
            List of detected LifeChapter objects.
        """
        # Sample memories for the AI prompt
        memories = timeline.memories[: self.MAX_MEMORIES_FOR_SAMPLE]

        # Prepare safe payloads through privacy gate
        safe_payloads = []
        for memory in memories:
            payload = memory.to_ai_payload(privacy_mode=self.config.privacy.mode.value)
            safe_payloads.append(payload)

        # Build prompt
        stats = timeline.compute_statistics()
        years_covered = stats.years_covered
        min_chapters = max(2, years_covered // 5)
        max_chapters = min(10, max(5, years_covered // 2))

        prompt = f"""Analyze this media timeline and identify {min_chapters} to {max_chapters} distinct life chapters.

Timeline Summary:
- Total memories: {stats.total_memories}
- Date range: {stats.earliest_memory} to {stats.latest_memory}
- Years covered: {years_covered}
- Platforms: {len(stats.platform_counts)} different sources

Sample Memories (chronological):
{json.dumps(safe_payloads[:50], indent=2, default=str)}

For each chapter, provide:
1. A creative, descriptive title (e.g., "The Chicago Years")
2. Start and end dates (YYYY-MM-DD format)
3. Key themes (1-5 themes)
4. Location summary if discernible
5. Confidence level: "high", "medium", or "low"
6. Brief reasoning for this chapter boundary

Respond with a JSON array of chapters."""

        system_instruction = """You are a life historian AI analyzing someone's personal media timeline.
Identify distinct chapters or phases based on patterns:
- Moving to new locations
- Changes in who appears in photos
- Shifts in activity patterns or content types
- Temporal clustering suggesting different life phases

Be thoughtful and respectful of the personal nature of this data."""

        # Call AI
        response = self.client.generate_structured(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        if not response.parse_success:
            self._logger.warning(f"JSON parse failed: {response.parse_error}")
            raise AnalysisError(f"Failed to parse chapter response: {response.parse_error}")

        # Convert to LifeChapter objects
        chapters_data = response.data
        if isinstance(chapters_data, dict) and "chapters" in chapters_data:
            chapters_data = chapters_data["chapters"]

        chapters = []
        for idx, chapter_dict in enumerate(chapters_data):
            try:
                chapter = LifeChapter(
                    title=chapter_dict.get("title", f"Chapter {idx + 1}"),
                    start_date=self._parse_date(chapter_dict.get("start_date", "")),
                    end_date=self._parse_date(chapter_dict.get("end_date", "")),
                    themes=chapter_dict.get("themes", []),
                    location_summary=chapter_dict.get("location_summary"),
                    confidence=chapter_dict.get("confidence", "medium"),
                )

                # Count memories in this chapter
                chapter.media_count = sum(
                    1
                    for m in timeline.memories
                    if m.created_at
                    and chapter.start_date <= m.created_at.date() <= chapter.end_date
                )

                chapters.append(chapter)
            except Exception as e:
                self._logger.warning(f"Failed to parse chapter {idx}: {e}")
                continue

        return sorted(chapters, key=lambda c: c.start_date)

    def _generate_chapter_narratives(
        self,
        chapters: list[LifeChapter],
        timeline: Timeline,
    ) -> list[LifeChapter]:
        """Generate narratives for each chapter.

        Args:
            chapters: Chapters to generate narratives for.
            timeline: Timeline with all memories.

        Returns:
            Chapters with narratives filled in.
        """
        for chapter in chapters:
            try:
                # Get memories for this chapter
                chapter_memories = [
                    m
                    for m in timeline.memories
                    if m.created_at
                    and chapter.start_date <= m.created_at.date() <= chapter.end_date
                ]

                if not chapter_memories:
                    chapter.narrative = (
                        f"No detailed data available for this period "
                        f"({chapter.start_date} to {chapter.end_date})."
                    )
                    continue

                # Sample and prepare for AI
                sample = chapter_memories[:50]
                safe_payloads = [
                    m.to_ai_payload(privacy_mode=self.config.privacy.mode.value) for m in sample
                ]

                prompt = f"""Write a narrative for this life chapter:

Chapter: {chapter.title}
Period: {chapter.start_date} to {chapter.end_date}
Themes: {", ".join(chapter.themes) if chapter.themes else "Not yet determined"}
Location: {chapter.location_summary or "Various locations"}
Media Count: {chapter.media_count} items

Sample Moments:
{json.dumps(safe_payloads[:20], indent=2, default=str)}

Write 2-3 paragraphs that:
1. Capture the essence of this life phase
2. Highlight key patterns, activities, or moments
3. Suggest the emotional arc or growth during this period
4. Note any transitions or turning points if evident

Also identify 3-5 key events or moments that define this chapter.

Respond with JSON:
{{
  "narrative": "Your 2-3 paragraph narrative here...",
  "key_events": ["Event 1", "Event 2", "Event 3"],
  "emotional_arc": "Brief description of emotional journey"
}}"""

                system_instruction = """You are a skilled biographer and life storyteller.
Write in third person, as if telling their story to someone who wants to understand
this phase of their life. Be specific where the data allows, but respectful of privacy.
Focus on the human experience behind the media."""

                response = self.client.generate_structured(
                    prompt=prompt,
                    system_instruction=system_instruction,
                )

                if response.parse_success:
                    chapter.narrative = response.data.get("narrative", "")
                    chapter.key_events = response.data.get("key_events", [])
                else:
                    chapter.narrative = (
                        f"This chapter spans from {chapter.start_date} to "
                        f"{chapter.end_date}, containing {chapter.media_count} "
                        f"media items."
                    )

            except Exception as e:
                self._logger.warning(f"Narrative generation failed for '{chapter.title}': {e}")
                chapter.narrative = (
                    f"This chapter spans from {chapter.start_date} to {chapter.end_date}, "
                    f"containing {chapter.media_count} media items."
                )

        return chapters

    def _generate_executive_summary(
        self,
        chapters: list[LifeChapter],
        stats: TimelineStatistics,
    ) -> str:
        """Generate executive summary of the life story.

        Args:
            chapters: All chapters.
            stats: Timeline statistics.

        Returns:
            Executive summary text.
        """
        # Prepare chapter summaries
        chapter_summaries = []
        for chapter in chapters:
            chapter_summaries.append(
                {
                    "title": chapter.title,
                    "period": f"{chapter.start_date} to {chapter.end_date}",
                    "themes": chapter.themes,
                    "media_count": chapter.media_count,
                }
            )

        prompt = f"""Create an executive summary of this person's life story:

Life Chapters:
{json.dumps(chapter_summaries, indent=2, default=str)}

Key Statistics:
- Total memories: {stats.total_memories}
- Time span: {stats.earliest_memory} to {stats.latest_memory}
- Years covered: {stats.years_covered}
- Platforms: {list(stats.platform_counts.keys())}

Write a cohesive 3-5 paragraph life story summary that:
1. Opens with a compelling hook about their journey
2. Traces major arcs and transitions
3. Highlights defining themes across their life
4. Notes patterns in how they document their life
5. Ends with a forward-looking or reflective note

This is the opening of their life story book - make it count."""

        system_instruction = """You are a master biographer tasked with writing the opening
summary of someone's life story. Drawing from chapter summaries and statistical data,
craft a compelling narrative that captures the essence of their journey."""

        response = self.client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        return response.text

    def _create_fallback_report(self, memories: list[Memory]) -> LifeStoryReport:
        """Create a basic report without AI.

        Args:
            memories: List of memories.

        Returns:
            LifeStoryReport in fallback mode.
        """
        timeline = Timeline(memories)
        stats = timeline.compute_statistics()

        # Create yearly chapters
        chapters = self._create_fallback_chapters(timeline)

        # Simple summary
        exec_summary = (
            f"This collection contains {len(memories)} memories spanning "
            f"{stats.years_covered} years, from {stats.earliest_memory} to "
            f"{stats.latest_memory}. "
            f"Memories are organized into {len(chapters)} chapters. "
            f"\n\nNote: This is a basic statistical summary. For AI-powered "
            f"narrative reconstruction, please enable AI features and configure "
            f"your Gemini API key."
        )

        date_range = None
        if stats.earliest_memory and stats.latest_memory:
            date_range = (stats.earliest_memory.date(), stats.latest_memory.date())

        return LifeStoryReport(
            generated_at=datetime.now(timezone.utc),
            ai_model="none (fallback mode)",
            total_memories=len(memories),
            date_range=date_range,
            executive_summary=exec_summary,
            chapters=chapters,
            timeline_stats=self._stats_to_dict(stats),
            data_quality_notes=self._assess_data_quality(timeline),
            is_fallback=True,
        )

    def _create_fallback_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Create simple yearly chapters as fallback.

        Args:
            timeline: Timeline object.

        Returns:
            List of yearly chapters.
        """
        chapters = []
        memories_by_year: dict[int, list[Memory]] = defaultdict(list)

        for memory in timeline.memories:
            if memory.created_at:
                memories_by_year[memory.created_at.year].append(memory)

        for year in sorted(memories_by_year.keys()):
            year_memories = memories_by_year[year]
            chapter = LifeChapter(
                title=f"Year {year}",
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                themes=[],
                narrative=(f"This chapter contains {len(year_memories)} media items from {year}."),
                key_events=[],
                location_summary=None,
                media_count=len(year_memories),
                confidence="low",
            )
            chapters.append(chapter)

        return chapters

    def _create_fallback_summary(self, chapters: list[LifeChapter]) -> str:
        """Create fallback executive summary.

        Args:
            chapters: List of chapters.

        Returns:
            Simple summary text.
        """
        return (
            f"This life story contains {len(chapters)} chapters. "
            f"The narrative spans from {chapters[0].start_date} to "
            f"{chapters[-1].end_date}."
        )

    def _assess_data_quality(self, timeline: Timeline) -> list[str]:
        """Assess data quality and identify issues.

        Args:
            timeline: Timeline object.

        Returns:
            List of quality notes/warnings.
        """
        notes = []
        stats = timeline.compute_statistics()

        # Check for missing timestamps
        missing_timestamps = sum(1 for m in timeline.memories if not m.created_at)
        if missing_timestamps > 0:
            pct = (missing_timestamps / stats.total_memories) * 100
            notes.append(f"{missing_timestamps} memories ({pct:.1f}%) are missing timestamps")

        # Check for gaps
        gaps = timeline.detect_gaps(min_gap_days=90)
        if gaps:
            notes.append(f"Detected {len(gaps)} significant gaps in timeline")

        # Check for single platform dominance
        if stats.platform_counts:
            max_platform = max(stats.platform_counts.values())
            if max_platform / stats.total_memories > 0.9:
                notes.append("Data heavily concentrated in one platform")

        return notes

    def _stats_to_dict(self, stats: TimelineStatistics) -> dict[str, Any]:
        """Convert TimelineStatistics to dict.

        Args:
            stats: TimelineStatistics object.

        Returns:
            Dictionary representation.
        """
        return {
            "total_memories": stats.total_memories,
            "years_covered": stats.years_covered,
            "earliest_memory": str(stats.earliest_memory) if stats.earliest_memory else None,
            "latest_memory": str(stats.latest_memory) if stats.latest_memory else None,
            "platform_counts": dict(stats.platform_counts),
            "media_type_counts": dict(stats.media_type_counts),
        }

    def _parse_date(self, date_str: str) -> date:
        """Parse a date string.

        Args:
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            Parsed date.

        Raises:
            ValueError: If date cannot be parsed.
        """
        if not date_str:
            raise ValueError("Empty date string")

        # Try ISO format
        try:
            return datetime.fromisoformat(date_str).date()
        except ValueError:
            pass

        # Try just year
        try:
            year = int(date_str[:4])
            return date(year, 1, 1)
        except (ValueError, IndexError):
            pass

        raise ValueError(f"Could not parse date: {date_str}")
