"""Life Story Analyzer for Digital Life Narrative AI.

THIS IS THE CORE OF THE PRODUCT. Takes normalized MediaItem data and uses
Gemini to generate the Life Story Report. Implements all AI-driven
narrative reconstruction with sophisticated prompt engineering.

The analyzer:
1. Preprocesses and privacy-filters media items
2. Generates temporal summaries for context
3. Detects distinct life chapters via AI
4. Generates narratives for each chapter
5. Analyzes cross-platform behavior patterns
6. Creates an executive summary of the life story
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from PIL import Image

from google.genai import types

from organizer.ai.client import (
    AIClient,
    AIClientError,
    AIRequestError,
    get_client,
)
from organizer.config import PrivacySettings
from organizer.ai.client import AIClient, AIResponse, estimate_tokens
from organizer.ai.usage_tracker import get_tracker, UsageTracker
from organizer.ai.cache import AICache
from organizer.models import (
    AnalysisConfig,
    Confidence,
    DataGap,
    LifeChapter,
    LifeStoryReport,
    MediaItem,
    MediaType,
    PlatformBehaviorInsight,
    SourcePlatform,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class AnalysisError(Exception):
    """Base exception for analysis errors."""

    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's not enough data for meaningful analysis."""

    pass


class AINotAvailableError(AnalysisError):
    """Raised when AI service is completely unavailable."""

    pass


# =============================================================================
# Prompt Templates
# =============================================================================


CHAPTER_DETECTION_SYSTEM_PROMPT = """You are a life historian AI analyzing someone's personal media timeline. Your task is to identify distinct chapters or phases in their life based on patterns in their media collection.

Analyze both the metadata (dates, locations, folders) and the provided visual samples to find natural breakpoints:
- **Visual Context**: Look for recurring settings (beach, office, city), people groups, and atmospheric changes (festive, professional, outdoorsy).
- **Thematic Shifts**: Changes in what they are photographing or doing (actions like hiking, cooking, traveling).
- **Relational Landmarks**: Changes in who appears in photos (new friends, family events).
- **Temporal Clustering**: Shifts in documentation frequency suggesting different life phases.

Output your analysis as valid JSON only. Be thoughtful and respectful of the personal nature of this data."""

CHAPTER_DETECTION_USER_PROMPT = """Based on this chronological media timeline summary and samples, identify the core 'chapters' of this person's life journey. 

## Timeline Summary
{temporal_summary}

## Sample Items (Pixels & Discovery)
{sample_items}

## Instructions
Identify {min_chapters} to {max_chapters} distinct life chapters. 

CRITICAL: Do not just summarize dates. **Discover the story** from the evidence.
Look for:
1. **Thematic Shifts**: Changes in what they are photographing. Look at `context_folder` and `file_name` (e.g., shifts from "Work" to "Wedding" folder, or "IMG_123" to "Snapchat").
2. **Location Landmarks**: Moving from one city or area to another.
3. **Activity Clusters**: A sudden focus on travel, a new hobby, or a specific group of people.
4. **Platform Shifts**: Moving from DSLR photography to casual Snapchat memories.

For each chapter, provide:
1. A poetic, evocative title (e.g., "The Nomadic Summer", "Foundations in Seattle", "The Creative Renaissance")
2. Precise start and end dates (YYYY-MM-DD)
3. A list of 3-5 sub-themes
4. A location summary
5. A 'reasoning' block explaining the **archaeological evidence** (specific folders, filenames, or content clusters) that suggested this breakpoint.

Respond with valid JSON only:
```json
[
  {{
    "title": "Evocative Title",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "themes": ["theme1", "theme2"],
    "location_summary": "Location name or null",
    "confidence": "high|medium|low",
    "reasoning": "Archeological evidence: shift from local photos in 'Home' folder to international travel metadata in 'Europe2024' folder and new face clusters."
  }}
]
```"""

NARRATIVE_SYSTEM_PROMPT = """You are a skilled biographer and life storyteller with advanced visual intelligence. Given information about a chapter in someone's life and representative photos, write a warm, engaging narrative that brings this period to life.

Capture the "Visual Spirit" of the chapter:
- **Objects & Actions**: Mention significant objects (musical instruments, cars, pets) or actions (celebrating, gardening, trekking) you see in the photos.
- **Atmosphere**: Describe the overall aesthetic and setting (urban grit, coastal peace, cozy interiors).
- **Human Connection**: Note the presence of people, their interactions, and the general community or solitude of the phase.

Write in third person, as if telling their story to someone who wants to understand this phase of their life. Be specific where the visual and metadata evidence allow. Avoid generic filler; if you see a specific kind of camera or a specific mountain range, mention it."""

NARRATIVE_USER_PROMPT = """Write a narrative for this life chapter, discovered from their digital collection:

## Chapter: {title}
- Period: {start_date} to {end_date}
- Themes: {themes}
- Location: {location}
- Media Count: {media_count} items

## Archaeological Evidence (Metadata & Visual Cues)
{sample_items}

## Instructions
Write 2-3 paragraphs that:
1. Capture the human essence of this life phase. Avoid generic corporate speak.
2. Highlight specific patterns found in the **folder names, filenames, and captions**. (e.g., "The transition into the 'University' folder marks a shift...")
3. Connect the metadata to the human journey. If there are 500 photos in one week, what does that say?
4. Suggest the emotional arc—from quiet introspection to vibrant social activity.

Also identify 3-5 key events or moments that define this chapter based on burst activity or specific filenames.

Respond with JSON:
```json
{{
  "narrative": "Your 2-3 paragraph biography here...",
  "key_events": ["Event 1", "Event 2", "Event 3"],
  "emotional_arc": "Brief description of emotional journey"
}}
```"""

PLATFORM_ANALYSIS_SYSTEM_PROMPT = """You are a digital behavior analyst. Analyze how someone uses different platforms to document their life, identifying distinct usage patterns and purposes for each platform."""

PLATFORM_ANALYSIS_USER_PROMPT = """Analyze the platform usage patterns from this media collection:

## Platform Statistics
{platform_stats}

## Sample Items by Platform
{platform_samples}

For each platform, describe:
1. How this person uses the platform (purpose, content type)
2. Peak usage periods (years)
3. What makes their usage of this platform unique
4. Common content types on this platform

Respond with JSON array:
```json
[
  {{
    "platform": "platform_name",
    "usage_pattern": "Description of how they use this platform",
    "peak_years": [2019, 2020],
    "common_content_types": ["photo", "video"],
    "unique_aspects": ["aspect1", "aspect2"]
  }}
]
```"""

EXECUTIVE_SUMMARY_SYSTEM_PROMPT = """You are a master biographer and digital archaeologist. Your task is to write a deeply moving and evocative opening summary of someone's life story based on the digital trail they've left behind.

You weave together statistical facts (pixels, metadata, counts) with the human essence revealed in their chapters. 

**Archeological Discovery Instructions**:
- **Visual Synthesis**: Connect visual landmarks (objects, actions, atmosphere) discovered across chapters into a cohesive life journey.
- **Evidence-Based Narrative**: Mention how specific evidence (folder names, platform shifts, metadata density) reveals human growth and change.
- **The Voice**: Your tone is literary, reflective, and observant. Focus on the human experience behind the bits and pixels.

Write 5-6 paragraphs that serve as the definitive summary of this person's digital legacy."""

EXECUTIVE_SUMMARY_USER_PROMPT = """Write a definitive, 5-paragraph life story summary based on this digital collection:

## Life Chapters
{chapters_summary}

## Key Statistics
{statistics}

## Platform Insights
{platform_insights}

## Instructions
1. **Paragraph 1: The Hook.** Open with a compelling observation about this person's journey. Describe the "digital fingerprint" they’ve left and the era it covers.
2. **Paragraph 2: The Arc.** Trace the major transitions. How did they move from one phase to the next? Use the metadata (locations, years) to ground the story.
3. **Paragraph 3: The Themes.** Identify the core threads that bind these years together—family, travel, career, or personal evolution.
4. **Paragraph 4: The Medium.** Reflect on how they documented their life (Snapchat vs Google Photos, etc.) and what that says about their priorities and the way they saw the world.
5. **Paragraph 5: Reflection.** End with a poignant summary of who this person was during this decade and the legacy these 2,000+ memories create.

Write in a warm, professional, and slightly poetic third-person narrative. Make it feel like a cherished biography found among their files."""

GAP_ANALYSIS_PROMPT = """Given these gaps in the media timeline, suggest possible reasons for each gap:

{gaps}

For each gap, consider:
- Life transitions (moving, job changes)
- Platform changes (stopped using one service)
- Life events (busy periods, difficult times)
- Technical reasons (lost data, device changes)

Respond with JSON array matching the gaps:
```json
[
  {{
    "gap_number": 1,
    "possible_reasons": ["reason1", "reason2"]
  }}
]
```"""


# =============================================================================
# Life Story Analyzer
# =============================================================================


class LifeStoryAnalyzer:
    """Core analyzer for generating life stories from media data.

    Takes normalized MediaItem data and uses Gemini AI to generate
    a comprehensive LifeStoryReport with chapters, narratives, and insights.

    Attributes:
        client: The AI client for Gemini communication.
        config: Analysis configuration settings.
        privacy: Privacy settings for data handling.

    Example:
        ```python
        analyzer = LifeStoryAnalyzer()
        report = await analyzer.analyze(media_items)
        print(report.executive_summary)
        ```
    """

    MIN_ITEMS_FOR_ANALYSIS = 10
    ITEMS_PER_CHAPTER_SAMPLE = 50
    MAX_ITEMS_FOR_PROMPT = 200

    def __init__(
        self,
        client: AIClient | None = None,
        config: AnalysisConfig | None = None,
        privacy: PrivacySettings | None = None,
        cache_enabled: bool = True,
    ) -> None:
        """Initialize the Life Story Analyzer.

        Args:
            client: AI client for Gemini. Creates one if None.
            config: Analysis configuration. Uses defaults if None.
            privacy: Privacy settings. Uses defaults if None.
            cache_enabled: Whether to enable AI response caching.

        Raises:
            AINotAvailableError: If AI client cannot be initialized.
        """
        self.config = config or AnalysisConfig()
        self.privacy = privacy or PrivacySettings()
        self.cache = AICache(enabled=cache_enabled)
        self._media_fingerprint: str | None = None
        self._config_fingerprint: str | None = None

        if client is None:
            try:
                # Use AI settings from AppConfig if available
                ai_settings = getattr(self.config, "ai", None)
                self.client = get_client(settings=ai_settings)
            except AIClientError as e:
                logger.error(f"Failed to initialize AI client: {e}")
                raise AINotAvailableError(f"AI service unavailable: {e}")
        else:
            self.client = client

        logger.debug("LifeStoryAnalyzer initialized")

    async def analyze(
        self,
        items: list[MediaItem],
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> LifeStoryReport:
        """Analyze media items and generate a life story report.

        This is the main entry point for life story generation.

        Args:
            items: List of parsed MediaItem objects to analyze.
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, percent_complete).

        Returns:
            Complete LifeStoryReport with chapters and narratives.

        Raises:
            InsufficientDataError: If not enough items for analysis.
            AINotAvailableError: If AI completely fails.
        """
        analysis_start_time = datetime.now(tz=timezone.utc)

        def report_progress(stage: str, percent: float) -> None:
            if progress_callback:
                progress_callback(stage, percent)

        report_progress("Initializing", 0.0)

        # Validate input
        if len(items) < self.MIN_ITEMS_FOR_ANALYSIS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_ITEMS_FOR_ANALYSIS} items for analysis, "
                f"got {len(items)}"
            )

        # Check for local-only mode
        if self.privacy.local_only_mode:
            logger.warning("Local-only mode enabled, returning fallback report")
            return self._create_fallback_report(items)

        # Preprocess items
        report_progress("Preprocessing", 5.0)
        sorted_items = self._preprocess_items(items)

        # Generate temporal summary
        report_progress("Analyzing Timeline", 10.0)
        temporal_summary = self._generate_temporal_summary(sorted_items)

        # Detect chapters
        report_progress("Detecting Life Chapters", 20.0)
        try:
            chapters = self._detect_chapters(sorted_items, temporal_summary)
        except AIClientError as e:
            logger.error(f"Chapter detection failed: {e}")
            chapters = self._fallback_chapters(sorted_items)

        # Generate narratives for each chapter
        report_progress("Writing Narratives", 40.0)
        chapters = self._generate_chapter_narratives(chapters, sorted_items)

        # Analyze platform behavior
        report_progress("Analyzing Platform Patterns", 60.0)
        platform_insights: list[PlatformBehaviorInsight] = []
        if self.config.include_platform_analysis:
            try:
                platform_insights = self._analyze_platform_behavior(sorted_items)
            except AIClientError as e:
                logger.warning(f"Platform analysis failed: {e}")

        # Detect data gaps
        report_progress("Finding Data Gaps", 70.0)
        data_gaps = self._detect_data_gaps(sorted_items)

        # Generate executive summary
        report_progress("Creating Summary", 80.0)
        try:
            executive_summary = self._generate_executive_summary(
                chapters, platform_insights, temporal_summary
            )
        except AIClientError as e:
            logger.warning(f"Executive summary failed: {e}")
            executive_summary = self._fallback_executive_summary(chapters)

        # Assemble final report
        report_progress("Finalizing", 95.0)

        # Calculate date range
        date_range = None
        dates = [item.timestamp.date() for item in sorted_items if item.timestamp]
        if dates:
            date_range = (min(dates), max(dates))

        # Capture usage metrics for this run
        usage_summary = get_tracker().get_summary(start=analysis_start_time)

        report = LifeStoryReport(
            analysis_config=self.config,
            generated_at=datetime.now(tz=timezone.utc),
            ai_model_used=self.client.model_name,
            total_media_analyzed=len(items),
            date_range=date_range,
            executive_summary=executive_summary,
            chapters=chapters,
            platform_insights=platform_insights,
            detected_patterns=self._detect_patterns(temporal_summary),
            data_gaps=data_gaps,
            data_quality_notes=self._assess_data_quality(items, temporal_summary),
            usage_metrics=usage_summary.__dict__,
            is_fallback_mode=False,
        )

        report_progress("Complete", 100.0)
        logger.info(f"Analysis complete: {len(chapters)} chapters, {len(items)} items")

        return report

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def _preprocess_items(self, items: list[MediaItem]) -> list[MediaItem]:
        """Preprocess items: sort and filter.

        Args:
            items: Raw media items.

        Returns:
            Sorted and filtered items.
        """
        # Filter excluded platforms
        if self.privacy.exclude_platforms:
            items = [
                item for item in items if item.source_platform not in self.privacy.exclude_platforms
            ]

        # Sort by timestamp (items without timestamp go to end)
        def sort_key(item: MediaItem) -> tuple[int, datetime]:
            if item.timestamp:
                return (0, item.timestamp)
            return (1, datetime.min.replace(tzinfo=timezone.utc))

        return sorted(items, key=sort_key)

    def _prepare_items_for_ai(self, items: list[MediaItem]) -> list[dict[str, Any]]:
        """Convert items to AI-safe summaries.

        Applies privacy settings and formats for prompts.

        Args:
            items: Media items to convert.

        Returns:
            List of dicts safe for AI prompts.
        """
        result = []

        for item in items:
            summary = item.to_ai_summary(privacy_mode=self.privacy.anonymize_paths)

            # Apply additional privacy filters
            if self.privacy.truncate_captions and summary.get("caption"):
                caption = summary["caption"]
                if len(caption) > self.privacy.truncate_captions:
                    summary["caption"] = caption[: self.privacy.truncate_captions] + "..."

            if self.privacy.hash_people_names and summary.get("people"):
                summary["people"] = [self._hash_name(name) for name in summary["people"]]

            result.append(summary)

        return result

    def _hash_name(self, name: str) -> str:
        """Hash a name for privacy.

        Args:
            name: Person's name.

        Returns:
            Anonymized identifier.
        """
        hashed = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"Person_{hashed}"

    # =========================================================================
    # Temporal Analysis
    # =========================================================================

    def _generate_temporal_summary(self, items: list[MediaItem]) -> dict[str, Any]:
        """Generate statistical summary of the media timeline.

        Args:
            items: Sorted media items.

        Returns:
            Summary dict with temporal and categorical statistics.
        """
        summary: dict[str, Any] = {
            "total_items": len(items),
            "items_by_year": defaultdict(int),
            "items_by_month": defaultdict(int),
            "items_by_platform": defaultdict(int),
            "items_by_type": defaultdict(int),
            "locations": defaultdict(int),
            "people": defaultdict(int),
            "date_range": None,
            "years_covered": 0,
        }

        for item in items:
            # By platform
            summary["items_by_platform"][item.source_platform.value] += 1

            # By type
            summary["items_by_type"][item.media_type.value] += 1

            # By time
            if item.timestamp:
                year = item.timestamp.year
                month = f"{year}-{item.timestamp.month:02d}"
                summary["items_by_year"][year] += 1
                summary["items_by_month"][month] += 1

            # Locations
            if item.location and item.location.place_name:
                summary["locations"][item.location.place_name] += 1
            elif item.location and item.location.country:
                summary["locations"][item.location.country] += 1

            # People
            for person in item.people:
                summary["people"][person] += 1

        # Convert defaultdicts to regular dicts
        summary["items_by_year"] = dict(summary["items_by_year"])
        summary["items_by_month"] = dict(summary["items_by_month"])
        summary["items_by_platform"] = dict(summary["items_by_platform"])
        summary["items_by_type"] = dict(summary["items_by_type"])
        summary["locations"] = dict(summary["locations"])
        summary["people"] = dict(summary["people"])

        # Date range
        dates = [item.timestamp for item in items if item.timestamp]
        if dates:
            summary["date_range"] = {
                "start": min(dates).isoformat(),
                "end": max(dates).isoformat(),
            }
            summary["years_covered"] = max(dates).year - min(dates).year + 1

        return summary

    # =========================================================================
    # Chapter Detection
    # =========================================================================

    def _detect_chapters(
        self,
        items: list[MediaItem],
        temporal_summary: dict[str, Any],
    ) -> list[LifeChapter]:
        """Detect life chapters using AI.

        Args:
            items: Sorted media items.
            temporal_summary: Temporal statistics.

        Returns:
            List of detected LifeChapter objects.
        """
        # Sample items for the prompt
        sampled = self._sample_items_for_prompt(items, self.MAX_ITEMS_FOR_PROMPT)
        prepared = self._prepare_items_for_ai(sampled)

        # Format temporal summary for prompt
        summary_str = self._format_temporal_summary(temporal_summary)

        # Format sample items
        items_str = json.dumps(prepared, indent=2, default=str)

        # Sampling logic based on depth
        max_vision_images = 5 # Default standard
        if self.config.analysis_depth == AnalysisDepth.QUICK:
            max_vision_images = 1
        elif self.config.analysis_depth == AnalysisDepth.DEEP:
            max_vision_images = 10

        # Prepare fingerprints if not already done
        if not self._media_fingerprint:
            self._media_fingerprint = self._fingerprint_media_set(items)
        if not self._config_fingerprint:
            self._config_fingerprint = self._fingerprint_config()

        # Determine chapter count
        years_covered = temporal_summary.get("years_covered", 1)
        min_chapters = max(2, years_covered // 5)
        max_chapters = min(self.config.max_chapters, max(5, years_covered // 2))

        prompt = CHAPTER_DETECTION_USER_PROMPT.format(
            temporal_summary=summary_str,
            sample_items=items_str,
            min_chapters=min_chapters,
            max_chapters=max_chapters,
        )
        
        # Prepare multimodal contents
        contents = [prompt]
        
        # Add a few representative images from across the collection to anchor the detection
        vision_parts = self._get_vision_parts(items, max_images=max_vision_images)
        contents.extend(vision_parts)

        logger.debug(f"Detecting chapters (prompt tokens: ~{len(prompt) // 4}, images: {len(vision_parts)})")

        try:
            cache_key = self.cache.build_cache_key(self._media_fingerprint, self._config_fingerprint, "chapters")
            result = self.client.generate_json(
                contents=contents if len(contents) > 1 else prompt,
                system_instruction=CHAPTER_DETECTION_SYSTEM_PROMPT,
                operation="chapter_detection",
                cache=self.cache,
                cache_key=cache_key,
            )

            return self._parse_chapters(result, items)

        except AIRequestError as e:
            logger.error(f"Chapter detection AI failed: {e}")
            raise

    def _parse_chapters(
        self,
        ai_result: Any,
        items: list[MediaItem],
    ) -> list[LifeChapter]:
        """Parse AI response into LifeChapter objects.

        Args:
            ai_result: Parsed JSON from AI.
            items: Original items for date validation.

        Returns:
            List of validated LifeChapter objects.
        """
        chapters = []

        # Ensure result is a list
        if isinstance(ai_result, dict) and "chapters" in ai_result:
            ai_result = ai_result["chapters"]
        if not isinstance(ai_result, list):
            ai_result = [ai_result]

        raw_chapters = ai_result # Renamed for clarity

        for chapter_data in raw_chapters:
            try:
                chapter = LifeChapter.model_validate(chapter_data)
                # Map 'reasoning' to 'discovery_evidence' if present
                if "reasoning" in chapter_data:
                    chapter.discovery_evidence = chapter_data["reasoning"]
                chapters.append(chapter)

                # Parse dates
                start_date = self._parse_date(chapter_data.get("start_date", ""))
                end_date = self._parse_date(chapter_data.get("end_date", ""))

                if not start_date or not end_date:
                    logger.warning(f"Invalid dates in chapter {chapter.title}, skipping")
                    continue

                # Parse confidence
                confidence_str = chapter_data.get("confidence", "medium").lower()
                confidence_map = {
                    "high": Confidence.HIGH,
                    "medium": Confidence.MEDIUM,
                    "low": Confidence.LOW,
                }
                confidence = confidence_map.get(confidence_str, Confidence.MEDIUM)

                # Count items in this chapter
                media_count = sum(
                    1
                    for item in items
                    if item.timestamp and start_date <= item.timestamp.date() <= end_date
                )

                chapter = LifeChapter(
                    title=chapter_data.get("title", f"Chapter {idx + 1}"),
                    start_date=start_date,
                    end_date=end_date,
                    themes=chapter_data.get("themes", []),
                    narrative="",  # Will be generated later
                    key_events=[],  # Will be generated later
                    location_summary=chapter_data.get("location_summary"),
                    media_count=media_count,
                    representative_media_ids=[],
                    confidence=confidence,
                )
                chapters.append(chapter)

            except Exception as e:
                logger.warning(f"Failed to parse chapter {idx}: {e}")
                continue

        # Sort by start date
        chapters.sort(key=lambda c: c.start_date)

        return chapters

    def _parse_date(self, date_str: str) -> date | None:
        """Parse a date string.

        Args:
            date_str: Date string in various formats.

        Returns:
            Parsed date or None.
        """
        if not date_str:
            return None

        formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        # Try parsing just year
        try:
            year = int(date_str[:4])
            return date(year, 1, 1)
        except (ValueError, IndexError):
            pass

        return None

    def _fallback_chapters(self, items: list[MediaItem]) -> list[LifeChapter]:
        """Create simple chapters when AI fails.

        Divides timeline into yearly chapters.

        Args:
            items: Sorted media items.

        Returns:
            Basic chapter list.
        """
        chapters = []
        items_with_dates = [i for i in items if i.timestamp]

        if not items_with_dates:
            return chapters

        # Group by year
        by_year: dict[int, list[MediaItem]] = defaultdict(list)
        for item in items_with_dates:
            by_year[item.timestamp.year].append(item)

        for year in sorted(by_year.keys()):
            year_items = by_year[year]
            chapter = LifeChapter(
                title=f"Year {year}",
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                themes=[],
                narrative=f"This chapter contains {len(year_items)} media items from {year}.",
                key_events=[],
                location_summary=None,
                media_count=len(year_items),
                representative_media_ids=[],
                confidence=Confidence.LOW,
            )
            chapters.append(chapter)

        return chapters

    # =========================================================================
    # Narrative Generation
    # =========================================================================

    def _generate_chapter_narratives(
        self,
        chapters: list[LifeChapter],
        items: list[MediaItem],
    ) -> list[LifeChapter]:
        """Generate narratives for each chapter.

        Args:
            chapters: Chapters to generate narratives for.
            items: All media items.

        Returns:
            Chapters with narratives filled in.
        """
        for chapter in chapters:
            try:
                # Get items for this chapter
                chapter_items = [
                    item
                    for item in items
                    if item.timestamp
                    and chapter.start_date <= item.timestamp.date() <= chapter.end_date
                ]

                if not chapter_items:
                    chapter.narrative = f"No detailed data available for this period ({chapter.start_date} to {chapter.end_date})."
                    continue

                # Sample items for prompt
                sampled = self._sample_items_for_prompt(
                    chapter_items, self.ITEMS_PER_CHAPTER_SAMPLE
                )
                prepared = self._prepare_items_for_ai(sampled)

                # Store representative IDs (prefer items with file paths for previews)
                visual_items = [
                    i for i in chapter_items 
                    if i.file_path and i.media_type in (MediaType.PHOTO, MediaType.VIDEO)
                ]
                if not visual_items:
                    visual_items = [i for i in chapter_items if i.file_path]
                
                # Take up to 4 representative items
                if visual_items:
                    # If many items, take some from start, middle, end
                    if len(visual_items) > 4:
                        indices = [0, len(visual_items)//3, 2*len(visual_items)//3, len(visual_items)-1]
                        chapter.representative_media_ids = [visual_items[i].id for i in indices]
                    else:
                        chapter.representative_media_ids = [item.id for item in visual_items]
                else:
                    chapter.representative_media_ids = [item.id for item in sampled[:4]]

                prompt = NARRATIVE_USER_PROMPT.format(
                    title=chapter.title,
                    start_date=chapter.start_date.isoformat(),
                    end_date=chapter.end_date.isoformat(),
                    themes=", ".join(chapter.themes) if chapter.themes else "Not yet determined",
                    location=chapter.location_summary or "Various locations",
                    media_count=chapter.media_count,
                    sample_items=json.dumps(prepared, indent=2, default=str),
                )

                # Prepare multimodal content for narrative
                contents = [prompt]
                
                # Determine max images based on depth
                max_narrative_images = 3 # Default standard
                if self.config.analysis_depth == AnalysisDepth.QUICK:
                    max_narrative_images = 1
                elif self.config.analysis_depth == AnalysisDepth.DEEP:
                    max_narrative_images = 8

                vision_parts = self._get_vision_parts(chapter_items, max_images=max_narrative_images)
                contents.extend(vision_parts)

                cache_key = self.cache.build_cache_key(self._media_fingerprint, self._config_fingerprint, f"narrative_{chapter.id}")
                result = self.client.generate_json(
                    contents=contents if len(contents) > 1 else prompt,
                    system_instruction=NARRATIVE_SYSTEM_PROMPT,
                    operation="chapter_narrative",
                    cache=self.cache,
                    cache_key=cache_key,
                    model_name=self.config.vision_model_name,
                )

                # Update chapter with narrative
                if isinstance(result, dict):
                    chapter.narrative = result.get("narrative", "")
                    chapter.key_events = result.get("key_events", [])
                else:
                    chapter.narrative = str(result)

            except AIClientError as e:
                logger.warning(f"Failed to generate narrative for '{chapter.title}': {e}")
                chapter.narrative = (
                    f"This chapter spans from {chapter.start_date} to {chapter.end_date}, "
                    f"containing {chapter.media_count} media items."
                )

        return chapters

    # =========================================================================
    # Platform Analysis
    # =========================================================================

    def _analyze_platform_behavior(
        self,
        items: list[MediaItem],
    ) -> list[PlatformBehaviorInsight]:
        """Analyze platform usage patterns.

        Args:
            items: All media items.

        Returns:
            List of platform behavior insights.
        """
        # Group by platform
        by_platform: dict[SourcePlatform, list[MediaItem]] = defaultdict(list)
        for item in items:
            by_platform[item.source_platform].append(item)

        # Create statistics
        platform_stats = {}
        platform_samples = {}

        for platform, platform_items in by_platform.items():
            # Statistics
            years = set()
            types: dict[str, int] = defaultdict(int)
            for item in platform_items:
                if item.timestamp:
                    years.add(item.timestamp.year)
                types[item.media_type.value] += 1

            platform_stats[platform.value] = {
                "count": len(platform_items),
                "years": sorted(years) if years else [],
                "types": dict(types),
            }

            # Samples
            sampled = self._sample_items_for_prompt(platform_items, 20)
            platform_samples[platform.value] = self._prepare_items_for_ai(sampled)

        prompt = PLATFORM_ANALYSIS_USER_PROMPT.format(
            platform_stats=json.dumps(platform_stats, indent=2),
            platform_samples=json.dumps(platform_samples, indent=2, default=str),
        )

        result = self.client.generate_json(
            contents=prompt,
            system_instruction=PLATFORM_ANALYSIS_SYSTEM_PROMPT,
        )

        insights = []
        result_list = result if isinstance(result, list) else [result]

        for insight_data in result_list:
            try:
                platform_str = insight_data.get("platform", "unknown")
                platform = SourcePlatform(platform_str.lower().replace(" ", "_"))

                # Parse content types
                content_types = []
                for ct in insight_data.get("common_content_types", []):
                    try:
                        content_types.append(MediaType(ct.lower()))
                    except ValueError:
                        pass

                insight = PlatformBehaviorInsight(
                    platform=platform,
                    usage_pattern=insight_data.get("usage_pattern", ""),
                    peak_years=insight_data.get("peak_years", []),
                    common_content_types=content_types,
                    unique_aspects=insight_data.get("unique_aspects", []),
                )
                insights.append(insight)

            except Exception as e:
                logger.warning(f"Failed to parse platform insight: {e}")

        return insights

    # =========================================================================
    # Executive Summary
    # =========================================================================

    def _generate_executive_summary(
        self,
        chapters: list[LifeChapter],
        platform_insights: list[PlatformBehaviorInsight],
        temporal_summary: dict[str, Any],
    ) -> str:
        """Generate the executive summary.

        Args:
            chapters: All life chapters with narratives.
            platform_insights: Platform usage insights.
            temporal_summary: Timeline statistics.

        Returns:
            Executive summary text.
        """
        # Format chapters for prompt
        chapters_summary = []
        for chapter in chapters:
            chapters_summary.append(
                {
                    "title": chapter.title,
                    "period": f"{chapter.start_date} to {chapter.end_date}",
                    "themes": chapter.themes,
                    "narrative_preview": (
                        chapter.narrative[:300] + "..."
                        if len(chapter.narrative) > 300
                        else chapter.narrative
                    ),
                }
            )

        # Format statistics
        statistics = {
            "total_items": temporal_summary["total_items"],
            "years_covered": temporal_summary["years_covered"],
            "date_range": temporal_summary["date_range"],
            "platforms_used": list(temporal_summary["items_by_platform"].keys()),
            "top_locations": dict(
                sorted(temporal_summary["locations"].items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }

        # Format platform insights
        platform_text = []
        for insight in platform_insights:
            platform_text.append(
                {
                    "platform": insight.platform.value,
                    "pattern": insight.usage_pattern,
                }
            )

        prompt = EXECUTIVE_SUMMARY_USER_PROMPT.format(
            chapters_summary=json.dumps(chapters_summary, indent=2),
            statistics=json.dumps(statistics, indent=2),
            platform_insights=json.dumps(platform_text, indent=2),
        )
        msg = f"Generating executive summary (~{estimate_tokens(prompt)} tokens)"
        logger.debug(msg)

        cache_key = self.cache.build_cache_key(self._media_fingerprint, self._config_fingerprint, "executive_summary")
        response = self.client.generate(
            contents=prompt,
            system_instruction=EXECUTIVE_SUMMARY_SYSTEM_PROMPT,
            operation="executive_summary",
            cache=self.cache,
            cache_key=cache_key,
        )

        return response.text

    def _fallback_executive_summary(self, chapters: list[LifeChapter]) -> str:
        """Create basic summary when AI fails.

        Args:
            chapters: Life chapters.

        Returns:
            Basic summary text.
        """
        if not chapters:
            return "Unable to generate life story summary due to insufficient data."

        summary_parts = [
            f"This life story spans {len(chapters)} distinct chapters, "
            f"from {chapters[0].start_date} to {chapters[-1].end_date}."
        ]

        # Add chapter titles
        chapter_list = ", ".join(f'"{c.title}"' for c in chapters[:5])
        if len(chapters) > 5:
            chapter_list += f", and {len(chapters) - 5} more"
        summary_parts.append(f"The journey includes: {chapter_list}.")

        return " ".join(summary_parts)

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
                    gap = DataGap(
                        start_date=prev_date,
                        end_date=current_date,
                        gap_days=gap_duration.days,
                        possible_reasons=[],
                    )
                    gaps.append(gap)

            prev_date = current_date

        # Optionally get AI to suggest reasons
        if gaps and len(gaps) <= 10:
            try:
                gaps = self._enrich_gaps_with_ai(gaps)
            except AIClientError:
                pass  # Keep gaps without reasons

        return gaps

    def _enrich_gaps_with_ai(self, gaps: list[DataGap]) -> list[DataGap]:
        """Ask AI to suggest reasons for data gaps.

        Args:
            gaps: Detected gaps.

        Returns:
            Gaps with possible_reasons populated.
        """
        gaps_text = []
        for i, gap in enumerate(gaps):
            gaps_text.append(
                f"{i + 1}. Gap from {gap.start_date} to {gap.end_date} ({gap.gap_days} days)"
            )

        prompt = GAP_ANALYSIS_PROMPT.format(gaps="\n".join(gaps_text))

        result = self.client.generate_json(contents=prompt)

        if isinstance(result, list):
            for item in result:
                idx = item.get("gap_number", 0) - 1
                if 0 <= idx < len(gaps):
                    gaps[idx].possible_reasons = item.get("possible_reasons", [])

        return gaps

    # =========================================================================
    # Sampling and Utilities
    # =========================================================================

    def _fingerprint_media_set(self, items: list[MediaItem]) -> str:
        """Create a deterministic fingerprint of the media collection."""
        if not items:
            return hashlib.sha256(b"empty-media-set").hexdigest()
        
        # Build list of tuples for each item for stable hashing
        item_data = []
        for item in items:
            ts = item.timestamp.isoformat() if item.timestamp else ""
            plat = item.source_platform.value if item.source_platform else ""
            # Use filename as a proxy for content if path is available
            cid = item.file_path.name if item.file_path else str(item.id)
            item_data.append(f"{ts}:{plat}:{cid}")
            
        item_data.sort()
        combined = "|".join(item_data)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _fingerprint_config(self) -> str:
        """Create a deterministic fingerprint of the analysis configuration."""
        config_dict = {
            "min_chapters": self.config.min_chapters,
            "max_chapters": self.config.max_chapters,
            "narrative_style": self.config.narrative_style,
            "include_platform_analysis": self.config.include_platform_analysis,
            "include_gap_analysis": self.config.include_gap_analysis,
            "privacy_mode": self.privacy.local_only_mode,
        }
        json_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _get_vision_parts(self, items: list[MediaItem], max_images: int = 3) -> list[types.Part]:
        """Get visual parts (images) for multimodal analysis.
        
        Args:
            items: Media items to sample from.
            max_images: Maximum number of images to include.
            
        Returns:
            List of types.Part objects containing image data.
        """
        # Skip if privacy mode is too strict
        if self.privacy.local_only_mode:
            return []

        # Find items with valid local image files
        visual_items = [
            i for i in items 
            if i.file_path and i.file_path.exists() 
            and i.media_type == MediaType.PHOTO
            and i.file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        ]
        
        if not visual_items:
            return []

        # Sample spanning the range with intelligence
        if len(visual_items) > max_images:
            # Sort items by "richness" (people count + caption)
            # but we still want them to be somewhat chronologically distributed.
            # We'll split the items into 'max_images' buckets and pick the "best" from each bucket.
            bucket_size = len(visual_items) // max_images
            sampled = []
            for i in range(max_images):
                start_idx = i * bucket_size
                end_idx = start_idx + bucket_size if i < max_images - 1 else len(visual_items)
                bucket = visual_items[start_idx:end_idx]
                
                # Pick the "best" item in the bucket: prioritize people > captions > random
                # (Simple score: people_count*2 + (1 if caption else 0))
                best_item = max(
                    bucket, 
                    key=lambda x: (len(x.people) * 2 + (1 if x.caption else 0) + random.random())
                )
                sampled.append(best_item)
        else:
            sampled = visual_items

        parts = []
        for item in sampled:
            try:
                # Open and process image efficiently
                with Image.open(item.file_path) as img:
                    # Convert to RGB if necessary (e.g. for RGBA or CMYK)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Resize if too large - maintaining aspect ratio
                    max_dim = 1600
                    if max(img.width, img.height) > max_dim:
                        if img.width > img.height:
                            new_size = (max_dim, int(img.height * (max_dim / img.width)))
                        else:
                            new_size = (int(img.width * (max_dim / img.height)), max_dim)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        logger.debug(f"Resized image for vision: {item.file_path.name} to {new_size}")

                    # Save to bytes as JPEG
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80, optimize=True)
                    img_bytes = buf.getvalue()

                # Basic size check (Gemini parts should be < 20MB)
                if len(img_bytes) > 15 * 1024 * 1024:
                    logger.debug(f"Skipping extremely large image: {item.file_path.name}")
                    continue

                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                logger.debug(f"Added vision part (optimized): {item.file_path.name} ({len(img_bytes) / 1024:.1f} KB)")
            except Exception as e:
                logger.warning(f"Failed to process image for vision: {e}")

        return parts

    def _sample_items_for_prompt(
        self,
        items: list[MediaItem],
        max_items: int = 200,
    ) -> list[MediaItem]:
        """Smart sample items for AI prompts.

        Ensures coverage across:
        - Entire date range
        - All platforms
        - Prefers items with rich metadata

        Args:
            items: Items to sample from.
            max_items: Maximum items to return.

        Returns:
            Sampled list of items.
        """
        if len(items) <= max_items:
            return items

        result = []
        remaining = set(range(len(items)))

        # Helper to add item by index
        def add_item(idx: int) -> None:
            if idx in remaining:
                result.append(items[idx])
                remaining.remove(idx)

        # 1. Ensure temporal coverage - take items across timeline
        temporal_sample = max(max_items // 4, 10)
        if len(items) >= temporal_sample:
            step = len(items) // temporal_sample
            for i in range(0, len(items), step):
                add_item(i)
                if len(result) >= temporal_sample:
                    break

        # 2. Prioritize items with rich metadata
        rich_items = [
            i for i in remaining if items[i].location or items[i].people or items[i].caption
        ]
        random.shuffle(rich_items)
        for idx in rich_items[: max_items // 4]:
            add_item(idx)

        # 3. Platform diversity
        by_platform: dict[SourcePlatform, list[int]] = defaultdict(list)
        for i in remaining:
            by_platform[items[i].source_platform].append(i)

        for platform, indices in by_platform.items():
            random.shuffle(indices)
            for idx in indices[: max_items // (len(by_platform) * 2)]:
                add_item(idx)

        # 4. Fill remaining with random samples
        remaining_list = list(remaining)
        random.shuffle(remaining_list)
        for idx in remaining_list:
            if len(result) >= max_items:
                break
            add_item(idx)

        # Sort by timestamp
        result.sort(
            key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=timezone.utc)
        )

        return result

    def _format_temporal_summary(self, summary: dict[str, Any]) -> str:
        """Format temporal summary for prompts.

        Args:
            summary: Raw temporal summary.

        Returns:
            Formatted string.
        """
        lines = [
            f"Total items: {summary['total_items']}",
            f"Date range: {summary.get('date_range', 'Unknown')}",
            f"Years covered: {summary['years_covered']}",
            "",
            "Items by year:",
        ]

        for year, count in sorted(summary["items_by_year"].items()):
            lines.append(f"  {year}: {count}")

        lines.append("")
        lines.append("Items by platform:")
        for platform, count in summary["items_by_platform"].items():
            lines.append(f"  {platform}: {count}")

        if summary["locations"]:
            lines.append("")
            lines.append("Top locations:")
            sorted_locs = sorted(summary["locations"].items(), key=lambda x: x[1], reverse=True)[:5]
            for loc, count in sorted_locs:
                lines.append(f"  {loc}: {count}")

        return "\n".join(lines)

    def _detect_patterns(self, temporal_summary: dict[str, Any]) -> list[str]:
        """Detect high-level patterns from statistics.

        Args:
            temporal_summary: Timeline statistics.

        Returns:
            List of detected patterns.
        """
        patterns = []

        # Check for year-over-year trends
        years = sorted(temporal_summary["items_by_year"].items())
        if len(years) >= 3:
            counts = [c for _, c in years]
            if counts[-1] > counts[0] * 1.5:
                patterns.append("Increasing documentation activity over time")
            elif counts[-1] < counts[0] * 0.5:
                patterns.append("Decreasing documentation activity over time")

        # Platform dominance
        platforms = temporal_summary["items_by_platform"]
        if platforms:
            dominant = max(platforms, key=platforms.get)
            if platforms[dominant] > sum(platforms.values()) * 0.6:
                patterns.append(f"Primary platform: {dominant}")

        # Location diversity
        locations = temporal_summary["locations"]
        if len(locations) > 10:
            patterns.append("Geographically diverse content")
        elif len(locations) == 1:
            patterns.append("Single primary location")

        return patterns

    def _assess_data_quality(
        self,
        items: list[MediaItem],
        temporal_summary: dict[str, Any],
    ) -> list[str]:
        """Assess data quality and generate notes.

        Args:
            items: All media items.
            temporal_summary: Timeline statistics.

        Returns:
            List of quality notes.
        """
        notes = []

        # Check timestamp coverage
        items_with_ts = sum(1 for i in items if i.timestamp)
        ts_percent = (items_with_ts / len(items)) * 100 if items else 0
        if ts_percent < 50:
            notes.append(f"Warning: Only {ts_percent:.0f}% of items have timestamps")

        # Check location coverage
        items_with_loc = sum(1 for i in items if i.location)
        loc_percent = (items_with_loc / len(items)) * 100 if items else 0
        if loc_percent < 20:
            notes.append(f"Limited location data: {loc_percent:.0f}% of items")
        elif loc_percent > 50:
            notes.append(f"Good location coverage: {loc_percent:.0f}% of items")

        # Check confidence levels
        low_conf = sum(1 for i in items if i.timestamp_confidence == Confidence.LOW)
        if low_conf > len(items) * 0.3:
            notes.append("Many items have low-confidence timestamps")

        return notes

    def _create_fallback_report(self, items: list[MediaItem]) -> LifeStoryReport:
        """Create a basic report without AI.

        Used when in local-only mode or AI completely fails.

        Args:
            items: All media items.

        Returns:
            Basic LifeStoryReport.
        """
        sorted_items = self._preprocess_items(items)
        temporal_summary = self._generate_temporal_summary(sorted_items)
        chapters = self._fallback_chapters(sorted_items)

        date_range = None
        dates = [item.timestamp.date() for item in sorted_items if item.timestamp]
        if dates:
            date_range = (min(dates), max(dates))

        return LifeStoryReport(
            generated_at=datetime.now(tz=timezone.utc),
            ai_model_used="none (fallback mode)",
            total_media_analyzed=len(items),
            date_range=date_range,
            executive_summary=self._fallback_executive_summary(chapters),
            chapters=chapters,
            platform_insights=[],
            detected_patterns=self._detect_patterns(temporal_summary),
            data_gaps=self._detect_data_gaps(sorted_items),
            data_quality_notes=self._assess_data_quality(items, temporal_summary),
            is_fallback_mode=True,
        )
