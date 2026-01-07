"""Life Story Analyzer — The Core AI Engine.

This module contains the LifeStoryAnalyzer class that orchestrates multi-step
AI analysis to transform raw Memory data into a rich, narrative Life Story Report.

THIS IS THE HEART OF THE PRODUCT. The analyzer:
1. Takes normalized Memory objects from parsers
2. Prepares them for AI analysis (sampling, summarizing, privacy filtering)
3. Runs multiple AI prompts in sequence (chapter detection → narratives → summary)
4. Assembles the final LifeStoryReport
5. Handles failures gracefully (partial results are better than nothing)

Example:
    >>> from src.ai.life_analyzer import LifeStoryAnalyzer, AnalysisConfig
    >>> from src.core.memory import Memory
    >>>
    >>> # Load your memories
    >>> memories: list[Memory] = load_memories_from_somewhere()
    >>>
    >>> # Configure and analyze
    >>> config = AnalysisConfig(min_chapters=5, max_chapters=10)
    >>> analyzer = LifeStoryAnalyzer(config=config)
    >>>
    >>> # Run analysis with progress callback
    >>> def progress(p):
    ...     print(f"[{p.percentage():.0f}%] {p.message}")
    >>>
    >>> report = analyzer.analyze(memories, progress_callback=progress)
    >>>
    >>> # Use the report
    >>> print(f"Found {len(report.chapters)} chapters")
    >>> print(report.executive_summary)

Philosophy:
- Resilient: Partial success > Total failure
- Quality > Quantity: Better 5 good chapters than 15 shallow ones
- Privacy-first: All data goes through privacy filtering before AI
- Transparent: Progress callbacks at every stage
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, Field

from src.ai.client import (
    AIClient,
    AIClientError,
    AIUnavailableError,
    TokenLimitExceededError,
    get_client,
)
from src.ai.prompts import (
    CHAPTER_DETECTION_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    GAP_ANALYSIS_PROMPT,
    NARRATIVE_GENERATION_PROMPT,
    PATTERN_DETECTION_PROMPT,
    PLATFORM_ANALYSIS_PROMPT,
    PromptContext,
    build_prompt_context,
    prepare_chapters_for_prompt,
    prepare_memories_for_prompt,
    prepare_platform_breakdown,
    prepare_timeline_summary,
)
from src.core.memory import ConfidenceLevel, MediaType, Memory, SourcePlatform
from src.core.timeline import DateRange, Timeline, TimelineGap, TimelineStatistics
from src.ai.cache import (
    AICache,
    CacheMeta,
    fingerprint_media_set,
    fingerprint_analysis_config,
    get_machine_id,
)

if TYPE_CHECKING:
    from src.core.privacy import PrivacyGate

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AnalysisConfig:
    """Configuration for the life story analysis process.
    
    Controls the behavior of chapter detection, narrative generation,
    and optional analysis features.
    
    Attributes:
        min_chapters: Minimum number of chapters to detect.
        max_chapters: Maximum number of chapters to detect.
        min_chapter_duration_days: Minimum days for a valid chapter.
        max_memories_for_chapter_detection: Sample size for chapter detection.
        max_memories_per_chapter_narrative: Sample size per chapter narrative.
        include_platform_analysis: Whether to analyze platform usage.
        include_gap_analysis: Whether to analyze timeline gaps.
        detect_patterns: Whether to detect behavioral patterns.
        narrative_style: Tone for narratives ("warm", "neutral", "analytical").
        privacy_level: Privacy filtering level ("strict", "standard", "detailed").
        fail_on_partial: If True, fail if any step fails; if False, continue.
    
    Example:
        >>> config = AnalysisConfig(
        ...     min_chapters=3,
        ...     max_chapters=10,
        ...     narrative_style="warm",
        ... )
    """
    
    min_chapters: int = 3
    max_chapters: int = 15
    min_chapter_duration_days: int = 30
    max_memories_for_chapter_detection: int = 500
    max_memories_per_chapter_narrative: int = 200
    include_platform_analysis: bool = True
    include_gap_analysis: bool = True
    detect_patterns: bool = True
    narrative_style: Literal["warm", "neutral", "analytical"] = "warm"
    privacy_level: str = "standard"
    fail_on_partial: bool = False


# =============================================================================
# Progress Tracking
# =============================================================================


@dataclass
class AnalysisProgress:
    """Progress information for analysis callbacks.
    
    Provides real-time progress updates during analysis.
    
    Attributes:
        stage: Current stage name (e.g., "chapter_detection").
        current_step: Current step number within stage.
        total_steps: Total steps in current stage.
        current_chapter: Name of current chapter being processed.
        message: Human-readable progress message.
        elapsed_seconds: Time elapsed since analysis started.
    
    Example:
        >>> def callback(progress: AnalysisProgress):
        ...     print(progress.to_status_line())
    """
    
    stage: str
    current_step: int
    total_steps: int
    current_chapter: str | None = None
    message: str = ""
    elapsed_seconds: float = 0.0
    
    def percentage(self) -> float:
        """Calculate completion percentage.
        
        Returns:
            Percentage complete (0.0 to 100.0).
        """
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0
    
    def to_status_line(self) -> str:
        """Format as a status line for display.
        
        Returns:
            Formatted status string.
        """
        pct = self.percentage()
        elapsed = f"{self.elapsed_seconds:.1f}s"
        
        if self.current_chapter:
            return f"[{pct:3.0f}%] {self.stage}: {self.current_chapter} - {self.message} ({elapsed})"
        return f"[{pct:3.0f}%] {self.stage}: {self.message} ({elapsed})"


# =============================================================================
# Chapter Model
# =============================================================================


class ChapterTheme(BaseModel):
    """A theme identified within a chapter.
    
    Attributes:
        name: Theme keyword (e.g., "travel", "career").
        confidence: Confidence level (0.0 to 1.0).
    """
    
    name: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ChapterInsight(BaseModel):
    """An insight about a life chapter.
    
    Attributes:
        text: The insight description.
        evidence: What data supports this insight.
    """
    
    text: str
    evidence: str = ""


class LifeChapter(BaseModel):
    """A detected life chapter with narrative.
    
    Represents a distinct period in someone's life story with
    AI-generated narrative and metadata.
    
    Attributes:
        id: Unique chapter identifier.
        title: Chapter title (e.g., "The College Years").
        start_date: Chapter start date.
        end_date: Chapter end date.
        themes: Keywords describing this period.
        narrative: 2-4 paragraph narrative.
        opening_line: Hook for timeline view.
        key_events: Notable events/observations.
        insights: Deeper insights about this period.
        memory_ids: IDs of memories in this chapter.
        memory_count: Count of memories.
        reasoning: Why this was identified as a chapter.
        confidence: AI confidence (0.0 to 1.0).
    
    Example:
        >>> chapter = LifeChapter(
        ...     id="ch_001",
        ...     title="The College Years",
        ...     start_date=date(2015, 9, 1),
        ...     end_date=date(2019, 5, 15),
        ...     themes=["education", "friendships", "exploration"],
        ... )
    """
    
    id: str = Field(default_factory=lambda: f"ch_{uuid.uuid4().hex[:8]}")
    title: str
    start_date: date
    end_date: date
    themes: list[str] = Field(default_factory=list)
    narrative: str = ""
    opening_line: str = ""
    key_events: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    memory_count: int = 0
    reasoning: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @property
    def duration_days(self) -> int:
        """Duration of chapter in days."""
        return (self.end_date - self.start_date).days
    
    @property
    def date_range(self) -> DateRange:
        """Get as DateRange object."""
        return DateRange(start=self.start_date, end=self.end_date)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    def overlaps_with(self, other: LifeChapter) -> bool:
        """Check if this chapter overlaps with another."""
        return self.date_range.overlaps(other.date_range)

    def merge_with(self, other: LifeChapter) -> LifeChapter:
        """Merge another chapter into this one."""
        new_range = self.date_range.merge(other.date_range)
        
        # Combine themes and deduplicate
        combined_themes = list(set(self.themes + other.themes))
        
        # Combined memory IDs
        combined_memory_ids = list(set(self.memory_ids + other.memory_ids))
        
        return LifeChapter(
            title=f"{self.title} & {other.title}",
            start_date=new_range.start,
            end_date=new_range.end,
            themes=combined_themes,
            memory_ids=combined_memory_ids,
            memory_count=len(combined_memory_ids),
            narrative=f"{self.narrative}\n\n{other.narrative}",
            confidence=min(self.confidence, other.confidence)
        )

    def to_timeline_entry(self) -> dict[str, Any]:
        """Convert to minimal structure for timeline view."""
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start_date.isoformat(),
            "end": self.end_date.isoformat(),
            "memory_count": self.memory_count,
            "themes": self.themes
        }


# =============================================================================
# Analysis Results
# =============================================================================


class PlatformBehaviorInsight(BaseModel):
    """Analysis of platform-specific usage patterns.
    
    Attributes:
        platform: The source platform.
        usage_pattern: How they use this platform.
        peak_period: When they used it most.
        unique_characteristics: What's unique about their usage.
        memory_count: Memories from this platform.
        percentage_of_total: Percentage of all memories.
    """
    
    platform: SourcePlatform
    usage_pattern: str = ""
    peak_period: str | None = None
    unique_characteristics: list[str] = Field(default_factory=list)
    memory_count: int = 0
    percentage_of_total: float = 0.0


class DataGap(BaseModel):
    """Information about a gap in the timeline.
    
    Attributes:
        start_date: When the gap starts.
        end_date: When the gap ends.
        duration_days: Gap duration in days.
        possible_explanations: AI-suggested reasons.
        severity: Gap severity level.
        impacts_narrative: Whether gap affects story.
    """
    
    start_date: date
    end_date: date
    duration_days: int = 0
    possible_explanations: list[str] = Field(default_factory=list)
    severity: Literal["minor", "moderate", "significant", "major"] = "minor"
    impacts_narrative: bool = False


class LifeStoryReport(BaseModel):
    """The final output of life story analysis.
    
    Contains all analyzed data: chapters, narratives, insights,
    patterns, and metadata about the analysis process.
    
    Attributes:
        id: Unique report identifier.
        generated_at: When report was generated.
        ai_model: AI model used for analysis.
        analysis_config: Configuration used.
        date_range: Timeline date range.
        total_memories_analyzed: Memory count.
        executive_summary: Overall life story summary.
        chapters: Detected life chapters with narratives.
        platform_insights: Platform usage analysis.
        detected_patterns: Cross-cutting patterns.
        data_gaps: Identified timeline gaps.
        data_quality_notes: Notes about data quality.
        is_partial: True if some steps failed.
        partial_failures: List of failed steps.
        is_fallback: True if using fallback mode.
        tokens_used: Total tokens consumed.
        generation_time_seconds: Total analysis time.
    
    Example:
        >>> report = analyze_memories(memories)
        >>> print(f"Story spans {report.date_range}")
        >>> for chapter in report.chapters:
        ...     print(f"- {chapter.title}")
    """
    
    id: str = Field(default_factory=lambda: f"report_{uuid.uuid4().hex[:12]}")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ai_model: str = "gemini-1.5-pro"
    analysis_config: dict[str, Any] = Field(default_factory=dict)
    date_range: DateRange | None = None
    total_memories_analyzed: int = 0
    executive_summary: str = ""
    chapters: list[LifeChapter] = Field(default_factory=list)
    platform_insights: list[PlatformBehaviorInsight] = Field(default_factory=list)
    detected_patterns: list[str] = Field(default_factory=list)
    data_gaps: list[DataGap] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    is_partial: bool = False
    partial_failures: list[str] = Field(default_factory=list)
    is_fallback: bool = False
    tokens_used: int = 0
    generation_time_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=indent)
    
    def get_chapter(self, chapter_id: str) -> LifeChapter | None:
        """Find a chapter by ID.
        
        Args:
            chapter_id: The chapter ID to find.
        
        Returns:
            LifeChapter if found, None otherwise.
        """
        for chapter in self.chapters:
            if chapter.id == chapter_id:
                return chapter
        return None


# =============================================================================
# Main Analyzer Class
# =============================================================================


class LifeStoryAnalyzer:
    """The main AI analysis engine for life story reconstruction.
    
    Orchestrates the full analysis pipeline from raw memories to
    a complete life story report with chapters, narratives, and insights.
    
    The analyzer is designed for resilience:
    - Partial failures don't crash the analysis
    - Progress is reported at every stage
    - Privacy is respected throughout
    
    Attributes:
        _client: AI client for generation.
        _config: Analysis configuration.
        _privacy_gate: Optional privacy filtering.
        _logger: Logger instance.
    
    Example:
        >>> analyzer = LifeStoryAnalyzer()
        >>> report = analyzer.analyze(memories)
        >>> print(report.executive_summary)
    """
    
    # Stage definitions for progress tracking
    STAGES = {
        "preparation": {"weight": 1, "description": "Preparing data"},
        "chapter_detection": {"weight": 2, "description": "Detecting life chapters"},
        "narrative_generation": {"weight": 5, "description": "Writing narratives"},
        "platform_analysis": {"weight": 1, "description": "Analyzing platforms"},
        "gap_analysis": {"weight": 1, "description": "Analyzing gaps"},
        "executive_summary": {"weight": 1, "description": "Writing summary"},
        "pattern_detection": {"weight": 1, "description": "Finding patterns"},
        "finalization": {"weight": 1, "description": "Finalizing report"},
    }
    
    def __init__(
        self,
        client: AIClient | None = None,
        config: AnalysisConfig | None = None,
        privacy_gate: "PrivacyGate | None" = None,
        cache: AICache | None = None,
    ) -> None:
        """Initialize the analyzer.
        
        Args:
            client: AI client (uses get_client() if not provided).
            config: Analysis configuration (uses defaults if not provided).
            privacy_gate: Optional privacy filter for memories.
            cache: Optional cache instance. If None and caching is desired,
                   create one with AICache().
        """
        self._client = client
        self._config = config or AnalysisConfig()
        self._privacy_gate = privacy_gate
        self._cache = cache
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track tokens and timing
        self._tokens_used = 0
        self._start_time = 0.0
    
    def _get_client(self) -> AIClient:
        """Get or create the AI client.
        
        Returns:
            AIClient instance.
        
        Raises:
            AIUnavailableError: If AI is not available.
        """
        if self._client is None:
            self._client = get_client()
        return self._client
    
    def analyze(
        self,
        memories: list[Memory],
        progress_callback: Callable[[AnalysisProgress], None] | None = None,
    ) -> LifeStoryReport:
        """Execute the full analysis pipeline.
        
        MAIN ENTRY POINT for life story analysis.
        
        Pipeline stages:
        1. Validation - Check minimum data requirements
        2. Timeline Construction - Build and analyze timeline
        3. Data Preparation - Sample and filter for AI
        4. Chapter Detection - AI identifies life chapters
        5. Narrative Generation - AI writes per-chapter narratives
        6. Platform Analysis - AI analyzes cross-platform behavior
        7. Gap Analysis - AI explains timeline gaps
        8. Executive Summary - AI synthesizes overall story
        9. Pattern Detection - AI finds cross-cutting patterns
        10. Finalization - Assemble final report
        
        Args:
            memories: List of Memory objects to analyze.
            progress_callback: Optional callback for progress updates.
        
        Returns:
            LifeStoryReport containing the complete analysis.
        
        Raises:
            AIUnavailableError: If AI client is not available.
            ValueError: If insufficient memories for analysis.
        
        Example:
            >>> def on_progress(p):
            ...     print(f"{p.percentage():.0f}% - {p.message}")
            >>>
            >>> report = analyzer.analyze(memories, progress_callback=on_progress)
        """
        self._start_time = time.time()
        self._tokens_used = 0
        
        # Initialize report
        report = LifeStoryReport(
            analysis_config=vars(self._config),
            total_memories_analyzed=len(memories),
        )
        
        # -----------------------------------------------------------------
        # Cache: Compute fingerprints and check for cached result
        # -----------------------------------------------------------------
        media_fp: str | None = None
        config_fp: str | None = None
        cache_key: str | None = None
        
        if self._cache is not None:
            try:
                # Import privacy settings for fingerprinting if gate exists
                privacy_settings = None
                if self._privacy_gate is not None:
                    # Access privacy config through the gate if available
                    privacy_settings = getattr(self._privacy_gate, "_config", None)
                
                # Compute fingerprints
                media_fp = fingerprint_media_set(memories)
                config_fp = fingerprint_analysis_config(self._config, privacy_settings)
                cache_key = self._cache.build_cache_key(media_fp, config_fp, "full_report")
                
                # Try to load cached result
                cached_payload = self._cache.load(cache_key, media_fp, config_fp)
                
                if cached_payload is not None:
                    self._logger.info("Cache hit: returning cached analysis")
                    
                    # Reconstruct report from cached data
                    try:
                        cached_report = self._reconstruct_report_from_cache(cached_payload)
                        cached_report.generation_time_seconds = time.time() - self._start_time
                        
                        # Emit final progress for cache hit
                        if progress_callback:
                            progress = AnalysisProgress(
                                stage="cache_hit",
                                current_step=1,
                                total_steps=1,
                                message="Loaded from cache",
                                elapsed_seconds=cached_report.generation_time_seconds,
                            )
                            try:
                                progress_callback(progress)
                            except Exception:
                                pass
                        
                        return cached_report
                    except Exception as e:
                        self._logger.warning(f"Cache reconstruction failed: {e}. Proceeding with full analysis.")
                else:
                    self._logger.debug("Cache miss: proceeding with full analysis")
                    
            except Exception as e:
                # Cache errors should never break analysis
                self._logger.warning(f"Cache check failed (non-fatal): {e}")
                media_fp = None
                config_fp = None
                cache_key = None
        
        def emit_progress(
            stage: str,
            step: int,
            total: int,
            message: str,
            chapter: str | None = None,
        ) -> None:
            """Emit a progress update."""
            if progress_callback:
                progress = AnalysisProgress(
                    stage=stage,
                    current_step=step,
                    total_steps=total,
                    current_chapter=chapter,
                    message=message,
                    elapsed_seconds=time.time() - self._start_time,
                )
                try:
                    progress_callback(progress)
                except Exception as e:
                    self._logger.warning(f"Progress callback failed: {e}")
        
        # Calculate total steps for progress
        total_steps = 10  # Base steps
        if self._config.include_platform_analysis:
            total_steps += 1
        if self._config.include_gap_analysis:
            total_steps += 1
        if self._config.detect_patterns:
            total_steps += 1
        
        current_step = 0
        
        try:
            # -------------------------------------------------------------
            # Stage 1: Validation
            # -------------------------------------------------------------
            emit_progress("preparation", current_step, total_steps, "Validating data")
            
            if len(memories) < 10:
                raise ValueError(
                    f"Insufficient memories for analysis (got {len(memories)}, need 10+)"
                )
            
            # Check AI availability
            client = self._get_client()
            if not client.is_available():
                raise AIUnavailableError("disabled", "AI client is not available")
            
            report.ai_model = client.get_model_info().get("name", "unknown")
            current_step += 1
            
            # -------------------------------------------------------------
            # Stage 2: Timeline Construction
            # -------------------------------------------------------------
            emit_progress("preparation", current_step, total_steps, "Building timeline")
            
            timeline = Timeline(memories=memories)
            stats = timeline.compute_statistics()
            
            # Determine date range
            if stats.date_range:
                report.date_range = stats.date_range
            
            current_step += 1
            
            # -------------------------------------------------------------
            # Stage 3: Data Preparation
            # -------------------------------------------------------------
            emit_progress("preparation", current_step, total_steps, "Preparing context")
            
            context = self._prepare_timeline_for_ai(timeline)
            current_step += 1
            
            # -------------------------------------------------------------
            # Stage 4: Chapter Detection
            # -------------------------------------------------------------
            emit_progress("chapter_detection", current_step, total_steps, "Detecting chapters")
            
            try:
                chapters = self._detect_chapters(timeline, context)
                report.chapters = chapters
                self._logger.info(f"Detected {len(chapters)} chapters")
            except Exception as e:
                self._logger.error(f"Chapter detection failed: {e}")
                if self._config.fail_on_partial:
                    raise
                report.is_partial = True
                report.partial_failures.append(f"chapter_detection: {e}")
                # Create fallback chapter covering entire timeline
                chapters = self._create_fallback_chapters(timeline)
                report.chapters = chapters
            
            current_step += 1
            
            # Add chapter count to steps for narrative generation
            narrative_steps = len(chapters)
            total_steps += narrative_steps
            
            # -------------------------------------------------------------
            # Stage 5: Narrative Generation
            # -------------------------------------------------------------
            for i, chapter in enumerate(chapters):
                emit_progress(
                    "narrative_generation",
                    current_step + i,
                    total_steps,
                    f"Writing narrative ({i + 1}/{len(chapters)})",
                    chapter=chapter.title,
                )
                
                try:
                    # Get memories for this chapter
                    chapter_memories = timeline.get_memories_in_range(
                        chapter.start_date,
                        chapter.end_date,
                    )
                    
                    # Generate narrative
                    chapter = self._generate_chapter_narrative(
                        chapter=chapter,
                        memories=chapter_memories,
                        chapter_index=i,
                        all_chapters=chapters,
                    )
                    report.chapters[i] = chapter
                    
                except Exception as e:
                    self._logger.error(f"Narrative generation failed for {chapter.title}: {e}")
                    if self._config.fail_on_partial:
                        raise
                    report.is_partial = True
                    report.partial_failures.append(f"narrative_{chapter.id}: {e}")
            
            current_step += narrative_steps
            
            # -------------------------------------------------------------
            # Stage 6: Platform Analysis (optional)
            # -------------------------------------------------------------
            if self._config.include_platform_analysis:
                emit_progress("platform_analysis", current_step, total_steps, "Analyzing platforms")
                
                try:
                    platform_insights = self._analyze_platforms(timeline, memories)
                    report.platform_insights = platform_insights
                except Exception as e:
                    self._logger.error(f"Platform analysis failed: {e}")
                    if self._config.fail_on_partial:
                        raise
                    report.is_partial = True
                    report.partial_failures.append(f"platform_analysis: {e}")
                
                current_step += 1
            
            # -------------------------------------------------------------
            # Stage 7: Gap Analysis (optional)
            # -------------------------------------------------------------
            if self._config.include_gap_analysis:
                emit_progress("gap_analysis", current_step, total_steps, "Analyzing gaps")
                
                try:
                    gaps = timeline.detect_gaps(threshold_days=30)
                    if gaps:
                        data_gaps = self._analyze_gaps(timeline, gaps)
                        report.data_gaps = data_gaps
                except Exception as e:
                    self._logger.error(f"Gap analysis failed: {e}")
                    if self._config.fail_on_partial:
                        raise
                    report.is_partial = True
                    report.partial_failures.append(f"gap_analysis: {e}")
                
                current_step += 1
            
            # -------------------------------------------------------------
            # Stage 8: Executive Summary
            # -------------------------------------------------------------
            emit_progress("executive_summary", current_step, total_steps, "Writing summary")
            
            try:
                executive_summary = self._generate_executive_summary(
                    chapters=report.chapters,
                    timeline=timeline,
                    stats=stats,
                )
                report.executive_summary = executive_summary
            except Exception as e:
                self._logger.error(f"Executive summary failed: {e}")
                if self._config.fail_on_partial:
                    raise
                report.is_partial = True
                report.partial_failures.append(f"executive_summary: {e}")
                # Fallback summary
                report.executive_summary = self._generate_fallback_summary(report.chapters)
            
            current_step += 1
            
            # -------------------------------------------------------------
            # Stage 9: Pattern Detection (optional)
            # -------------------------------------------------------------
            if self._config.detect_patterns:
                emit_progress("pattern_detection", current_step, total_steps, "Finding patterns")
                
                try:
                    patterns = self._detect_patterns(timeline, report.chapters)
                    report.detected_patterns = patterns
                except Exception as e:
                    self._logger.error(f"Pattern detection failed: {e}")
                    if self._config.fail_on_partial:
                        raise
                    report.is_partial = True
                    report.partial_failures.append(f"pattern_detection: {e}")
                
                current_step += 1
            
            # -------------------------------------------------------------
            # Stage 10: Finalization
            # -------------------------------------------------------------
            emit_progress("finalization", current_step, total_steps, "Finalizing report")
            
            # Add data quality notes
            report.data_quality_notes = self._assess_data_quality(stats)
            
            # Update metrics
            report.tokens_used = self._tokens_used
            report.generation_time_seconds = time.time() - self._start_time
            
            current_step += 1
            emit_progress("finalization", current_step, total_steps, "Complete!")
            
            self._logger.info(
                f"Analysis complete: {len(report.chapters)} chapters, "
                f"{report.tokens_used} tokens, {report.generation_time_seconds:.1f}s"
            )
            
            # -----------------------------------------------------------------
            # Cache: Store result for future runs
            # -----------------------------------------------------------------
            if self._cache is not None and cache_key is not None and media_fp is not None and config_fp is not None:
                try:
                    meta = CacheMeta(
                        created_at=time.time(),
                        machine_id=get_machine_id(),
                        media_set_fingerprint=media_fp,
                        analysis_config_fingerprint=config_fp,
                        version=self._cache._version,
                        item_count=len(memories),
                    )
                    
                    # Serialize report to dict
                    report_dict = self._serialize_report_for_cache(report)
                    
                    success = self._cache.store(cache_key, meta, report_dict)
                    if success:
                        self._logger.debug(f"Cached analysis result: {cache_key[:16]}...")
                    else:
                        self._logger.warning("Failed to cache analysis result")
                except Exception as e:
                    # Never let cache failures break the analysis
                    self._logger.warning(f"Cache storage error (non-fatal): {e}")
            
            return report
            
        except AIUnavailableError:
            # Re-raise availability errors
            raise
        except Exception as e:
            self._logger.error(f"Analysis failed: {e}")
            
            if self._config.fail_on_partial:
                raise
            
            # Return partial report
            report.is_partial = True
            report.partial_failures.append(f"critical: {e}")
            report.generation_time_seconds = time.time() - self._start_time
            
            return report
    
    # =========================================================================
    # Cache Serialization Helpers
    # =========================================================================
    
    def _serialize_report_for_cache(self, report: LifeStoryReport) -> dict[str, Any]:
        """Convert a LifeStoryReport to a cache-safe dictionary.
        
        Uses Pydantic's model_dump() to serialize the report.
        
        Args:
            report: The report to serialize.
        
        Returns:
            Dictionary representation of the report.
        """
        return report.model_dump(mode="json")
    
    def _reconstruct_report_from_cache(self, payload: dict[str, Any]) -> LifeStoryReport:
        """Reconstruct a LifeStoryReport from a cached dictionary.
        
        Uses Pydantic's model_validate() to rebuild the report with proper validation.
        
        Args:
            payload: Cached report dictionary.
        
        Returns:
            Reconstructed LifeStoryReport.
        
        Raises:
            ValueError: If the payload cannot be validated.
        """
        return LifeStoryReport.model_validate(payload)
    
    # =========================================================================
    # Private Analysis Methods
    # =========================================================================
    
    def _detect_chapters(
        self,
        timeline: Timeline,
        context: PromptContext,
    ) -> list[LifeChapter]:
        """Use AI to identify life chapters.
        
        Args:
            timeline: The timeline to analyze.
            context: Prepared prompt context.
        
        Returns:
            List of LifeChapter objects.
        """
        client = self._get_client()
        
        # Render the prompt
        system, user = CHAPTER_DETECTION_PROMPT.render(
            date_range=context.date_range,
            total_memories=context.total_memories,
            platform_breakdown=context.platform_breakdown,
            timeline_summary=context.timeline_summary,
            sample_memories=context.sample_memories,
            min_chapters=self._config.min_chapters,
            max_chapters=self._config.max_chapters,
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        if not response.parse_success:
            self._logger.warning(f"Chapter detection JSON parse failed: {response.parse_error}")
            # Try to extract chapters anyway
            return self._fallback_parse_chapters(response.raw_text, timeline)
        
        # Parse chapters from response
        chapters = self._parse_chapters_from_response(response.data, timeline)
        
        # Validate and fix overlaps
        chapters = self._merge_overlapping_chapters(chapters)
        
        # Assign memories to chapters
        self._assign_memories_to_chapters(
            [m for m in timeline],
            chapters,
        )
        
        return chapters
    
    def _generate_chapter_narrative(
        self,
        chapter: LifeChapter,
        memories: list[Memory],
        chapter_index: int,
        all_chapters: list[LifeChapter],
    ) -> LifeChapter:
        """Generate narrative for a single chapter.
        
        Args:
            chapter: The chapter to generate narrative for.
            memories: Memories in this chapter.
            chapter_index: Index in chapter list.
            all_chapters: All chapters for context.
        
        Returns:
            Updated LifeChapter with narrative.
        """
        client = self._get_client()
        
        # Sample memories if too many
        if len(memories) > self._config.max_memories_per_chapter_narrative:
            memories = self._sample_memories_for_narrative(
                memories,
                chapter,
                self._config.max_memories_per_chapter_narrative,
            )
        
        # Prepare chapter memories as JSON
        chapter_memories = prepare_memories_for_prompt(
            memories,
            max_items=self._config.max_memories_per_chapter_narrative,
            privacy_level=self._config.privacy_level,
        )
        
        # Context about surrounding chapters
        prev_summary = "N/A (first chapter)"
        next_summary = "N/A (last chapter)"
        
        if chapter_index > 0:
            prev = all_chapters[chapter_index - 1]
            prev_summary = f"{prev.title} ({prev.start_date} to {prev.end_date})"
        
        if chapter_index < len(all_chapters) - 1:
            next_ = all_chapters[chapter_index + 1]
            next_summary = f"{next_.title} ({next_.start_date} to {next_.end_date})"
        
        # Render prompt
        system, user = NARRATIVE_GENERATION_PROMPT.render(
            chapter_title=chapter.title,
            chapter_start=str(chapter.start_date),
            chapter_end=str(chapter.end_date),
            chapter_themes=", ".join(chapter.themes),
            memory_count=len(memories),
            chapter_memories=chapter_memories,
            chapter_number=chapter_index + 1,
            total_chapters=len(all_chapters),
            previous_chapter_summary=prev_summary,
            next_chapter_summary=next_summary,
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        if response.parse_success:
            data = response.data
            chapter.narrative = data.get("narrative", "")
            chapter.opening_line = data.get("opening_line", "")
            chapter.key_events = data.get("key_events", [])
            chapter.insights = data.get("insights", [])
            if "confidence" in data:
                chapter.confidence = float(data["confidence"])
        else:
            self._logger.warning(f"Narrative parse failed for {chapter.title}")
            # Use raw text as narrative if JSON failed
            if response.raw_text:
                chapter.narrative = response.raw_text[:2000]
        
        return chapter
    
    def _analyze_platforms(
        self,
        timeline: Timeline,
        memories: list[Memory],
    ) -> list[PlatformBehaviorInsight]:
        """Analyze cross-platform behavior patterns.
        
        Args:
            timeline: The timeline being analyzed.
            memories: All memories.
        
        Returns:
            List of platform insights.
        """
        client = self._get_client()
        
        # Get platform statistics
        platform_stats_lines = []
        platform_samples: dict[str, str] = {}
        total_memories = len(memories)
        
        for platform in SourcePlatform:
            platform_memories = timeline.get_memories_by_platform(platform)
            if platform_memories:
                count = len(platform_memories)
                pct = (count / total_memories) * 100 if total_memories > 0 else 0
                platform_stats_lines.append(f"- {platform.value}: {count} ({pct:.1f}%)")
                
                # Sample memories for this platform
                sample = prepare_memories_for_prompt(
                    platform_memories[:50],
                    max_items=20,
                    privacy_level=self._config.privacy_level,
                )
                platform_samples[platform.value] = sample
        
        platform_stats = "\n".join(platform_stats_lines)
        samples_text = "\n\n".join(
            f"### {name}\n{sample}" for name, sample in platform_samples.items()
        )
        
        # Render prompt
        system, user = PLATFORM_ANALYSIS_PROMPT.render(
            platform_stats=platform_stats,
            platform_samples=samples_text,
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        insights: list[PlatformBehaviorInsight] = []
        
        if response.parse_success and "platforms" in response.data:
            for p_data in response.data["platforms"]:
                try:
                    platform_name = p_data.get("name", "unknown")
                    platform_enum = SourcePlatform(platform_name)
                    
                    insight = PlatformBehaviorInsight(
                        platform=platform_enum,
                        usage_pattern=p_data.get("primary_use", ""),
                        peak_period=p_data.get("peak_period"),
                        unique_characteristics=p_data.get("unique_patterns", []),
                        memory_count=p_data.get("memory_count", 0),
                    )
                    insights.append(insight)
                except (ValueError, KeyError) as e:
                    self._logger.warning(f"Failed to parse platform insight: {e}")
        
        return insights
    
    def _analyze_gaps(
        self,
        timeline: Timeline,
        gaps: list[TimelineGap],
    ) -> list[DataGap]:
        """Analyze timeline gaps with AI explanations.
        
        Args:
            timeline: The timeline.
            gaps: Detected gaps from timeline.
        
        Returns:
            List of analyzed gaps.
        """
        client = self._get_client()
        
        # Filter to significant gaps
        significant_gaps = [g for g in gaps if g.duration_days >= 30][:10]
        
        if not significant_gaps:
            return []
        
        # Format gaps for prompt
        gaps_data_lines = []
        for g in significant_gaps:
            gaps_data_lines.append(
                f"- {g.start_date} to {g.end_date}: {g.duration_days} days"
            )
        
        # Get surrounding context
        gap_context_parts = []
        for g in significant_gaps[:5]:
            # Get memories just before and after
            before = timeline.get_memories_in_range(
                g.start_date - timedelta(days=30),
                g.start_date,
            )[-5:] if g.start_date else []
            after = timeline.get_memories_in_range(
                g.end_date,
                g.end_date + timedelta(days=30),
            )[:5] if g.end_date else []
            
            if before or after:
                gap_context_parts.append(
                    f"Gap {g.start_date} to {g.end_date}:\n"
                    f"  Before: {len(before)} memories\n"
                    f"  After: {len(after)} memories"
                )
        
        # Render prompt
        system, user = GAP_ANALYSIS_PROMPT.render(
            gaps_data="\n".join(gaps_data_lines),
            gap_context="\n".join(gap_context_parts) or "No surrounding context available",
            timeline_summary=prepare_timeline_summary([m for m in timeline]),
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        data_gaps: list[DataGap] = []
        
        if response.parse_success and "gaps" in response.data:
            for gap_data in response.data["gaps"]:
                try:
                    data_gap = DataGap(
                        start_date=self._parse_date(gap_data.get("start_date", "")),
                        end_date=self._parse_date(gap_data.get("end_date", "")),
                        duration_days=gap_data.get("duration_days", 0),
                        possible_explanations=gap_data.get("possible_explanations", []),
                        severity=gap_data.get("gap_type", "minor"),
                        impacts_narrative=bool(gap_data.get("narrative_impact")),
                    )
                    data_gaps.append(data_gap)
                except Exception as e:
                    self._logger.warning(f"Failed to parse gap: {e}")
        
        return data_gaps
    
    def _generate_executive_summary(
        self,
        chapters: list[LifeChapter],
        timeline: Timeline,
        stats: TimelineStatistics,
    ) -> str:
        """Generate overall life story summary.
        
        Args:
            chapters: All detected chapters.
            timeline: The timeline.
            stats: Timeline statistics.
        
        Returns:
            Executive summary text.
        """
        client = self._get_client()
        
        # Prepare chapters summary
        chapters_summary = prepare_chapters_for_prompt(
            [c.to_dict() for c in chapters]
        )
        
        # Render prompt
        system, user = EXECUTIVE_SUMMARY_PROMPT.render(
            chapters_summary=chapters_summary,
            total_memories=len(timeline),
            date_range=f"{stats.date_range.start} to {stats.date_range.end}" if stats.date_range else "Unknown",
            platforms=prepare_platform_breakdown([m for m in timeline]),
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        if response.parse_success and "summary" in response.data:
            return response.data["summary"]
        elif response.raw_text:
            return response.raw_text[:3000]
        else:
            return self._generate_fallback_summary(chapters)
    
    def _detect_patterns(
        self,
        timeline: Timeline,
        chapters: list[LifeChapter],
    ) -> list[str]:
        """Detect cross-cutting behavioral patterns.
        
        Args:
            timeline: The timeline.
            chapters: Detected chapters.
        
        Returns:
            List of pattern descriptions.
        """
        client = self._get_client()
        
        # Prepare context
        context = build_prompt_context([m for m in timeline])
        
        # Render prompt
        system, user = PATTERN_DETECTION_PROMPT.render(
            date_range=context.date_range,
            total_memories=context.total_memories,
            timeline_summary=context.timeline_summary,
            sample_memories=context.sample_memories,
        )
        
        # Call AI
        response = client.generate_json(user, system_instruction=system)
        self._tokens_used += response.tokens_used
        
        patterns: list[str] = []
        
        if response.parse_success:
            data = response.data
            
            # Extract patterns from various categories
            for key in ["temporal_patterns", "location_patterns", "social_patterns"]:
                if key in data:
                    for p in data[key]:
                        if isinstance(p, dict) and "pattern" in p:
                            patterns.append(p["pattern"])
                        elif isinstance(p, str):
                            patterns.append(p)
            
            if "documentation_style" in data:
                patterns.append(f"Documentation style: {data['documentation_style']}")
            
            if "notable_anomalies" in data:
                patterns.extend(data["notable_anomalies"])
        
        return patterns[:20]  # Cap at 20 patterns
    
    # =========================================================================
    # Data Preparation Methods
    # =========================================================================
    
    def _prepare_timeline_for_ai(self, timeline: Timeline) -> PromptContext:
        """Prepare timeline data for AI prompts.
        
        Args:
            timeline: The timeline to prepare.
        
        Returns:
            PromptContext with all necessary data.
        """
        # Sample memories for AI
        sampled = self._sample_memories_for_detection(
            list(timeline),
            self._config.max_memories_for_chapter_detection,
        )
        
        # Build context using helper
        context = build_prompt_context(sampled)
        
        return context
    
    def _sample_memories_for_detection(
        self,
        memories: list[Memory],
        max_count: int,
    ) -> list[Memory]:
        """Intelligently sample memories for chapter detection.
        
        Strategy:
        - Ensure temporal coverage (samples from entire range)
        - Include variety of platforms
        - Prioritize memories with rich metadata
        - Include edge memories (first, last)
        
        Args:
            memories: All memories.
            max_count: Maximum to sample.
        
        Returns:
            Sampled list of memories.
        """
        if len(memories) <= max_count:
            return memories
        
        # Always include first and last
        result: list[Memory] = []
        dated = [m for m in memories if m.timestamp]
        
        if dated:
            dated.sort(key=lambda m: m.timestamp)
            result.append(dated[0])
            result.append(dated[-1])
        
        # Stratified sampling by year
        remaining = max_count - len(result)
        by_year: dict[int, list[Memory]] = {}
        
        for m in memories:
            if m.timestamp:
                year = m.timestamp.year
                if year not in by_year:
                    by_year[year] = []
                by_year[year].append(m)
        
        if by_year:
            per_year = max(1, remaining // len(by_year))
            for year in sorted(by_year.keys()):
                year_memories = by_year[year]
                # Prefer memories with rich metadata
                year_memories.sort(
                    key=lambda m: (
                        (1 if m.location else 0) +
                        (1 if m.people else 0) +
                        (1 if m.caption else 0)
                    ),
                    reverse=True,
                )
                result.extend(year_memories[:per_year])
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for m in result:
            if m.id not in seen:
                seen.add(m.id)
                unique.append(m)
        
        return unique[:max_count]
    
    def _sample_memories_for_narrative(
        self,
        memories: list[Memory],
        chapter: LifeChapter,
        max_count: int,
    ) -> list[Memory]:
        """Sample memories for a chapter's narrative.
        
        Args:
            memories: Chapter memories.
            chapter: The chapter.
            max_count: Maximum to include.
        
        Returns:
            Sampled memories.
        """
        if len(memories) <= max_count:
            return memories
        
        # Prioritize memories with rich metadata
        def richness(m: Memory) -> int:
            score = 0
            if m.location:
                score += 2
            if m.people:
                score += len(m.people)
            if m.caption:
                score += 1
            return score
        
        # Sort by richness, then sample stratified
        memories = sorted(memories, key=richness, reverse=True)
        
        # Take most rich, plus some random for variety
        rich_count = max_count * 2 // 3
        random_count = max_count - rich_count
        
        result = memories[:rich_count]
        
        # Add some from the middle of the chapter
        remaining = memories[rich_count:]
        if remaining:
            step = max(1, len(remaining) // random_count)
            result.extend(remaining[::step][:random_count])
        
        return result[:max_count]
    
    # =========================================================================
    # Validation & Parsing Methods
    # =========================================================================
    
    def _parse_chapters_from_response(
        self,
        data: dict[str, Any],
        timeline: Timeline,
    ) -> list[LifeChapter]:
        """Parse AI response into LifeChapter objects.
        
        Args:
            data: Parsed JSON response.
            timeline: The timeline for context.
        
        Returns:
            List of LifeChapter objects.
        """
        chapters: list[LifeChapter] = []
        
        if "chapters" not in data:
            return chapters
        
        for ch_data in data["chapters"]:
            try:
                chapter = LifeChapter(
                    title=ch_data.get("title", "Untitled Chapter"),
                    start_date=self._parse_date(ch_data.get("start_date", "")),
                    end_date=self._parse_date(ch_data.get("end_date", "")),
                    themes=ch_data.get("themes", []),
                    reasoning=ch_data.get("reasoning", ""),
                    confidence=float(ch_data.get("confidence", 0.5)),
                    memory_count=ch_data.get("estimated_memory_count", 0),
                )
                chapters.append(chapter)
            except Exception as e:
                self._logger.warning(f"Failed to parse chapter: {e}")
        
        # Sort by start date
        chapters.sort(key=lambda c: c.start_date)
        
        return chapters
    
    def _fallback_parse_chapters(
        self,
        raw_text: str,
        timeline: Timeline,
    ) -> list[LifeChapter]:
        """Create fallback chapters when AI response fails.
        
        Args:
            raw_text: Raw AI response text.
            timeline: The timeline.
        
        Returns:
            Fallback chapter list.
        """
        self._logger.warning("Using fallback chapter detection")
        return self._create_fallback_chapters(timeline)
    
    def _create_fallback_chapters(self, timeline: Timeline) -> list[LifeChapter]:
        """Create basic chapters based on years.
        
        Args:
            timeline: The timeline.
        
        Returns:
            Year-based chapters.
        """
        chapters: list[LifeChapter] = []
        
        # Group by year
        years = timeline.get_activity_by_year()
        
        for year, count in sorted(years.items()):
            chapters.append(
                LifeChapter(
                    title=f"Year {year}",
                    start_date=date(year, 1, 1),
                    end_date=date(year, 12, 31),
                    themes=["yearly-overview"],
                    memory_count=count,
                    confidence=0.3,
                    reasoning="Fallback: created from yearly grouping",
                )
            )
        
        return chapters
    
    def _merge_overlapping_chapters(
        self,
        chapters: list[LifeChapter],
    ) -> list[LifeChapter]:
        """Fix overlapping chapter boundaries.
        
        Args:
            chapters: Chapters to validate.
        
        Returns:
            Fixed chapter list.
        """
        if len(chapters) <= 1:
            return chapters
        
        # Sort by start date
        chapters.sort(key=lambda c: c.start_date)
        
        # Fix overlaps by adjusting end dates
        for i in range(len(chapters) - 1):
            current = chapters[i]
            next_ch = chapters[i + 1]
            
            if current.end_date >= next_ch.start_date:
                # Adjust current's end to day before next starts
                from datetime import timedelta
                current.end_date = next_ch.start_date - timedelta(days=1)
        
        return chapters
    
    def _assign_memories_to_chapters(
        self,
        memories: list[Memory],
        chapters: list[LifeChapter],
    ) -> None:
        """Assign each memory to its containing chapter.
        
        Updates chapter.memory_ids in place.
        
        Args:
            memories: All memories.
            chapters: Chapters to assign to.
        """
        for chapter in chapters:
            chapter.memory_ids = []
            chapter.memory_count = 0
        
        for memory in memories:
            if not memory.timestamp:
                continue
            
            memory_date = memory.timestamp.date() if hasattr(memory.timestamp, 'date') else memory.timestamp
            
            for chapter in chapters:
                if chapter.start_date <= memory_date <= chapter.end_date:
                    chapter.memory_ids.append(memory.id)
                    chapter.memory_count += 1
                    break
    
    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object.
        
        Args:
            date_str: Date string (YYYY-MM-DD format).
        
        Returns:
            date object.
        """
        if not date_str:
            return date.today()
        
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            # Try other formats
            for fmt in ["%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            
            self._logger.warning(f"Could not parse date: {date_str}")
            return date.today()
    
    def _assess_data_quality(self, stats: TimelineStatistics) -> list[str]:
        """Assess quality of timeline data.
        
        Args:
            stats: Timeline statistics.
        
        Returns:
            List of quality notes.
        """
        notes: list[str] = []
        
        if stats.undated_memories > 0:
            pct = (stats.undated_memories / stats.total_memories) * 100 if stats.total_memories else 0
            notes.append(f"{stats.undated_memories} memories ({pct:.1f}%) lack timestamps")
        
        if stats.location_coverage < 0.2:
            notes.append(f"Only {stats.location_coverage * 100:.0f}% of memories have location data")
        
        if stats.people_coverage < 0.1:
            notes.append(f"Only {stats.people_coverage * 100:.0f}% of memories have people tags")
        
        if stats.gaps:
            major_gaps = [g for g in stats.gaps if g.duration_days > 90]
            if major_gaps:
                notes.append(f"Found {len(major_gaps)} gaps longer than 3 months")
        
        return notes
    
    def _generate_fallback_summary(self, chapters: list[LifeChapter]) -> str:
        """Create fallback summary when AI fails.
        
        Args:
            chapters: Detected chapters.
        
        Returns:
            Basic summary text.
        """
        if not chapters:
            return "Unable to generate executive summary due to insufficient data."
        
        total_memories = sum(c.memory_count for c in chapters)
        date_range = f"{chapters[0].start_date} to {chapters[-1].end_date}"
        
        return (
            f"This life story spans {len(chapters)} chapters from {date_range}, "
            f"covering {total_memories} memories. "
            f"The chapters include: {', '.join(c.title for c in chapters[:5])}"
            f"{'...' if len(chapters) > 5 else ''}."
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_memories(
    memories: list[Memory],
    config: AnalysisConfig | None = None,
    progress: Callable[[AnalysisProgress], None] | None = None,
) -> LifeStoryReport:
    """Convenience function for life story analysis.
    
    Simpler interface for common use cases.
    
    Args:
        memories: List of Memory objects to analyze.
        config: Optional analysis configuration.
        progress: Optional progress callback.
    
    Returns:
        LifeStoryReport with analysis results.
    
    Example:
        >>> report = analyze_memories(memories)
        >>> print(report.executive_summary)
    """
    analyzer = LifeStoryAnalyzer(config=config)
    return analyzer.analyze(memories, progress_callback=progress)


def quick_analyze(memories: list[Memory]) -> LifeStoryReport:
    """Quickest path to a life story report.
    
    Uses all defaults for fastest results.
    
    Args:
        memories: List of Memory objects.
    
    Returns:
        LifeStoryReport.
    
    Example:
        >>> report = quick_analyze(memories)
    """
    config = AnalysisConfig(
        min_chapters=3,
        max_chapters=8,
        include_platform_analysis=False,
        include_gap_analysis=False,
        detect_patterns=False,
    )
    return analyze_memories(memories, config=config)


# Import for gap analysis date math
from datetime import timedelta
