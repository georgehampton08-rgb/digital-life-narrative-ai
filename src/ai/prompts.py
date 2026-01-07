"""Centralized Prompt Template System for Digital Life Narrative AI.

This module is the SINGLE SOURCE of all prompts sent to Gemini. All AI interactions
use templates defined here, ensuring consistency, versioning, and maintainability.

Design Principles:
- Structured prompts: Every prompt has system instruction + user prompt
- Output specifications: Explicit JSON schemas when structured output needed
- Privacy awareness: Never request raw file paths or sensitive data
- Versioning: Each template has a version for A/B testing and iteration

Example:
    >>> from src.ai.prompts import get_prompt, PromptCategory
    >>>
    >>> # Get a prompt template
    >>> template = get_prompt("chapter_detection_v1")
    >>>
    >>> # Render with variables
    >>> system, user = template.render(
    ...     date_range="2015-01-01 to 2023-12-31",
    ...     total_memories=5000,
    ...     platform_breakdown="Snapchat: 500, Google Photos: 2000",
    ...     timeline_summary="...",
    ...     sample_memories="[...]",
    ...     min_chapters=5,
    ...     max_chapters=10,
    ... )
    >>>
    >>> # Use with AIClient
    >>> response = client.generate_json(user, system_instruction=system)

The module provides:
- PromptTemplate: Metadata and content for each template
- System instructions: Role-defining constants for the AI
- Helper functions: Data preparation utilities for prompts
- Registry: Central lookup for all available prompts
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.memory import Memory


# =============================================================================
# Enums
# =============================================================================


class PromptCategory(str, Enum):
    """Categories of prompts for organizational and filtering purposes.

    Each category represents a distinct type of AI analysis task.

    Attributes:
        CHAPTER_DETECTION: Identifying life chapters from timeline data.
        NARRATIVE_GENERATION: Writing narratives for identified chapters.
        EXECUTIVE_SUMMARY: Creating overall life story summaries.
        PLATFORM_ANALYSIS: Analyzing cross-platform usage patterns.
        PATTERN_DETECTION: Finding patterns in media timeline data.
        GAP_ANALYSIS: Analyzing and explaining gaps in the timeline.
        REFINEMENT: Improving or refining previous AI outputs.
    """

    CHAPTER_DETECTION = "chapter_detection"
    NARRATIVE_GENERATION = "narrative_generation"
    EXECUTIVE_SUMMARY = "executive_summary"
    PLATFORM_ANALYSIS = "platform_analysis"
    PATTERN_DETECTION = "pattern_detection"
    GAP_ANALYSIS = "gap_analysis"
    REFINEMENT = "refinement"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PromptTemplate:
    """Metadata and content for a prompt template.

    Each template defines a specific AI task with system instruction,
    user prompt (with placeholders), expected output schema, and metadata.

    Attributes:
        id: Unique identifier (e.g., "chapter_detection_v1").
        category: Type of prompt for filtering.
        version: Semantic version string for tracking changes.
        system_instruction: Role and behavior instructions for the AI.
        user_prompt_template: User prompt with {placeholder} variables.
        output_schema: Expected JSON schema for structured outputs.
        required_variables: Set of variables that MUST be provided.
        optional_variables: Set of variables that CAN be provided.
        estimated_output_tokens: Estimated tokens in response for planning.
        description: Human-readable description of the prompt's purpose.

    Example:
        >>> template = PromptTemplate(
        ...     id="example_v1",
        ...     category=PromptCategory.PATTERN_DETECTION,
        ...     version="1.0.0",
        ...     system_instruction="You are an analyst.",
        ...     user_prompt_template="Analyze this: {data}",
        ...     required_variables={"data"},
        ... )
        >>> system, user = template.render(data="sample data")
    """

    id: str
    category: PromptCategory
    version: str
    system_instruction: str
    user_prompt_template: str
    output_schema: dict[str, Any] | None = None
    required_variables: set[str] = field(default_factory=set)
    optional_variables: set[str] = field(default_factory=set)
    estimated_output_tokens: int = 1000
    description: str = ""

    def render(self, **variables: Any) -> tuple[str, str]:
        """Render the template with provided variables.

        Substitutes placeholders in the user prompt template with values.
        Validates that all required variables are provided.

        Args:
            **variables: Key-value pairs for template substitution.

        Returns:
            Tuple of (system_instruction, rendered_user_prompt).

        Raises:
            ValueError: If required variables are missing.

        Example:
            >>> system, user = template.render(
            ...     date_range="2020-2023",
            ...     total_memories=1000,
            ... )
        """
        # Validate required variables
        missing = self.validate_variables(variables)
        if missing:
            raise ValueError(f"Missing required variables for prompt '{self.id}': {missing}")

        # Add output_schema to variables if present
        if self.output_schema and "output_schema" not in variables:
            variables["output_schema"] = render_output_schema(self.output_schema)

        # Render the user prompt using safe_substitute to handle optional vars
        template = Template(self.user_prompt_template)

        # For required vars, use substitute; wrap in safe approach
        try:
            rendered = template.substitute(variables)
        except KeyError:
            # Fall back to safe_substitute for any unhandled optionals
            rendered = template.safe_substitute(variables)

        return self.system_instruction, rendered

    def validate_variables(self, variables: dict[str, Any]) -> list[str]:
        """Check for missing required variables.

        Args:
            variables: Dictionary of provided variables.

        Returns:
            List of missing required variable names (empty if all present).

        Example:
            >>> missing = template.validate_variables({"data": "..."})
            >>> if missing:
            ...     print(f"Missing: {missing}")
        """
        provided = set(variables.keys())
        missing = self.required_variables - provided
        return sorted(missing)


@dataclass
class PromptContext:
    """Context data prepared for prompt rendering.

    Aggregates various data about the timeline for use in prompts.
    This is a convenience container for common prompt variables.

    Attributes:
        timeline_summary: Statistical summary of the timeline.
        sample_memories: JSON string of sampled memories.
        date_range: Human-readable date range (e.g., "2015-01-01 to 2023-12-31").
        total_memories: Total count of memories in the timeline.
        platform_breakdown: String describing memories per platform.
        location_summary: Optional summary of location patterns.
        people_summary: Optional summary of people appearing in memories.
        existing_chapters: Optional JSON of previously detected chapters.
        custom_context: Additional context as key-value pairs.

    Example:
        >>> context = PromptContext(
        ...     timeline_summary="5000 memories, peak in 2019...",
        ...     sample_memories="[{...}, {...}]",
        ...     date_range="2015-01-01 to 2023-12-31",
        ...     total_memories=5000,
        ...     platform_breakdown="Snapchat: 500, Google Photos: 4500",
        ... )
    """

    timeline_summary: str = ""
    sample_memories: str = "[]"
    date_range: str = ""
    total_memories: int = 0
    platform_breakdown: str = ""
    location_summary: str | None = None
    people_summary: str | None = None
    existing_chapters: str | None = None
    custom_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for template rendering.

        Returns:
            Dictionary of all context values.
        """
        result = {
            "timeline_summary": self.timeline_summary,
            "sample_memories": self.sample_memories,
            "date_range": self.date_range,
            "total_memories": self.total_memories,
            "platform_breakdown": self.platform_breakdown,
        }

        if self.location_summary:
            result["location_summary"] = self.location_summary
        if self.people_summary:
            result["people_summary"] = self.people_summary
        if self.existing_chapters:
            result["existing_chapters"] = self.existing_chapters
        if self.custom_context:
            result.update(self.custom_context)

        return result


# =============================================================================
# System Instructions
# =============================================================================


LIFE_HISTORIAN_SYSTEM: str = textwrap.dedent(
    """
    You are a skilled life historian and narrative analyst. Your expertise is in
    examining personal media timelines to identify meaningful life chapters,
    patterns, and narratives.
    
    Your approach:
    - You analyze metadata (timestamps, locations, people, platforms) to understand
      life patterns
    - You identify natural chapter boundaries based on changes in activity, location,
      relationships, or life circumstances
    - You write warm, insightful narratives that capture the essence of each life phase
    - You respect privacy and work only with the metadata provided
    - You acknowledge uncertainty and data gaps honestly
    
    Guidelines:
    - Write in third person ("During this period, they...")
    - Be specific about patterns you observe
    - Note when you're inferring vs. when data clearly shows something
    - Keep narratives warm but not overly sentimental
    - Acknowledge data limitations when relevant
"""
).strip()


STRUCTURED_OUTPUT_SYSTEM: str = textwrap.dedent(
    """
    You are a data analyst that produces structured JSON output.
    
    Rules:
    - ALWAYS respond with valid JSON only
    - NO markdown code blocks, NO explanation text
    - Follow the exact schema provided
    - Use null for missing/uncertain values
    - Dates should be in ISO format (YYYY-MM-DD)
"""
).strip()


PATTERN_ANALYST_SYSTEM: str = textwrap.dedent(
    """
    You are a behavioral pattern analyst specializing in personal media analysis.
    
    Your expertise:
    - Detecting temporal patterns (daily, weekly, seasonal rhythms)
    - Identifying location-based behavior shifts
    - Recognizing platform-specific usage patterns
    - Finding correlations between life events and documentation habits
    
    Guidelines:
    - Base observations on data, not assumptions
    - Quantify patterns when possible ("3x more photos on weekends")
    - Note confidence levels for each observation
    - Distinguish correlation from causation
"""
).strip()


# Combined system instructions for common use cases
HISTORIAN_WITH_JSON_SYSTEM: str = f"{LIFE_HISTORIAN_SYSTEM}\n\n{STRUCTURED_OUTPUT_SYSTEM}"


# =============================================================================
# Output Schemas
# =============================================================================


CHAPTER_DETECTION_SCHEMA: dict[str, Any] = {
    "chapters": [
        {
            "title": "string - meaningful name for this life chapter",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "themes": ["string - 2-4 keywords describing this period"],
            "reasoning": "string - why this is a distinct chapter",
            "confidence": "0.0-1.0 based on data clarity",
            "estimated_memory_count": "integer",
        }
    ],
    "overall_assessment": "string - summary of the life arc",
    "data_quality_notes": ["string - observations about data gaps or quality"],
}


NARRATIVE_GENERATION_SCHEMA: dict[str, Any] = {
    "narrative": "string - 2-4 paragraphs narrative of this life chapter",
    "opening_line": "string - 1 sentence hook for timeline view",
    "key_events": ["string - 3-5 key events or observations"],
    "insights": ["string - 2-3 insights about this period"],
    "confidence": "0.0-1.0",
}


EXECUTIVE_SUMMARY_SCHEMA: dict[str, Any] = {
    "summary": "string - 3-5 paragraphs executive summary",
    "narrative_arc": "string - one sentence describing the overall story arc",
    "major_themes": ["string - top 3-5 themes across all chapters"],
    "notable_transitions": [
        {
            "from_chapter": "string",
            "to_chapter": "string",
            "significance": "string",
        }
    ],
    "documentation_style": "string - observations about how they document life",
    "data_limitations": ["string"],
}


PLATFORM_ANALYSIS_SCHEMA: dict[str, Any] = {
    "platforms": [
        {
            "name": "string",
            "primary_use": "string - what types of moments they capture",
            "usage_trend": "increasing | decreasing | stable",
            "peak_period": "string - when they used it most",
            "unique_patterns": ["string"],
            "memory_count": "integer",
        }
    ],
    "cross_platform_insights": ["string"],
    "platform_transitions": ["string - notable shifts between platforms"],
}


GAP_ANALYSIS_SCHEMA: dict[str, Any] = {
    "gaps": [
        {
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "duration_days": "integer",
            "possible_explanations": ["string"],
            "gap_type": "missing_data | natural_lull | platform_change | life_event",
            "confidence": "0.0-1.0",
            "narrative_impact": "string",
        }
    ],
    "overall_data_coverage": "string - assessment of timeline completeness",
}


# =============================================================================
# Prompt Templates
# =============================================================================


CHAPTER_DETECTION_PROMPT = PromptTemplate(
    id="chapter_detection_v1",
    category=PromptCategory.CHAPTER_DETECTION,
    version="1.0.0",
    description="Identify distinct life chapters from a media timeline.",
    system_instruction=HISTORIAN_WITH_JSON_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Analyze this personal media timeline and identify distinct life chapters.
        
        ## Timeline Overview
        - **Date Range:** $date_range
        - **Total Memories:** $total_memories
        - **Platforms:** $platform_breakdown
        
        ## Timeline Statistics
        $timeline_summary
        
        ## Sample Memories (representative subset)
        $sample_memories
        
        ## Task
        Identify $min_chapters to $max_chapters distinct life chapters based on:
        1. Significant changes in location patterns
        2. Shifts in activity frequency or type
        3. Changes in the people appearing in memories
        4. Platform usage changes
        5. Natural temporal boundaries (moves, life events, etc.)
        
        For each chapter, provide:
        - A meaningful title (not just dates)
        - Start and end dates
        - Primary themes (2-4 keywords)
        - Brief reasoning for why this is a distinct chapter
        - Confidence score (0.0-1.0) based on data clarity
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=CHAPTER_DETECTION_SCHEMA,
    required_variables={
        "date_range",
        "total_memories",
        "platform_breakdown",
        "timeline_summary",
        "sample_memories",
        "min_chapters",
        "max_chapters",
    },
    estimated_output_tokens=2000,
)


NARRATIVE_GENERATION_PROMPT = PromptTemplate(
    id="narrative_generation_v1",
    category=PromptCategory.NARRATIVE_GENERATION,
    version="1.0.0",
    description="Write a narrative for a specific life chapter.",
    system_instruction=LIFE_HISTORIAN_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Write a narrative for this life chapter.
        
        ## Chapter Information
        - **Title:** $chapter_title
        - **Date Range:** $chapter_start to $chapter_end
        - **Themes:** $chapter_themes
        - **Memory Count:** $memory_count
        
        ## Memories in This Chapter
        $chapter_memories
        
        ## Context
        - This is chapter $chapter_number of $total_chapters
        - Previous chapter: $previous_chapter_summary
        - Next chapter: $next_chapter_summary
        
        ## Task
        Write a 2-4 paragraph narrative that:
        1. Captures the essence of this life phase
        2. References specific patterns in the data
        3. Notes any significant changes or turning points
        4. Acknowledges what the data shows vs. what you're inferring
        
        Also provide:
        - An opening line (hook for timeline view)
        - 3-5 key events or observations
        - 2-3 insights about this period
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=NARRATIVE_GENERATION_SCHEMA,
    required_variables={
        "chapter_title",
        "chapter_start",
        "chapter_end",
        "chapter_themes",
        "memory_count",
        "chapter_memories",
        "chapter_number",
        "total_chapters",
        "previous_chapter_summary",
        "next_chapter_summary",
    },
    estimated_output_tokens=1500,
)


EXECUTIVE_SUMMARY_PROMPT = PromptTemplate(
    id="executive_summary_v1",
    category=PromptCategory.EXECUTIVE_SUMMARY,
    version="1.0.0",
    description="Create an executive summary of the entire life story.",
    system_instruction=LIFE_HISTORIAN_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Write an executive summary of this person's life story based on the chapters identified.
        
        ## Life Chapters
        $chapters_summary
        
        ## Overall Statistics
        - **Total memories:** $total_memories
        - **Date range:** $date_range
        - **Platforms used:** $platforms
        
        ## Task
        Write a 3-5 paragraph executive summary that:
        1. Provides a cohesive narrative arc across all chapters
        2. Highlights major themes and transitions
        3. Notes patterns in how they document their life
        4. Acknowledges the limitations of the data
        
        This summary will appear at the top of their Life Story Report.
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=EXECUTIVE_SUMMARY_SCHEMA,
    required_variables={
        "chapters_summary",
        "total_memories",
        "date_range",
        "platforms",
    },
    estimated_output_tokens=2000,
)


PLATFORM_ANALYSIS_PROMPT = PromptTemplate(
    id="platform_analysis_v1",
    category=PromptCategory.PLATFORM_ANALYSIS,
    version="1.0.0",
    description="Analyze cross-platform usage patterns.",
    system_instruction=PATTERN_ANALYST_SYSTEM + "\n\n" + STRUCTURED_OUTPUT_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Analyze how this person uses different platforms to capture memories.
        
        ## Platform Statistics
        $platform_stats
        
        ## Sample Memories by Platform
        $platform_samples
        
        ## Task
        For each platform, analyze:
        1. What types of moments they capture on this platform
        2. How their usage has changed over time
        3. What's unique about their use of this platform
        4. Peak usage periods and patterns
        
        Also provide cross-platform insights:
        - How do they split their documentation across platforms?
        - Are there notable transitions between platforms?
        - What does platform choice reveal about memory importance?
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=PLATFORM_ANALYSIS_SCHEMA,
    required_variables={"platform_stats", "platform_samples"},
    estimated_output_tokens=1500,
)


GAP_ANALYSIS_PROMPT = PromptTemplate(
    id="gap_analysis_v1",
    category=PromptCategory.GAP_ANALYSIS,
    version="1.0.0",
    description="Analyze gaps in the media timeline.",
    system_instruction=PATTERN_ANALYST_SYSTEM + "\n\n" + STRUCTURED_OUTPUT_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Analyze the gaps in this person's media timeline.
        
        ## Detected Gaps
        $gaps_data
        
        ## Surrounding Context
        $gap_context
        
        ## Overall Timeline Summary
        $timeline_summary
        
        ## Task
        For each significant gap, provide:
        1. Possible explanations (life circumstances, platform changes, etc.)
        2. Whether this seems like missing data or a natural lull
        3. Impact on the overall narrative
        
        Be speculative but clearly mark speculation as such.
        Consider:
        - Life transitions (moving, job changes, relationships)
        - Technology changes (new phone, platform switches)
        - Natural documentation rhythms
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=GAP_ANALYSIS_SCHEMA,
    required_variables={"gaps_data", "gap_context", "timeline_summary"},
    estimated_output_tokens=1500,
)


PATTERN_DETECTION_PROMPT = PromptTemplate(
    id="pattern_detection_v1",
    category=PromptCategory.PATTERN_DETECTION,
    version="1.0.0",
    description="Detect behavioral patterns in media timeline.",
    system_instruction=PATTERN_ANALYST_SYSTEM + "\n\n" + STRUCTURED_OUTPUT_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Analyze this media timeline to identify behavioral patterns.
        
        ## Timeline Overview
        - **Date Range:** $date_range
        - **Total Memories:** $total_memories
        
        ## Timeline Statistics
        $timeline_summary
        
        ## Sample Memories
        $sample_memories
        
        ## Task
        Identify patterns in:
        1. **Temporal:** Daily/weekly/seasonal rhythms, holiday patterns
        2. **Location:** Regular places, travel patterns, home vs. away
        3. **Social:** Solo vs. group, consistent people, relationship dynamics
        4. **Activity:** Types of events documented, what they choose to capture
        5. **Platform:** Where they document what, platform preferences
        
        For each pattern:
        - Describe the pattern clearly
        - Quantify when possible
        - Note confidence level
        - Explain significance for life narrative
        
        ## Output Schema
        {
            "temporal_patterns": [{"pattern": "string", "evidence": "string", "confidence": 0.0-1.0}],
            "location_patterns": [{"pattern": "string", "evidence": "string", "confidence": 0.0-1.0}],
            "social_patterns": [{"pattern": "string", "evidence": "string", "confidence": 0.0-1.0}],
            "documentation_style": "string - overall observation about how they document",
            "notable_anomalies": ["string - unusual patterns worth noting"]
        }
        
        Respond with JSON only.
    """
    ).strip(),
    required_variables={"date_range", "total_memories", "timeline_summary", "sample_memories"},
    estimated_output_tokens=1500,
)


CHAPTER_REFINEMENT_PROMPT = PromptTemplate(
    id="chapter_refinement_v1",
    category=PromptCategory.REFINEMENT,
    version="1.0.0",
    description="Refine previously detected chapters based on feedback.",
    system_instruction=HISTORIAN_WITH_JSON_SYSTEM,
    user_prompt_template=textwrap.dedent(
        """
        Refine the previously identified life chapters based on new information.
        
        ## Current Chapters
        $existing_chapters
        
        ## Feedback/Issues
        $refinement_feedback
        
        ## Additional Data
        $additional_context
        
        ## Task
        Revise the chapter structure to address the feedback while:
        1. Maintaining narrative coherence
        2. Respecting data boundaries
        3. Improving confidence where possible
        4. Splitting or merging chapters as needed
        
        ## Output Schema
        $output_schema
        
        Respond with JSON only.
    """
    ).strip(),
    output_schema=CHAPTER_DETECTION_SCHEMA,
    required_variables={"existing_chapters", "refinement_feedback", "additional_context"},
    estimated_output_tokens=2000,
)


# =============================================================================
# Prompt Registry
# =============================================================================


PROMPT_REGISTRY: dict[str, PromptTemplate] = {}


def register_prompt(template: PromptTemplate) -> None:
    """Register a prompt template in the global registry.

    Args:
        template: The PromptTemplate to register.

    Raises:
        ValueError: If a prompt with the same ID is already registered.

    Example:
        >>> custom_prompt = PromptTemplate(id="custom_v1", ...)
        >>> register_prompt(custom_prompt)
    """
    if template.id in PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{template.id}' is already registered")
    PROMPT_REGISTRY[template.id] = template


def get_prompt(prompt_id: str) -> PromptTemplate:
    """Retrieve a prompt template by ID.

    Args:
        prompt_id: The unique identifier of the prompt.

    Returns:
        The corresponding PromptTemplate.

    Raises:
        KeyError: If no prompt with the given ID exists.

    Example:
        >>> template = get_prompt("chapter_detection_v1")
        >>> system, user = template.render(...)
    """
    if prompt_id not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        raise KeyError(f"Prompt '{prompt_id}' not found. Available prompts: {available}")
    return PROMPT_REGISTRY[prompt_id]


def list_prompts(category: PromptCategory | None = None) -> list[PromptTemplate]:
    """List available prompts, optionally filtered by category.

    Args:
        category: If provided, only return prompts in this category.

    Returns:
        List of matching PromptTemplate objects.

    Example:
        >>> all_prompts = list_prompts()
        >>> narrative_prompts = list_prompts(PromptCategory.NARRATIVE_GENERATION)
    """
    templates = list(PROMPT_REGISTRY.values())

    if category is not None:
        templates = [t for t in templates if t.category == category]

    return sorted(templates, key=lambda t: t.id)


# Register all built-in prompts
def _register_builtin_prompts() -> None:
    """Register all built-in prompt templates."""
    for template in [
        CHAPTER_DETECTION_PROMPT,
        NARRATIVE_GENERATION_PROMPT,
        EXECUTIVE_SUMMARY_PROMPT,
        PLATFORM_ANALYSIS_PROMPT,
        GAP_ANALYSIS_PROMPT,
        PATTERN_DETECTION_PROMPT,
        CHAPTER_REFINEMENT_PROMPT,
    ]:
        register_prompt(template)


_register_builtin_prompts()


# =============================================================================
# Helper Functions
# =============================================================================


def prepare_memories_for_prompt(
    memories: list["Memory"],
    max_items: int = 100,
    privacy_level: str = "standard",
) -> str:
    """Convert memories to prompt-safe JSON string.

    Samples memories if there are too many, applies privacy filtering,
    and formats as compact JSON ready for prompt insertion.

    Args:
        memories: List of Memory objects to prepare.
        max_items: Maximum number of memories to include.
        privacy_level: Privacy level ("strict", "standard", "detailed").
                       Controls what fields are included.

    Returns:
        JSON string of prepared memories.

    Example:
        >>> memories_json = prepare_memories_for_prompt(memories, max_items=50)
        >>> system, user = template.render(sample_memories=memories_json, ...)
    """
    if not memories:
        return "[]"

    # Sample if too many
    if len(memories) > max_items:
        # Stratified sampling: take evenly across timeline
        step = len(memories) // max_items
        sampled = memories[::step][:max_items]
    else:
        sampled = memories

    # Convert to prompt-safe dictionaries
    result = []
    for memory in sampled:
        item: dict[str, Any] = {
            "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
            "media_type": memory.media_type.value if memory.media_type else None,
            "platform": memory.source_platform.value if memory.source_platform else None,
        }

        # Privacy filtering
        if privacy_level != "strict":
            if memory.location:
                # Only include city/country level
                item["location"] = memory.location.to_ai_summary()

            if privacy_level == "detailed" and memory.caption:
                # Truncate caption for privacy
                item["caption"] = (
                    memory.caption[:100] + "..." if len(memory.caption) > 100 else memory.caption
                )

            if memory.people:
                # Just count, don't include names
                item["people_count"] = len(memory.people)

        result.append(item)

    return json.dumps(result, indent=2, default=str)


def prepare_timeline_summary(memories: list["Memory"]) -> str:
    """Generate statistical summary for prompt context.

    Creates a text summary of the timeline including yearly distribution,
    platform breakdown, location patterns, and media types.

    Args:
        memories: List of Memory objects to summarize.

    Returns:
        Multi-line string summary suitable for prompt insertion.

    Example:
        >>> summary = prepare_timeline_summary(memories)
        >>> print(summary)
        ### Yearly Distribution
        2020: 500 memories
        2021: 750 memories
        ...
    """
    if not memories:
        return "No memories in timeline."

    lines = []

    # Yearly distribution
    years: Counter[int] = Counter()
    for m in memories:
        if m.timestamp:
            years[m.timestamp.year] += 1

    if years:
        lines.append("### Yearly Distribution")
        for year in sorted(years.keys()):
            lines.append(f"- {year}: {years[year]} memories")

    # Platform distribution
    platforms: Counter[str] = Counter()
    for m in memories:
        if m.source_platform:
            platforms[m.source_platform.value] += 1

    if platforms:
        lines.append("\n### Platform Distribution")
        for platform, count in platforms.most_common():
            lines.append(f"- {platform}: {count} memories")

    # Media type distribution
    media_types: Counter[str] = Counter()
    for m in memories:
        if m.media_type:
            media_types[m.media_type.value] += 1

    if media_types:
        lines.append("\n### Media Types")
        for mt, count in media_types.most_common():
            lines.append(f"- {mt}: {count}")

    # Location summary
    countries: Counter[str] = Counter()
    for m in memories:
        if m.location and m.location.country:
            countries[m.location.country] += 1

    if countries:
        lines.append("\n### Locations (by country)")
        for country, count in countries.most_common(10):
            lines.append(f"- {country}: {count} memories")

    return "\n".join(lines)


def prepare_chapters_for_prompt(chapters: list[Any]) -> str:
    """Format existing chapters for context in prompts.

    Converts chapter objects to a readable summary format for use
    in refinement or summary prompts.

    Args:
        chapters: List of chapter dictionaries or objects with title,
                  start_date, end_date, and themes attributes.

    Returns:
        Formatted string summary of chapters.

    Example:
        >>> chapters_str = prepare_chapters_for_prompt(detected_chapters)
    """
    if not chapters:
        return "No chapters detected yet."

    lines = []
    for i, chapter in enumerate(chapters, 1):
        if isinstance(chapter, dict):
            title = chapter.get("title", f"Chapter {i}")
            start = chapter.get("start_date", "unknown")
            end = chapter.get("end_date", "unknown")
            themes = chapter.get("themes", [])
        else:
            # Assume object with attributes
            title = getattr(chapter, "title", f"Chapter {i}")
            start = getattr(chapter, "start_date", "unknown")
            end = getattr(chapter, "end_date", "unknown")
            themes = getattr(chapter, "themes", [])

        themes_str = ", ".join(themes) if themes else "no themes"
        lines.append(f"{i}. **{title}** ({start} to {end})")
        lines.append(f"   Themes: {themes_str}")

    return "\n".join(lines)


def render_output_schema(schema: dict[str, Any]) -> str:
    """Convert schema dict to pretty JSON string for prompt.

    Args:
        schema: Dictionary representing the expected JSON schema.

    Returns:
        Formatted JSON string suitable for prompt insertion.

    Example:
        >>> schema_str = render_output_schema({"name": "string", "count": "integer"})
    """
    return json.dumps(schema, indent=2)


def prepare_platform_breakdown(memories: list["Memory"]) -> str:
    """Generate platform breakdown string for prompts.

    Args:
        memories: List of Memory objects.

    Returns:
        Comma-separated platform counts (e.g., "Snapchat: 500, Google Photos: 2000").
    """
    platforms: Counter[str] = Counter()
    for m in memories:
        if m.source_platform:
            platforms[m.source_platform.value] += 1

    if not platforms:
        return "No platform information available"

    return ", ".join(f"{p}: {c}" for p, c in platforms.most_common())


def prepare_date_range(memories: list["Memory"]) -> str:
    """Generate date range string for prompts.

    Args:
        memories: List of Memory objects.

    Returns:
        Date range string (e.g., "2015-01-01 to 2023-12-31").
    """
    dates = [m.timestamp for m in memories if m.timestamp]

    if not dates:
        return "Unknown date range"

    min_date = min(dates)
    max_date = max(dates)

    return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"


def build_prompt_context(memories: list["Memory"], **kwargs: Any) -> PromptContext:
    """Build a complete PromptContext from memories.

    Convenience function that calls all prepare_* functions to build
    a complete context object.

    Args:
        memories: List of Memory objects.
        **kwargs: Additional context to include.

    Returns:
        Populated PromptContext object.

    Example:
        >>> context = build_prompt_context(memories)
        >>> system, user = template.render(**context.to_dict())
    """
    return PromptContext(
        timeline_summary=prepare_timeline_summary(memories),
        sample_memories=prepare_memories_for_prompt(memories),
        date_range=prepare_date_range(memories),
        total_memories=len(memories),
        platform_breakdown=prepare_platform_breakdown(memories),
        custom_context=kwargs if kwargs else None,
    )
