"""Legacy Life Story Analyzer Proxy.

This module is DEPRECATED and remains for backward compatibility only.
It proxies all core functionality to the new consolidated Narrative Engine 
in `dlnai.ai.analyzer`.

WARNING: Please migrate your imports to `dlnai.ai` as this file will be 
removed in v1.1.

NEW: Two-Stage Multimodal Flow
-------------------------------
The analyzer now supports intelligent visual analysis:
- Stage 1: Visual Tagging (Gemini Flash) - extracts scenes, vibes, motifs
- Stage 2: Narrative Synthesis (Gemini Pro) - generates grounded stories

See MULTIMODAL_ARCHITECTURE.md for details.
"""

import warnings
import logging
from typing import Any, Callable

# Trigger a loud deprecation warning on import
warnings.warn(
    "dlnai.ai.life_analyzer is deprecated and will be removed in v1.1. "
    "Please migrate to dlnai.ai (LifeStoryAnalyzer) or dlnai.ai.analyzer.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)
logger.warning("DEPRECATION: dlnai.ai.life_analyzer is being used. Switch to dlnai.ai.")

# Re-export from the new consolidated location
from dlnai.ai.analyzer import (
    LifeStoryAnalyzer,
    AnalysisProgress,
    AnalysisError,
    InsufficientDataError,
)

# Re-export data models (proxied via src.ai or directly from core)
from dlnai.core import (
    AnalysisConfig,
    LifeStoryReport,
    LifeChapter,
    Memory,
    DataGap,
    PlatformBehaviorInsight,
    SourcePlatform,
    MediaType,
    ConfidenceLevel,
    DepthMode,
)

# Maintain existing top-level functions for compatibility
def analyze_memories(
    memories: list[Memory],
    config: AnalysisConfig | None = None,
    progress_callback: Callable[[Any], None] | None = None,
) -> LifeStoryReport:
    """Legacy entry point. Proxies to LifeStoryAnalyzer.analyze.
    
    The analyzer now runs a two-stage multimodal flow:
    1. Visual Intelligence (Stage 1): Samples and tags images
    2. Narrative Synthesis (Stage 2): Generates grounded stories
    """
    analyzer = LifeStoryAnalyzer()
    return analyzer.analyze(memories, config=config, progress_callback=progress_callback)

def quick_analyze(memories: list[Memory]) -> LifeStoryReport:
    """Legacy entry point for rapid analysis.
    
    Quick mode minimizes visual analysis to reduce costs.
    """
    config = AnalysisConfig(depth=DepthMode.QUICK)
    return analyze_memories(memories, config=config)


# Convenience methods for two-stage flow configuration
def analyze_with_visual_depth(
    memories: list[Memory],
    depth: str | DepthMode = "standard",
    max_images: int | None = None,
    vision_model: str | None = None,
    progress_callback: Callable[[Any], None] | None = None,
) -> LifeStoryReport:
    """Analyze memories with explicit visual depth control.
    
    Args:
        memories: List of Memory objects to analyze.
        depth: Analysis depth ("quick", "standard", or "deep").
        max_images: Maximum images to analyze (overrides depth default).
        vision_model: Vision model override (e.g., "gemini-2.0-flash-exp").
        progress_callback: Optional progress reporting callback.
        
    Returns:
        Complete LifeStoryReport with visual grounding.
        
    Example:
        >>> report = analyze_with_visual_depth(
        ...     memories, 
        ...     depth="deep",
        ...     max_images=200
        ... )
    """
    if isinstance(depth, str):
        depth = DepthMode(depth.upper())
    
    config = AnalysisConfig(
        depth=depth,
        max_images=max_images,
        vision_model=vision_model or "gemini-2.0-flash-exp"
    )
    return analyze_memories(memories, config=config, progress_callback=progress_callback)


def estimate_visual_cost(
    memories: list[Memory],
    depth: str | DepthMode = "standard",
    max_images: int | None = None,
) -> dict[str, Any]:
    """Estimate the cost of visual analysis without running it.
    
    Args:
        memories: List of Memory objects.
        depth: Planned analysis depth.
        max_images: Maximum images to analyze.
        
    Returns:
        Dict with cost estimates:
        - estimated_images: Number of images to analyze
        - estimated_vision_cost_usd: Vision API cost
        - estimated_narrative_cost_usd: Narrative API cost
        - total_estimated_cost_usd: Total cost
        
    Example:
        >>> cost = estimate_visual_cost(memories, depth="deep")
        >>> print(f"Analysis will cost ~${cost['total_estimated_cost_usd']:.2f}")
    """
    from dlnai.config import get_config, DEPTH_MODE_CONFIGS, STANDARD_MODE_CONFIG
    
    if isinstance(depth, str):
        depth = DepthMode(depth.upper())
    
    # Get depth config
    depth_cfg = DEPTH_MODE_CONFIGS.get(depth.value.lower(), STANDARD_MODE_CONFIG)
    ai_cfg = get_config().ai
    
    # Count visual candidates
    visual_memories = [
        m for m in memories 
        if m.media_type in ["photo", "video"] and m.source_path
    ]
    
    # Apply sampling estimation
    if max_images:
        estimated_images = min(len(visual_memories), max_images)
    else:
        # Use depth-based fraction
        depth_fractions = {
            DepthMode.QUICK: 0.05,
            DepthMode.STANDARD: 0.15,
            DepthMode.DEEP: 0.40
        }
        fraction = depth_fractions.get(depth, 0.15)
        estimated_images = max(12, int(len(visual_memories) * fraction)) # Increased base min to 12
        estimated_images = min(estimated_images, ai_cfg.max_vision_images_per_run)
    
    # Cost calculation
    vision_cost = estimated_images * ai_cfg.vision_cost_per_image_usd
    
    # Narrative cost (rough estimate: 1k tokens per chapter + overhead)
    estimated_chapters = max(3, len(memories) // 200)
    estimated_narrative_tokens = (estimated_chapters + 4) * 1200
    narrative_cost = (estimated_narrative_tokens / 1000.0) * ai_cfg.narrative_cost_per_1k_tokens_usd
    
    return {
        "estimated_images": estimated_images,
        "total_visual_candidates": len(visual_memories),
        "estimated_vision_cost_usd": vision_cost,
        "estimated_narrative_cost_usd": narrative_cost,
        "total_estimated_cost_usd": vision_cost + narrative_cost,
        "depth_mode": depth.value,
    }


def analyze_without_vision(
    memories: list[Memory],
    progress_callback: Callable[[Any], None] | None = None,
) -> LifeStoryReport:
    """Analyze memories without visual intelligence (metadata only).
    
    This skips Stage 1 visual tagging and relies purely on metadata
    for narrative generation. Useful for privacy-sensitive scenarios.
    
    Args:
        memories: List of Memory objects to analyze.
        progress_callback: Optional progress reporting callback.
        
    Returns:
        LifeStoryReport without visual grounding.
    """
    config = AnalysisConfig(
        depth=DepthMode.QUICK,
        max_images=0  # Disable visual analysis
    )
    return analyze_memories(memories, config=config, progress_callback=progress_callback)


# Map everything else that might be expected
__all__ = [
    "LifeStoryAnalyzer",
    "AnalysisConfig",
    "AnalysisProgress",
    "LifeStoryReport",
    "LifeChapter",
    "Memory",
    "DepthMode",
    "analyze_memories",
    "quick_analyze",
    "analyze_with_visual_depth",
    "estimate_visual_cost",
    "analyze_without_vision",
]
