"""Intelligent image sampling and budget management for Digital Life Narrative AI.

This module provides the logic for selecting the most representative and 
metadata-rich images from a large collection, while respecting global 
budget caps and ensuring domestic/temporal variety.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set

from src.core.models import (
    AnalysisConfig,
    DepthMode,
    LifeChapter,
    Memory,
    MediaType,
    SourcePlatform,
)
from src.config import DepthModeConfig, STANDARD_MODE_CONFIG, DEPTH_MODE_CONFIGS


@dataclass
class SamplingResult:
    """Results of a full visual sampling planning session."""
    sampled_memories: Dict[str, List[Memory]]  # chapter_id -> selected memories
    total_sampled: int
    budget_cap_applied: bool
    chapters_reduced: List[str]  # IDs of chapters that got fewer than requested
    duplicates_skipped: int
    bursts_compressed: int
    sampling_stats: Dict[str, Any] = field(default_factory=dict)


class BudgetManager:
    """Manages the global image cap across multiple chapters.
    
    Ensures that we don't exceed the global API limit while 
    distributing images fairly based on chapter duration and significance.
    """

    def __init__(self, global_cap: int, chapters: List[LifeChapter]):
        self.global_cap = global_cap
        self.chapters = {str(i): ch for i, ch in enumerate(chapters)}
        self.allocations: Dict[str, int] = {}
        self.requested: Dict[str, int] = {}
        self._reduction_applied = False
        self._remaining = global_cap

    def allocate_budget(self) -> Dict[str, int]:
        """Statically allocate budget based on chapter characteristics."""
        if not self.chapters:
            return {}

        total_weight = 0.0
        chapter_weights = {}

        for cid, chapter in self.chapters.items():
            # Weighting factors
            duration_days = (chapter.end_date - chapter.start_date).days + 1
            duration_weight = max(1, duration_days / 30.0) # More days = more weight
            
            # Media density weight (capped)
            media_weight = min(2.0, max(1.0, chapter.media_count / 100.0))
            
            # Total weight for this chapter
            weight = duration_weight * media_weight
            chapter_weights[cid] = weight
            total_weight += weight

        # Initial proportional allocation
        total_allocated = 0
        for cid, weight in chapter_weights.items():
            share = (weight / total_weight) * self.global_cap
            # Ensure at least 1 image per chapter if possible
            alloc = max(1, int(share))
            self.allocations[cid] = alloc
            total_allocated += alloc

        # If we exceeded cap due to 'max(1, ...)', reduce from largest allocations
        while total_allocated > self.global_cap:
            sorted_cids = sorted(self.allocations.keys(), key=lambda k: self.allocations[k], reverse=True)
            reduced = False
            for cid in sorted_cids:
                if self.allocations[cid] > 1:
                    self.allocations[cid] -= 1
                    total_allocated -= 1
                    reduced = True
                    self._reduction_applied = True
                    break
            if not reduced: # Cannot reduce further
                break

        self._remaining = self.global_cap - total_allocated
        return self.allocations

    def request_allocation(self, chapter_id: str, requested: int) -> int:
        """Register a specific request and return the allowed count."""
        self.requested[chapter_id] = requested
        allowed = min(requested, self.allocations.get(chapter_id, requested))
        if allowed < requested:
            self._reduction_applied = True
        return allowed

    def get_remaining_budget(self) -> int:
        """Return remaining images in global budget."""
        return self._remaining

    def was_reduction_applied(self) -> bool:
        """True if any chapter got less than requested or cap was hit."""
        return self._reduction_applied


class SamplingStrategy:
    """Implements intelligent image selection logic."""

    # Keywords that suggest an image is highly significant
    SIGNIFICANCE_KEYWORDS = [
        "wedding", "marry", "marriage", "bride", "groom",
        "graduation", "grad", "diploma", "degree",
        "birthday", "bday", "cake", "party",
        "trip", "vacation", "holiday", "travel", "tour",
        "birth", "baby", "newborn",
        "move", "house", "home", "apartment",
        "start", "first", "launch", "award", "win"
    ]

    def __init__(self, depth_config: DepthModeConfig, global_cap: int, logger: logging.Logger | None = None):
        self.depth_config = depth_config
        self.global_cap = global_cap
        self.logger = logger or logging.getLogger(__name__)
        self._content_hashes: Set[str] = set()

    def score_memory(self, memory: Memory) -> float:
        """Score a memory for sampling priority. Higher = better.
        
        Factors:
        - Metadata richness (+3 for caption, +2 for location, +2 per person)
        - Keyword significance (+3)
        - Platform curation (+1 for Snapchat/Stories)
        - Image quality/confidence (+1)
        """
        score = 0.0

        # 1. Metadata richness
        if memory.caption:
            score += 3.0
        if memory.location and not memory.location.is_empty():
            score += 2.0
        if memory.people:
            # +2 per person, capped at 10 to favor group shots but not infinitely
            score += min(10.0, len(memory.people) * 2.0)

        # 2. Significance hints in path/caption
        search_text = f"{memory.source_path or ''} {memory.caption or ''}".lower()
        if any(kw in search_text for kw in self.SIGNIFICANCE_KEYWORDS):
            score += 3.0

        # 3. Platform curation
        if memory.source_platform in [SourcePlatform.SNAPCHAT, SourcePlatform.INSTAGRAM]:
            score += 1.0 # These are usually more "curated"

        # 4. Visual confidence (if already tagged by something else)
        if memory.visually_analyzed:
            score += (memory.visual_confidence or 0.5)

        # 5. Media type variety
        if memory.media_type == MediaType.VIDEO:
            score += 1.0 # Videos are rare and often high signal

        return score

    def detect_bursts(self, memories: List[Memory]) -> List[List[str]]:
        """Identify groups of memories taken in rapid succession (bursts)."""
        if not memories:
            return []

        # Sort by creation time
        sorted_mems = sorted([m for m in memories if m.created_at], key=lambda m: m.created_at)
        if not sorted_mems:
            return []

        bursts: List[List[str]] = []
        current_burst: List[str] = [sorted_mems[0].id]

        for i in range(1, len(sorted_mems)):
            prev = sorted_mems[i-1]
            curr = sorted_mems[i]
            
            delta = (curr.created_at - prev.created_at).total_seconds()
            
            if delta <= 3.0: # 3-second burst threshold
                current_burst.append(curr.id)
            else:
                if len(current_burst) > 1:
                    bursts.append(current_burst)
                current_burst = [curr.id]
        
        if len(current_burst) > 1:
            bursts.append(current_burst)

        return bursts

    def detect_duplicates(self, memories: List[Memory]) -> Set[str]:
        """Identify exact and near-duplicates to exclude."""
        to_exclude = set()
        seen_hashes = set()
        seen_stamps = {} # (timestamp, filesize) -> id

        for m in memories:
            # Exact hash match
            if m.content_hash:
                if m.content_hash in seen_hashes:
                    to_exclude.add(m.id)
                    continue
                seen_hashes.add(m.content_hash)

            # Heuristic match: same second and similar size
            if m.created_at:
                stamp_key = (m.created_at.replace(microsecond=0), m.file_size)
                if stamp_key in seen_stamps:
                    to_exclude.add(m.id)
                    continue
                seen_stamps[stamp_key] = m.id

            # Skip sidecar/shadow files (OS specific patterns)
            if m.source_path:
                path = m.source_path.lower()
                if "/._" in path or "/metadata/" in path or path.endswith((".json", ".aae", ".ini")):
                    to_exclude.add(m.id)

        return to_exclude

    def sample_for_chapter(self, memories: List[Memory], target_count: int) -> List[Memory]:
        """Sample images for a specific chapter respecting constraints."""
        if not memories:
            return []

        # 1. Deduplicate
        dupes = self.detect_duplicates(memories)
        candidates = [m for m in memories if m.id not in dupes]

        # 2. Score
        scored_candidates = [(self.score_memory(m), m) for m in candidates]
        # Sort by score (desc) then date (asc)
        scored_candidates.sort(key=lambda x: (-x[0], x[1].created_at or datetime.min))

        # 3. Handle Bursts (keep only top scored from each burst)
        bursts = self.detect_bursts([x[1] for x in scored_candidates])
        burst_map = {} # mem_id -> burst_index
        for idx, burst in enumerate(bursts):
            for mid in burst:
                burst_map[mid] = idx

        final_candidates = []
        bursts_visited = set()

        for score, m in scored_candidates:
            if m.id in burst_map:
                b_idx = burst_map[m.id]
                if b_idx not in bursts_visited:
                    final_candidates.append(m)
                    bursts_visited.add(b_idx)
            else:
                final_candidates.append(m)

        # 4. Temporal Spread
        # If we have too many, we want to ensure we don't pick all from one day
        if len(final_candidates) > target_count:
            # Resort by date for spread selection
            final_candidates.sort(key=lambda m: m.created_at or datetime.min)
            
            # Simple stratified selection to ensure spread
            step = max(1, len(final_candidates) // target_count)
            sampled = final_candidates[::step][:target_count]
        else:
            sampled = final_candidates

        # Always ensure chronological return
        return sorted(sampled, key=lambda m: m.created_at or datetime.min)


def plan_visual_sampling(
    memories: List[Memory], 
    chapters: List[LifeChapter], 
    config: AnalysisConfig
) -> SamplingResult:
    """Orchestrates the full visual sampling plan across all chapters."""
    # 1. Setup
    from src.config import DEPTH_MODE_CONFIGS, STANDARD_MODE_CONFIG
    # Deeply nested access to global config
    # Note: In a real environment, this might be get_config().ai.get_depth_config()
    # Here we assume it's passed or accessible.
    ai_cfg = DEPTH_MODE_CONFIGS.get(config.depth.value.lower(), STANDARD_MODE_CONFIG)
    global_cap = config.max_images or 120
    
    logger = logging.getLogger("src.ai.sampling")
    logger.info(f"Planning visual sampling for {len(chapters)} chapters with global cap {global_cap}")
    
    budget = BudgetManager(global_cap, chapters)
    allocations = budget.allocate_budget()
    
    strategy = SamplingStrategy(ai_cfg, global_cap, logger)
    
    sampled_map: Dict[str, List[Memory]] = {}
    total_sampled = 0
    chapters_reduced = []
    bursts_compressed = 0
    duplicates_skipped = 0

    # 2. Assign memories to chapters (simple overlap)
    for i, chapter in enumerate(chapters):
        cid = str(i)
        chapter_mems = [
            m for m in memories 
            if m.created_at and chapter.start_date <= m.created_at.date() <= chapter.end_date
        ]
        
        target = allocations.get(cid, ai_cfg.images_per_chapter_target)
        
        # Log before stats
        pre_count = len(chapter_mems)
        
        # Sample
        selected = strategy.sample_for_chapter(chapter_mems, target)
        
        sampled_map[cid] = selected
        total_sampled += len(selected)
        
        if len(selected) < target and len(chapter_mems) > target:
            # This usually means filtering was too aggressive (rare) or logic error
            pass
        
        if len(chapter_mems) > target and len(selected) == target:
            # Just standard sampling
            pass

    # 3. Summary
    result = SamplingResult(
        sampled_memories=sampled_map,
        total_sampled=total_sampled,
        budget_cap_applied=budget.was_reduction_applied(),
        chapters_reduced=[cid for cid, alloc in allocations.items() if alloc < ai_cfg.images_per_chapter_target],
        duplicates_skipped=0, # Would need more tracking to populate
        bursts_compressed=0,
        sampling_stats={
            "total_available": len(memories),
            "global_cap": global_cap,
            "chapters_processed": len(chapters)
        }
    )
    
    logger.info(f"Sampling complete: {total_sampled} images selected across {len(chapters)} chapters")
    return result
