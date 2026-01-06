"""Parsing pipeline orchestrator for Digital Life Narrative AI.

This module provides the high-level parsing orchestrator that ties together
detection, parser selection, and execution. It's the main entry point for
the ingestion layer: give it paths, get back unified Memory objects ready
for AI analysis.

The pipeline:
1. Accepts one or more input paths
2. Detects which platforms are present
3. Runs appropriate parsers
4. Merges and deduplicates results
5. Returns a unified list of Memory objects

Typical usage:
    >>> from pathlib import Path
    >>> from src.parsers.pipeline import run_pipeline, PipelineConfig
    >>> 
    >>> config = PipelineConfig(recursive_detection=True)
    >>> result = run_pipeline([Path("/exports")], config)
    >>> print(f"Parsed {result.total_memories} memories from {len(result.detections)} sources")
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.memory import ConfidenceLevel, Memory, SourcePlatform
from src.detection import (
    DetectionResult,
    detect_sources,
   detect_sources_recursive,
    summarize_detections,
)
from src.parsers.base import BaseParser, ParseResult, ParseStatus, ProgressCallback

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the parsing pipeline.
    
    Attributes:
        recursive_detection: Scan subdirectories for exports.
        max_detection_depth: Maximum depth for recursive detection.
        parallel_parsing: Run parsers in parallel (experimental).
        max_workers: Number of parallel workers if parallel_parsing enabled.
        deduplicate_results: Remove duplicate memories.
        dedup_tolerance_seconds: Tolerance for same-moment detection.
        include_platforms: Only parse these platforms (None = all).
        exclude_platforms: Skip these platforms (None = none).
        min_confidence: Skip detections below this confidence.
        fail_fast: Stop on first parser error.
        progress_granularity: Report progress every N files.
    """
    recursive_detection: bool = True
    max_detection_depth: int = 2
    parallel_parsing: bool = False
    max_workers: int = 4
    deduplicate_results: bool = True
    dedup_tolerance_seconds: int = 60
    include_platforms: Optional[Set[SourcePlatform]] = None
    exclude_platforms: Optional[Set[SourcePlatform]] = None
    min_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    fail_fast: bool = False
    progress_granularity: int = 10


@dataclass
class PipelineProgress:
    """Detailed progress information during pipeline execution.
    
    Attributes:
        stage: Current stage (detection, parsing, deduplication, etc.).
        current_platform: Platform being processed (None if N/A).
        files_processed: Files processed so far.
        files_total: Total files to process.
        memories_found: Memories extracted so far.
        errors_count: Errors encountered.
        elapsed_seconds: Time elapsed since pipeline start.
    """
    stage: str
    current_platform: Optional[SourcePlatform] = None
    files_processed: int = 0
    files_total: int = 0
    memories_found: int = 0
    errors_count: int = 0
    elapsed_seconds: float = 0.0
    
    def percentage(self) -> float:
        """Calculate completion percentage.
        
        Returns:
            Percentage (0-100) of files processed.
        """
        if self.files_total == 0:
            return 0.0
        return (self.files_processed / self.files_total) * 100.0
    
    def to_status_line(self) -> str:
        """Generate human-readable status line.
        
        Returns:
            Status string like "Parsing SNAPCHAT: 150/500 files (30%)".
        """
        if self.stage == "detection":
            return f"Detecting sources... ({self.elapsed_seconds:.1f}s)"
        elif self.stage == "parsing" and self.current_platform:
            platform_name = self.current_platform.value.upper()
            pct = self.percentage()
            return f"Parsing {platform_name}: {self.files_processed}/{self.files_total} files ({pct:.0f}%)"
        elif self.stage == "deduplication":
            return f"Deduplicating {self.memories_found} memories... ({self.elapsed_seconds:.1f}s)"
        elif self.stage == "aggregation":
            return f"Aggregating results... ({self.memories_found} memories)"
        else:
            return f"{self.stage}: {self.files_processed}/{self.files_total}"


@dataclass
class PipelineResult:
    """Complete result of running the parsing pipeline.
    
    Attributes:
        memories: Extracted Memory objects (main output).
        detections: Detection results for each source.
        parse_results: Per-parser results.
        total_files_processed: Total files processed.
        total_memories: Total memories extracted.
        duplicates_removed: Count of duplicates removed.
        total_warnings: Total warnings across all parsers.
        total_errors: Total errors across all parsers.
        elapsed_seconds: Total time elapsed.
        input_paths: Input paths provided to pipeline.
    """
    memories: List[Memory] = field(default_factory=list)
    detections: List[DetectionResult] = field(default_factory=list)
    parse_results: List[ParseResult] = field(default_factory=list)
    total_files_processed: int = 0
    total_memories: int = 0
    duplicates_removed: int = 0
    total_warnings: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
    input_paths: List[Path] = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate success rate.
        
        Returns:
            Ratio of memories to (memories + errors).
        """
        total = self.total_memories + self.total_errors
        if total == 0:
            return 0.0
        return self.total_memories / total
    
    def to_summary(self) -> str:
        """Generate human-readable summary.
        
        Returns:
            Multi-line summary string.
        """
        lines = [
            f"Pipeline Results:",
            f"  Memories: {self.total_memories}",
            f"  Files processed: {self.total_files_processed}",
            f"  Duplicates removed: {self.duplicates_removed}",
            f"  Warnings: {self.total_warnings}",
            f"  Errors: {self.total_errors}",
            f"  Success rate: {self.success_rate():.1%}",
            f"  Time: {self.elapsed_seconds:.1f}s",
            f"  Sources detected: {len(self.detections)}"
        ]
        return "\n".join(lines)
    
    def get_memories_by_platform(self, platform: SourcePlatform) -> List[Memory]:
        """Get memories from a specific platform.
        
        Args:
            platform: Platform to filter by.
            
        Returns:
            List of memories from that platform.
        """
        return [m for m in self.memories if m.source_platform == platform]
    
    def has_critical_failures(self) -> bool:
        """Check if any critical failures occurred.
        
        Returns:
            True if any parser completely failed.
        """
        for result in self.parse_results:
            if result.status == ParseStatus.FAILED:
                return True
        return False


class PipelineError(Exception):
    """Exception for pipeline-level failures.
    
    Attributes:
        message: Error message.
        stage: Pipeline stage where error occurred.
        partial_result: Partial results before failure (if any).
    """
    
    def __init__(self, message: str, stage: str, partial_result: Optional[PipelineResult] = None):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.partial_result = partial_result


def run_pipeline(
    paths: List[Path],
    config: Optional[PipelineConfig] = None,
    progress: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """Execute the full parsing pipeline.
    
    This is the main entry point for parsing media exports. It coordinates
    detection, parser selection, parsing, and deduplication.
    
    Args:
        paths: List of paths to scan and parse.
        config: Pipeline configuration (uses defaults if None).
        progress: Optional progress callback.
        
    Returns:
        PipelineResult with all extracted memories and metadata.
        
    Raises:
        PipelineError: If fail_fast enabled and critical error occurs.
        
    Workflow:
        1. Validate input paths
        2. Detect sources (platforms present)
        3. Select appropriate parsers
        4. Parse each source
        5. Aggregate results
        6. Deduplicate memories
        7. Sort and finalize
    """
    start_time = time.time()
    config = config or PipelineConfig()
    
    result = PipelineResult(input_paths=list(paths))
    
    def report_progress(stage: str, **kwargs):
        """Helper to report progress."""
        if progress:
            elapsed = time.time() - start_time
            prog = PipelineProgress(stage=stage, elapsed_seconds=elapsed, **kwargs)
            progress(prog)
    
    # Stage 1: Validation
    logger.info(f"Starting pipeline with {len(paths)} input paths")
    valid_paths = []
    for path in paths:
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            result.total_errors += 1
        elif not path.is_dir():
            logger.warning(f"Path is not a directory: {path}")
            result.total_errors += 1
        else:
            valid_paths.append(path)
    
    if not valid_paths:
        logger.error("No valid paths to process")
        result.elapsed_seconds = time.time() - start_time
        if config.fail_fast:
            raise PipelineError("No valid paths", "validation", result)
        return result
    
    # Remove duplicate paths
    valid_paths = list(set(valid_paths))
    
    # Stage 2: Detection
    report_progress("detection")
    logger.info(f"Detecting sources in {len(valid_paths)} paths")
    
    all_detections: List[DetectionResult] = []
    for path in valid_paths:
        try:
            if config.recursive_detection:
                detections = detect_sources_recursive(path, config.max_detection_depth)
            else:
                detections = detect_sources(path)
            
            # Apply filters
            filtered_detections = []
            for detection in detections:
                # Platform filters
                if config.include_platforms and detection.platform not in config.include_platforms:
                    continue
                if config.exclude_platforms and detection.platform in config.exclude_platforms:
                    continue
                # Confidence filter
                if detection.confidence.value < config.min_confidence.value:
                    continue
                filtered_detections.append(detection)
            
            all_detections.extend(filtered_detections)
            logger.info(f"Detected {len(filtered_detections)} sources in {path}")
            
        except Exception as e:
            logger.error(f"Error detecting sources in {path}: {e}", exc_info=True)
            result.total_errors += 1
    
    result.detections = all_detections
    
    if not all_detections:
        logger.warning("No sources detected")
        result.elapsed_seconds = time.time() - start_time
        return result
    
    # Log detection summary
    for line in summarize_detections(all_detections):
        logger.info(line)
    
    # Stage 3: Parser Selection
    from src.parsers import ParserRegistry
    registry = ParserRegistry()
    
    parse_tasks: List[Tuple[BaseParser, Path]] = []
    for detection in all_detections:
        try:
            parser = registry.get_parser(detection.platform)
            if parser:
                parse_tasks.append((parser, detection.root_path))
            else:
                logger.warning(f"No parser available for {detection.platform.value}")
                result.total_warnings += 1
        except Exception as e:
            logger.error(f"Error getting parser for {detection.platform.value}: {e}")
            result.total_errors += 1
    
    if not parse_tasks:
        logger.warning("No parsers available for detected sources")
        result.elapsed_seconds = time.time() - start_time
        return result
    
    # Deduplicate tasks (same parser + path)
    unique_tasks = list(set(parse_tasks))
    logger.info(f"Prepared {len(unique_tasks)} parsing tasks")
    
    # Stage 4: Parsing
    all_memories: List[Memory] = []
    
    def parse_with_tracking(parser: BaseParser, path: Path) -> ParseResult:
        """Parse with progress tracking."""
        platform = parser.platform
        
        def parser_progress(prog):
            """Wrap parser progress into pipeline progress."""
            report_progress(
                "parsing",
                current_platform=platform,
                files_processed=prog.current,
                files_total=prog.total,
                memories_found=len(all_memories)
            )
        
        logger.info(f"Parsing {platform.value} at {path}")
        return parser.parse(path, parser_progress)
    
    if config.parallel_parsing and len(unique_tasks) > 1:
        # Parallel execution
        logger.info(f"Parsing {len(unique_tasks)} sources in parallel (max_workers={config.max_workers})")
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(parse_with_tracking, parser, path): (parser, path)
                for parser, path in unique_tasks
            }
            
            for future in as_completed(futures):
                parser, path = futures[future]
                try:
                    parse_result = future.result()
                    result.parse_results.append(parse_result)
                    all_memories.extend(parse_result.memories)
                    result.total_warnings += len(parse_result.warnings)
                    result.total_errors += len(parse_result.errors)
                    
                    if parse_result.status == ParseStatus.FAILED and config.fail_fast:
                        raise PipelineError(
                            f"Parser failed for {parser.platform.value}",
                            "parsing",
                            result
                        )
                    
                except Exception as e:
                    logger.error(f"Parser {parser.platform.value} failed: {e}", exc_info=True)
                    result.total_errors += 1
                    if config.fail_fast:
                        raise PipelineError(str(e), "parsing", result)
    else:
        # Sequential execution
        for parser, path in unique_tasks:
            try:
                parse_result = parse_with_tracking(parser, path)
                result.parse_results.append(parse_result)
                all_memories.extend(parse_result.memories)
                result.total_warnings += len(parse_result.warnings)
                result.total_errors += len(parse_result.errors)
                
                if parse_result.status == ParseStatus.FAILED and config.fail_fast:
                    raise PipelineError(
                        f"Parser failed for {parser.platform.value}",
                        "parsing",
                        result
                    )
                    
            except Exception as e:
                logger.error(f"Parser {parser.platform.value} failed: {e}", exc_info=True)
                result.total_errors += 1
                if config.fail_fast:
                    raise PipelineError(str(e), "parsing", result)
    
    # Stage 5: Aggregation
    report_progress("aggregation", memories_found=len(all_memories))
    logger.info(f"Aggregated {len(all_memories)} memories from {len(result.parse_results)} parsers")
    
    # Stage 6: Deduplication
    if config.deduplicate_results and all_memories:
        report_progress("deduplication", memories_found=len(all_memories))
        deduplicated, removed_count = deduplicate_memories(
            all_memories,
            config.dedup_tolerance_seconds
        )
        result.memories = deduplicated
        result.duplicates_removed = removed_count
        logger.info(f"Removed {removed_count} duplicates, {len(deduplicated)} unique memories remain")
    else:
        result.memories = all_memories
    
    # Stage 7: Finalization
    # Sort by created_at (None dates last)
    result.memories.sort(key=lambda m: (m.created_at is None, m.created_at or 0))
    
    result.total_memories = len(result.memories)
    result.total_files_processed = sum(
        r.statistics.get("total_files_scanned", 0) for r in result.parse_results
    )
    result.elapsed_seconds = time.time() - start_time
    
    logger.info(f"Pipeline complete: {result.to_summary()}")
    
    return result


def detect_and_summarize(
    paths: List[Path],
    recursive: bool = True
) -> Tuple[List[DetectionResult], List[str]]:
    """Detection-only function for CLI preview.
    
    Run detection without parsing. Useful for preview/scan commands.
    
    Args:
        paths: Paths to scan.
        recursive: Use recursive detection.
        
    Returns:
        Tuple of (detections, summary_lines).
    """
    all_detections: List[DetectionResult] = []
    
    for path in paths:
        if not path.exists() or not path.is_dir():
            continue
        
        try:
            if recursive:
                detections = detect_sources_recursive(path)
            else:
                detections = detect_sources(path)
            all_detections.extend(detections)
        except Exception as e:
            logger.error(f"Error detecting {path}: {e}")
    
    summary = summarize_detections(all_detections)
    return (all_detections, summary)


def parse_single_source(
    path: Path,
    platform: SourcePlatform,
    progress: Optional[ProgressCallback] = None
) -> ParseResult:
    """Parse a single known source without detection.
    
    Args:
        path: Path to parse.
        platform: Known platform type.
        progress: Optional progress callback.
        
    Returns:
        ParseResult from the parser.
        
    Raises:
        PipelineError: If parser not found for platform.
    """
    from src.parsers import ParserRegistry
    
    registry = ParserRegistry()
    parser = registry.get_parser(platform)
    
    if not parser:
        raise PipelineError(
            f"No parser available for {platform.value}",
            "parser_selection"
        )
    
    logger.info(f"Parsing {platform.value} at {path}")
    return parser.parse(path, progress)


def merge_memories(memory_lists: List[List[Memory]]) -> List[Memory]:
    """Merge multiple memory lists into one.
    
    Args:
        memory_lists: Lists of memories to merge.
        
    Returns:
        Single merged list, sorted by created_at.
    """
    all_memories: List[Memory] = []
    for mem_list in memory_lists:
        all_memories.extend(mem_list)
    
    # Sort by created_at (None last)
    all_memories.sort(key=lambda m: (m.created_at is None, m.created_at or 0))
    
    return all_memories


def deduplicate_memories(
    memories: List[Memory],
    tolerance_seconds: int = 60
) -> Tuple[List[Memory], int]:
    """Remove duplicate memories using intelligent merging.
    
    Args:
        memories: List of memories to deduplicate.
        tolerance_seconds: Tolerance for same-moment detection.
        
    Returns:
        Tuple of (deduplicated_list, count_removed).
        
    Strategy:
        1. Group by approximate timestamp (within tolerance)
        2. Within groups, identify likely duplicates
        3. Merge duplicates, keeping richest metadata
    """
    if not memories:
        return ([], 0)
    
    # Sort by timestamp
    sorted_memories = sorted(
        memories,
        key=lambda m: (m.created_at is None, m.created_at or 0)
    )
    
    unique_memories: List[Memory] = []
    duplicates_count = 0
    skip_indices: Set[int] = set()
    
    for i, memory in enumerate(sorted_memories):
        if i in skip_indices:
            continue
        
        # Find potential duplicates in the next few memories
        duplicates = [memory]
        
        for j in range(i + 1, min(i + 10, len(sorted_memories))):
            if j in skip_indices:
                continue
            
            other = sorted_memories[j]
            
            # Check if potential duplicate
            if _is_duplicate(memory, other, tolerance_seconds):
                duplicates.append(other)
                skip_indices.add(j)
        
        # Merge duplicates if found
        if len(duplicates) > 1:
            merged = _merge_duplicate_memories(duplicates)
            unique_memories.append(merged)
            duplicates_count += len(duplicates) - 1
        else:
            unique_memories.append(memory)
    
    return (unique_memories, duplicates_count)


def _is_duplicate(mem1: Memory, mem2: Memory, tolerance_seconds: int) -> bool:
    """Check if two memories are duplicates.
    
    Args:
        mem1: First memory.
        mem2: Second memory.
        tolerance_seconds: Timestamp tolerance.
        
    Returns:
        True if likely duplicates.
    """
    # Must both have timestamps
    if not mem1.created_at or not mem2.created_at:
        return False
    
    # Check timestamp proximity
    time_diff = abs((mem1.created_at - mem2.created_at).total_seconds())
    if time_diff > tolerance_seconds:
        return False
    
    # Same source path = definitely duplicate
    if mem1.source_path and mem2.source_path and mem1.source_path == mem2.source_path:
        return True
    
    # Similar location = likely duplicate
    if mem1.location and mem2.location:
        if mem1.location.geo_point and mem2.location.geo_point:
            # Within ~100 meters
            lat_diff = abs(mem1.location.geo_point.latitude - mem2.location.geo_point.latitude)
            lon_diff = abs(mem1.location.geo_point.longitude - mem2.location.geo_point.longitude)
            if lat_diff < 0.001 and lon_diff < 0.001:
                return True
    
    # Same media type at same moment = likely duplicate
    if mem1.media_type == mem2.media_type and time_diff <= 2:
        return True
    
    return False


def _merge_duplicate_memories(duplicates: List[Memory]) -> Memory:
    """Merge duplicate memories into one, keeping best metadata.
    
    Args:
        duplicates: List of duplicate memories.
        
    Returns:
        Merged memory with best metadata from all.
    """
    # Sort by metadata completeness
    sorted_dups = sorted(duplicates, key=lambda m: _metadata_score(m), reverse=True)
    
    # Start with richest one
    base = sorted_dups[0]
    
    # Merge in data from others
    all_people = set()
    all_tags = set()
    
    for dup in sorted_dups:
        if dup.people:
            all_people.update(m.name for m in dup.people if m.name)
        if dup.tags:
            all_tags.update(dup.tags)
        
        # Use higher confidence location
        if dup.location and (not base.location or dup.location.confidence.value > base.location.confidence.value):
            base.location = dup.location
        
        # Use longer caption
        if dup.caption and (not base.caption or len(dup.caption) > len(base.caption)):
            base.caption = dup.caption
    
    # Update merged data
    from src.core.memory import PersonTag
    base.people = [PersonTag(name=name) for name in all_people]
    base.tags = list(all_tags)
    
    return base


def _metadata_score(memory: Memory) -> int:
    """Calculate metadata completeness score.
    
    Args:
        memory: Memory to score.
        
    Returns:
        Score (higher = more complete).
    """
    score = 0
    if memory.created_at:
        score += 10
    if memory.location:
        score += 10
    if memory.people:
        score += 5 * len(memory.people)
    if memory.caption:
        score += 5
    if memory.tags:
        score += 2 * len(memory.tags)
    if memory.album_name:
        score += 3
    return score


def find_duplicate_groups(
    memories: List[Memory],
    tolerance_seconds: int = 60
) -> List[List[Memory]]:
    """Find groups of potential duplicates for review.
    
    Args:
        memories: List of memories to check.
        tolerance_seconds: Timestamp tolerance.
        
    Returns:
        List of groups, each containing potential duplicates.
    """
    groups: List[List[Memory]] = []
    processed: Set[int] = set()
    
    for i, memory in enumerate(memories):
        if i in processed:
            continue
        
        group = [memory]
        processed.add(i)
        
        for j in range(i + 1, len(memories)):
            if j in processed:
                continue
            
            if _is_duplicate(memory, memories[j], tolerance_seconds):
                group.append(memories[j])
                processed.add(j)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups


def enrich_memories_cross_platform(memories: List[Memory]) -> List[Memory]:
    """Enrich memories using data from other platforms.
    
    Args:
        memories: List of memories to enrich.
        
    Returns:
        Enriched memories (modified in place, but also returned).
        
    Examples:
        Snapchat memory at 2:30 PM + Google Photos with GPS at 2:31 PM
        → Add GPS to Snapchat memory (with lower confidence).
    """
    # Group by approximate time
    time_groups: Dict[int, List[Memory]] = {}
    
    for memory in memories:
        if not memory.created_at:
            continue
        
        # Group by 5-minute buckets
        bucket = int(memory.created_at.timestamp() // 300)
        if bucket not in time_groups:
            time_groups[bucket] = []
        time_groups[bucket].append(memory)
    
    # Enrich within groups
    for group in time_groups.values():
        if len(group) < 2:
            continue
        
        # Find best location in group
        best_location = None
        for mem in group:
            if mem.location and mem.location.geo_point:
                if not best_location or mem.location.confidence.value > best_location.confidence.value:
                    best_location = mem.location
        
        # Apply to memories without location
        if best_location:
            for mem in group:
                if not mem.location:
                    from src.core.memory import Location
                    mem.location = Location(
                        geo_point=best_location.geo_point,
                        confidence=ConfidenceLevel.MEDIUM  # Lower confidence for inferred
                    )
    
    return memories


def quick_parse(path: Path) -> List[Memory]:
    """Convenience wrapper for simple parsing use cases.
    
    Args:
        path: Path to parse.
        
    Returns:
        List of extracted memories.
        
    Raises:
        PipelineError: If parsing fails.
        
    Example:
        >>> memories = quick_parse(Path("/my_photos"))
    """
    config = PipelineConfig(fail_fast=True)
    result = run_pipeline([path], config)
    
    if result.has_critical_failures():
        raise PipelineError(
            "Parsing failed",
            "parsing",
            result
        )
    
    return result.memories


def parse_to_timeline(
    paths: List[Path],
    config: Optional[PipelineConfig] = None
):
    """Parse and immediately construct Timeline.
    
    Args:
        paths: Paths to parse.
        config: Pipeline configuration.
        
    Returns:
        Timeline object ready for analysis.
        
    Note:
        Requires src.core.timeline.Timeline to be available.
    """
    result = run_pipeline(paths, config)
    
    try:
        from src.core.timeline import Timeline
        timeline = Timeline(memories=result.memories)
        return timeline
    except ImportError:
        logger.warning("Timeline module not available, returning memories only")
        return result.memories
