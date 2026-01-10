"""Visual tagging subsystem for Digital Life Narrative AI.

Uses Stage 1 (low-cost vision models) to extract scenes, vibes, and motifs 
from sampled images to ground the narrative in visual evidence.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from dlnai.ai.client import AIClient
from dlnai.core.models import AnalysisConfig, LifeChapter, Memory
from dlnai.core.privacy import PrivacyGate
from dlnai.ai.prompts import get_prompt


@dataclass
class VisualTagResult:
    """Detailed results of a single image visual analysis."""
    memory_id: str
    success: bool
    scene_tags: List[str] = field(default_factory=list)
    vibe_tags: List[str] = field(default_factory=list)
    visual_motifs: List[str] = field(default_factory=list)
    description: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    tokens_used: Optional[int] = None


@dataclass
class ChapterVisualSummary:
    """Aggregated visual intelligence for a whole life chapter."""
    dominant_scenes: List[str]
    dominant_vibes: List[str]
    recurring_motifs: List[str]
    scene_distribution: Dict[str, int]
    images_analyzed: int
    average_confidence: float


class VisualTagger:
    """Stage 1: Extracts visual attributes using Gemini Flash."""

    def __init__(
        self, 
        client: AIClient, 
        config: AnalysisConfig, 
        # Note: PrivacyGate is usually passed during run, but we can hold config
        logger: logging.Logger | None = None
    ):
        self.client = client
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def _prepare_image_for_api(self, path: str, max_dim: int = 512) -> Optional[str]:
        """Load, resize and encode image for low-cost analysis."""
        try:
            with Image.open(path) as img:
                # Convert to RGB if necessary (handles PNG/HEIC/RGBA)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize keeping aspect ratio
                img.thumbnail((max_dim, max_dim))
                
                # Save to buffer
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.warning(f"Failed to prepare image {path}: {e}")
            return None

    def tag_image(self, memory: Memory) -> VisualTagResult:
        """Analyze a single memory for visual context."""
        if not memory.source_path:
            return VisualTagResult(memory.id, False, error="No source path")

        # 1. Prepare image
        encoded = self._prepare_image_for_api(memory.source_path)
        if not encoded:
            return VisualTagResult(memory.id, False, error="Image preparation failed")

        # 2. Build Prompt
        # In a real impl, we'd get this from dlnai.ai.prompts
        template = get_prompt("visual_tagging_v1")
        system, user = template.render() 
        # The prompt might expect the image as a 'part' or 'blob' 
        # Our AIClient.generate takes a prompt string. 
        # For multimodal, we usually need the SDK's image part format.
        
        # PROXY: Assuming AIClient handles multimodal if prompt includes [IMAGE]
        # or we pass it separately.
        try:
            # Note: This is an idealized call; actual SDK usage requires specific content parts
            # We'll assume client.generate(multimodal_content) works or handles base64
            response = self.client.generate_json(
                user, 
                system_instruction=system,
                image_data=encoded, # Custom extension to AIClient for this phase
                model_override=self.config.vision_model
            )

            if response.parse_success:
                data = response.data
                return VisualTagResult(
                    memory_id=memory.id,
                    success=True,
                    scene_tags=data.get("scene_tags", []),
                    vibe_tags=data.get("vibe_tags", []),
                    visual_motifs=data.get("motifs", []),
                    description=data.get("description"),
                    confidence=data.get("confidence", 0.0),
                    tokens_used=response.usage.get("total_tokens") if response.usage else None
                )
            else:
                return VisualTagResult(memory.id, False, error=f"Parse failed: {response.parse_error}")

        except Exception as e:
            self.logger.error(f"Visual tagging error for {memory.id}: {e}")
            return VisualTagResult(memory.id, False, error=str(e))

    def tag_batch(
        self, 
        memories: List[Memory], 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[VisualTagResult]:
        """Analyze multiple images with batching and progress tracking."""
        results = []
        total = len(memories)
        
        # Small batches to avoid timeout and track progress
        batch_size = 5
        for i in range(0, total, batch_size):
            batch = memories[i : i + batch_size]
            for memory in batch:
                res = self.tag_image(memory)
                results.append(res)
                
                # Apply result to memory object immediately for narrative groundedness
                if res.success:
                    memory.scene_tags = res.scene_tags
                    memory.vibe_tags = res.vibe_tags
                    memory.visual_motifs = res.visual_motifs
                    memory.visual_confidence = res.confidence
                    memory.visually_analyzed = True
            
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)
        
        return results

    def aggregate_chapter_visuals(self, results: List[VisualTagResult]) -> ChapterVisualSummary:
        """Roll up individual image tags into a chapter-level summary."""
        from collections import Counter
        
        scenes = Counter()
        vibes = Counter()
        motifs = Counter()
        confidences = []
        success_count = 0

        for r in results:
            if r.success:
                scenes.update(r.scene_tags)
                vibes.update(r.vibe_tags)
                motifs.update(r.visual_motifs)
                confidences.append(r.confidence)
                success_count += 1

        def top_n(counter, n=5):
            return [tag for tag, count in counter.most_common(n)]

        return ChapterVisualSummary(
            dominant_scenes=top_n(scenes, 3),
            dominant_vibes=top_n(vibes, 3),
            recurring_motifs=top_n(motifs, 8),
            scene_distribution=dict(scenes),
            images_analyzed=success_count,
            average_confidence=sum(confidences) / len(confidences) if confidences else 0.0
        )
