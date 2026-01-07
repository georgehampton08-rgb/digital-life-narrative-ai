# Contributing to Digital Life Narrative AI

Welcome, and thank you for considering contributing to Digital Life Narrative AI! 🎉

This project reconstructs life narratives from scattered media exports using Google's Gemini AI. It started as a hackathon project with room to grow — your contributions can make a real difference.

---

## Valuable Contribution Areas

We especially welcome contributions in these areas:

| Area | Impact | Difficulty |
| ---- | ------ | ---------- |
| 🆕 **New platform parsers** | High — unlock more data sources | Medium |
| 🛡️ **Safety heuristics** | High — protect user privacy | Medium |
| 🧠 **Prompt engineering** | High — improve narrative quality | Medium |
| 🎨 **HTML report UI/UX** | Medium — better user experience | Low-Medium |
| 📚 **Documentation & examples** | Medium — help new users | Low |
| 🧪 **Test coverage** | Medium — catch bugs early | Low |

---

## Development Setup

### Prerequisites

- **Python 3.10+**
- **Poetry** (recommended) or pip
- **Git**

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai

# Install dependencies
make install

# Verify setup
make test
```

### Optional: Full Testing with AI

```bash
# Set up API key for AI integration tests
organizer config set-key

# Run demo with AI (requires API key)
make demo-ai
```

---

## Code Standards

### Type Hints

Required for all functions and methods. Use modern Python 3.10+ syntax:

```python
# ✅ Good
def process_memories(
    memories: list[Memory],
    config: AnalysisConfig | None = None,
) -> LifeStoryReport:
    ...

# ❌ Bad
def process_memories(memories, config=None):
    ...
```

### Docstrings

Google style, required for all public functions and classes:

```python
def analyze_timeline(
    memories: list[Memory],
    gap_threshold_days: int = 30,
) -> TimelineAnalysis:
    """Analyze a timeline of memories for patterns and gaps.
    
    Identifies significant gaps, clusters of activity, and temporal
    patterns across the memory collection.
    
    Args:
        memories: List of Memory objects to analyze.
        gap_threshold_days: Minimum days between memories to flag as a gap.
    
    Returns:
        TimelineAnalysis containing gaps, patterns, and statistics.
    
    Raises:
        ValueError: If memories list is empty.
    
    Example:
        >>> analysis = analyze_timeline(memories, gap_threshold_days=14)
        >>> print(f"Found {len(analysis.gaps)} significant gaps")
    """
```

### Formatting & Linting

```bash
# Run all checks (lint + format check + test)
make ci

# Format code automatically
make format

# Run linting only
make lint
```

### Testing

- **Required** for new features
- **Mock all external services** (never call real APIs in tests)
- **Aim for meaningful coverage**, not 100%

### Pydantic Models

Use Pydantic for all data models:

```python
from pydantic import BaseModel, Field, field_validator

class ChapterConfig(BaseModel):
    """Configuration for chapter detection."""
    
    min_chapters: int = Field(default=3, ge=1, le=50)
    max_chapters: int = Field(default=15, ge=1, le=50)
    
    @field_validator("max_chapters")
    @classmethod
    def max_greater_than_min(cls, v: int, info) -> int:
        if v < info.data.get("min_chapters", 1):
            raise ValueError("max_chapters must be >= min_chapters")
        return v
```

---

## Adding a New Parser

This is one of the highest-impact contributions! Follow these steps:

### Step 1: Understand the Contract

Review these files first:

- `src/parsers/base.py` — BaseParser abstract class
- `src/core/memory.py` — Memory model (your output format)
- `src/parsers/snapchat.py` — Example implementation

### Step 2: Create the Parser File

```bash
# Create new parser file
touch src/parsers/tiktok.py
```

### Step 3: Implement Required Methods

```python
# src/parsers/tiktok.py
"""Parser for TikTok data exports."""

from pathlib import Path
from typing import Callable

from src.parsers.base import BaseParser, ParseResult, register_parser
from src.core.memory import Memory, SourcePlatform, MediaType


@register_parser
class TikTokParser(BaseParser):
    """Parser for TikTok data export archives."""
    
    PLATFORM = SourcePlatform.TIKTOK
    PLATFORM_DISPLAY_NAME = "TikTok"
    
    @classmethod
    def get_signature_files(cls) -> list[str]:
        """Return files that identify a TikTok export."""
        return [
            "Video/Videos.json",
            "Activity/Video Browsing History.json",
        ]
    
    @classmethod
    def can_parse(cls, root: Path) -> bool:
        """Check if this looks like a TikTok export."""
        for sig in cls.get_signature_files():
            if (root / sig).exists():
                return True
        return False
    
    def parse(
        self,
        root: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ParseResult:
        """Parse TikTok export into Memory objects."""
        memories: list[Memory] = []
        warnings: list[str] = []
        
        # Parse posted videos
        videos_file = root / "Video" / "Videos.json"
        if videos_file.exists():
            videos = self._parse_videos(videos_file)
            memories.extend(videos)
        
        # ... more parsing logic ...
        
        return ParseResult(
            memories=memories,
            warnings=warnings,
            source_platform=self.PLATFORM,
        )
    
    def _parse_videos(self, path: Path) -> list[Memory]:
        """Parse video metadata from Videos.json."""
        # Implementation...
        pass
```

### Step 4: Add Platform to Enum

```python
# src/core/memory.py
class SourcePlatform(str, Enum):
    """Supported data export sources."""
    UNKNOWN = "unknown"
    LOCAL_PHOTOS = "local_photos"
    SNAPCHAT = "snapchat"
    GOOGLE_PHOTOS = "google_photos"
    TIKTOK = "tiktok"  # ← Add your platform
```

### Step 5: Add Detection Fingerprints

```python
# src/detection.py - update PLATFORM_SIGNATURES
PLATFORM_SIGNATURES = {
    # ... existing platforms ...
    SourcePlatform.TIKTOK: [
        "Video/Videos.json",
        "Activity/Video Browsing History.json",
    ],
}
```

### Step 6: Write Tests

```python
# tests/test_detection_and_parsers.py

class TestTikTokParser:
    """Tests for TikTok parser."""
    
    def test_can_parse_valid_export(self, tiktok_export_dir):
        """Parser correctly identifies TikTok exports."""
        assert TikTokParser.can_parse(tiktok_export_dir)
    
    def test_can_parse_rejects_invalid(self, empty_dir):
        """Parser rejects non-TikTok directories."""
        assert not TikTokParser.can_parse(empty_dir)
    
    def test_parse_extracts_videos(self, tiktok_export_dir):
        """Parser extracts video memories."""
        parser = TikTokParser()
        result = parser.parse(tiktok_export_dir)
        
        assert len(result.memories) > 0
        assert all(m.source_platform == SourcePlatform.TIKTOK for m in result.memories)
    
    def test_parse_handles_missing_files(self, partial_tiktok_export):
        """Parser handles missing optional files gracefully."""
        parser = TikTokParser()
        result = parser.parse(partial_tiktok_export)
        
        # Should not raise, may have warnings
        assert isinstance(result.warnings, list)
```

Add fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def tiktok_export_dir(tmp_path: Path) -> Path:
    """Create a mock TikTok export structure."""
    video_dir = tmp_path / "Video"
    video_dir.mkdir()
    
    videos_json = video_dir / "Videos.json"
    videos_json.write_text(json.dumps({
        "VideoList": [
            {"Date": "2023-06-15", "Link": "..."},
        ]
    }))
    
    return tmp_path
```

### Step 7: Document

Update `README.md` supported platforms table:

```markdown
| Platform | Status | Notes |
| -------- | ------ | ----- |
| 📱 TikTok | ✅ Supported | Posted videos, browsing history |
```

---

## Modifying AI Behavior

### Prompt Templates

Located in `src/ai/prompts.py`. Each prompt has a version and category:

```python
CHAPTER_DETECTION_PROMPT = PromptTemplate(
    name="chapter_detection",
    version="1.2",
    system_prompt="""You are analyzing a life timeline...""",
    user_prompt_template="""Here are the memories: {memories_json}""",
)
```

When modifying prompts:

- Increment version number
- Document why you made the change
- Test with real data if possible

### Adding New AI Calls

Always go through `AIClient`:

```python
# ✅ Good - use AIClient
from src.ai.client import get_client

client = get_client()
response = await client.generate(prompt, system=system_prompt)

# ❌ Bad - never import directly
import google.generativeai  # NO!
```

### Token Management

```python
# Estimate tokens before sending
from src.ai.client import estimate_tokens

estimated = estimate_tokens(prompt_text)
if estimated > MAX_TOKENS:
    # Sample or chunk the data
    sampled = sample_memories(memories, max_count=500)
```

### Privacy Considerations

**Critical**: All data must go through privacy filtering:

```python
# ✅ Good - use privacy-safe payload
from src.core.privacy import PrivacyGate

gate = PrivacyGate(settings)
safe_payload = gate.filter_for_ai(memories)

# ❌ Bad - sending raw data
payload = {"paths": [str(m.source_path) for m in memories]}  # NO!
```

### Testing AI Code

```python
def test_chapter_detection_parses_response(self, mock_ai_client):
    """AI correctly parses chapter detection response."""
    mock_ai_client.generate.return_value = """
    {
        "chapters": [
            {"title": "College Years", "start": "2018-01", "end": "2022-05"}
        ]
    }
    """
    
    analyzer = LifeStoryAnalyzer(client=mock_ai_client)
    chapters = analyzer._detect_chapters(timeline, context)
    
    assert len(chapters) == 1
    assert chapters[0].title == "College Years"
```

---

## Working with Safety & Privacy

### Critical Rules

> ⚠️ **These rules are non-negotiable**

1. **Never bypass ContentFilter** for report generation
2. **Never log sensitive content** (captions, paths, personal info)
3. **Never send pixel data** without explicit consent check
4. **Always use debug-level logging** for potentially sensitive operations

### Adding Safety Categories

1. Add to enum in `src/core/safety.py`:

```python
class SafetyCategory(str, Enum):
    """Categories of potentially sensitive content."""
    EXPLICIT = "explicit"
    VIOLENCE = "violence"
    PERSONAL_INFO = "personal_info"
    FINANCIAL = "financial"  # ← New category
```

1. Add detection in `ContentFilter`:

```python
def _check_financial(self, memory: Memory) -> SafetyFlag | None:
    """Check for financial information in content."""
    patterns = [r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b']  # Card numbers
    # ...
```

1. Add tests:

```python
def test_detects_credit_card_numbers(self):
    """Filter detects credit card patterns."""
    memory = Memory(caption="My card is 1234-5678-9012-3456")
    flag = filter._check_financial(memory)
    assert flag is not None
    assert flag.category == SafetyCategory.FINANCIAL
```

### Adding Privacy Controls

1. Add to `PrivacySettings` in `src/core/privacy.py`
2. Update `PrivacyGate.filter_for_ai()` to respect new setting
3. Document in `PRIVACY.md`

---

## HTML Report Improvements

### Structure

- Templates in `src/output/html_report.py`
- CSS/JS embedded (self-contained output)
- Jinja2 templating

### Constraints

| Requirement | Why |
| ----------- | --- |
| **Works offline** | Users may not have internet |
| **No external dependencies** | CDNs can go down |
| **Responsive design** | Mobile viewing |
| **Accessible** | Semantic HTML, good contrast |

### Testing Report Changes

```bash
# Generate test report
python -m demo.generate_demo_data --output ./demo_data
organizer analyze -i ./demo_data -o ./test_report.html --no-ai

# Test in multiple browsers
# Test mobile viewport (Chrome DevTools)
# Check accessibility (Lighthouse)
```

---

## Hackathon vs Production Quality

### Hackathon-Quality (Acceptable Now)

- ✅ Working over perfect
- ✅ Good enough error handling
- ✅ Basic test coverage
- ✅ Core functionality documented

### Production-Quality (Aspirational)

- 🎯 Comprehensive error handling
- 🎯 Full test coverage
- 🎯 Performance optimization
- 🎯 Extensive documentation

### Large Refactors

If you're proposing significant architectural changes:

1. **Open an issue first** describing the problem
2. **Discuss approach** before implementing
3. **Get buy-in** from maintainers
4. **Implement incrementally** if possible

---

## Submitting Pull Requests

### Before Submitting

- [ ] Tests pass: `python -m pytest tests/ --override-ini="addopts="`
- [ ] Linting passes: `black src tests && ruff check src tests`
- [ ] New code has docstrings
- [ ] Documentation updated if needed

### PR Title Format

Use conventional commits:

```text
feat: Add TikTok parser
fix: Handle missing EXIF timestamps  
docs: Update privacy documentation
refactor: Simplify cache validation
test: Add timeline gap detection tests
chore: Update dependencies
```

### PR Description Template

```markdown
## What
Brief description of changes.

## Why
Motivation for the change.

## How to Test
Steps to verify the changes work.

## Breaking Changes
Any breaking changes (or "None").

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passes
```

### Review Process

1. Maintainer will review within a few days
2. Address feedback in new commits
3. Discussion happens in PR comments
4. Squash merge when approved

---

## Getting Help

### Where to Ask Questions

- **GitHub Issues**: Open with "question" label
- **Discussions**: Use GitHub Discussions for broader topics

### When Asking for Help

Include:

- What you're trying to do
- What you've tried
- Relevant code snippets
- Full error messages

### Useful Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical deep-dive
- [PRIVACY.md](PRIVACY.md) — Privacy and security rules
- [demo/DEMO.md](demo/DEMO.md) — Running the demo

---

## Thank You

Every contribution matters — whether it's fixing a typo, adding a test, or implementing a new parser. Thank you for helping make this project better! 🙏

*Questions? Open an issue or reach out to <georgehampton08@gmail.com>*
