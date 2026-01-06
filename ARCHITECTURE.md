# Architecture

Technical documentation for the Digital Life Narrative AI system architecture.

## Table of Contents

- [Overview](#overview)
- [AI-First Philosophy](#ai-first-philosophy)
- [Module Responsibilities](#module-responsibilities)
- [Data Flow](#data-flow)
- [AI Integration](#ai-integration)
- [Adding New Platforms](#adding-new-platforms)
- [Security Considerations](#security-considerations)

---

## Overview

Digital Life Narrative AI transforms scattered media exports into a cohesive life story using AI-powered narrative analysis.

### System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                      │
│   ┌─────────────┐                                         ┌──────────────┐  │
│   │   CLI       │  organizer analyze -i ~/exports         │  HTML/JSON   │  │
│   │  (cli.py)   │  organizer config set-key               │   Report     │  │
│   └──────┬──────┘                                         └──────▲───────┘  │
└──────────┼───────────────────────────────────────────────────────┼──────────┘
           │                                                       │
           ▼                                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         Main Pipeline                                 │  │
│   │   Detection → Parsing → Normalization → Analysis → Report            │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
           │                                                       ▲
           ▼                                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE LAYER                                      │
│                                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │ detection  │    │  parsers/  │    │    ai/     │    │  report    │       │
│  │            │    │            │    │            │    │            │       │
│  │ Identify   │───▶│ Extract    │───▶│ Analyze    │───▶│ Generate   │       │
│  │ Platform   │    │ MediaItems │    │ Chapters   │    │ HTML/JSON  │       │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘       │
│                                             │                                │
│                                             ▼                                │
│                                    ┌────────────────┐                        │
│                                    │ Gemini API     │                        │
│                                    │ (External)     │                        │
│                                    └────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│                                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                         │
│  │ models.py  │    │ config.py  │    │ utils/     │                         │
│  │            │    │            │    │            │                         │
│  │ MediaItem  │    │ AppConfig  │    │ logging    │                         │
│  │ LifeReport │    │ APIKeyMgr  │    │ hashing    │                         │
│  └────────────┘    └────────────┘    └────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```text
User Exports     Detection      Parsing        Normalization    AI Analysis     Report
    │                │              │                │               │             │
    ▼                ▼              ▼                ▼               ▼             ▼
┌─────────┐    ┌──────────┐   ┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐
│Snapchat │    │Identify  │   │Extract   │    │Unified   │    │Chapter   │  │Beautiful │
│Google   │───▶│Platform  │──▶│Metadata  │───▶│MediaItem │───▶│Detection │─▶│HTML      │
│Facebook │    │Type      │   │& Content │    │Schema    │    │Narrative │  │Report    │
│Local    │    └──────────┘   └──────────┘    └──────────┘    └──────────┘  └──────────┘
└─────────┘         │              │                │               │             │
                    │              │                │               │             │
                Platform      Raw JSON/EXIF     Pydantic      AI Prompts     Jinja2
                Heuristics    Parsing           Models        to Gemini      Templates
```

---

## AI-First Philosophy

### Why AI is the Core

This project is fundamentally designed around AI as the central intelligence layer — not as an optional feature. Here's why:

#### 1. **Narrative Understanding Requires Intelligence**

Traditional photo organizers use simple heuristics:

- Sort by date ✓
- Group by location ✓
- Tag by detected faces ✓

But they cannot:

- Identify that a series of photos represents a "move to a new city"
- Recognize that a gap in data might indicate a difficult period
- Understand that mixed Snapchat and Instagram posts show different aspects of your personality
- Write a coherent story about your life

**AI bridges the gap from data to meaning.**

#### 2. **Pattern Recognition Across Unstructured Data**

Media exports are messy:

- Inconsistent timestamps
- Missing metadata
- Duplicate files
- Platform-specific quirks

AI can make sense of this chaos by understanding context, not just parsing rules.

#### 3. **The Fallback Acknowledged Gap**

When AI is unavailable (no API key, rate limits, errors), the system degrades to **fallback mode**:

| Feature | With AI | Fallback Mode |
| ------- | ------- | ------------- |
| Chapters | Semantic life chapters | Calendar years |
| Narratives | Rich, contextual stories | "AI analysis unavailable" |
| Insights | Platform behavior analysis | Basic statistics |
| Gaps | Speculation on reasons | Period identified only |
| Summary | Cohesive life story | Item count + date range |

**Fallback mode is explicitly marked** — we don't pretend rules can replicate intelligence.

### Why Not Rules-Based Alternative?

We considered building elaborate rule-based systems:

```python
# This approach was rejected:
if photos_in_week > 50:
    if unique_locations > 3:
        chapter = "Travel adventure"
    elif people_count > 10:
        chapter = "Social gathering"
```

**Problems:**

1. **Brittle**: Hardcoded thresholds break with different lifestyles
2. **Culturally biased**: "Wedding" rules differ across cultures
3. **Context-blind**: Can't understand caption semantics
4. **Unmaintainable**: Exponential rule combinations
5. **Lies to users**: Pretends to understand when it doesn't

**Our philosophy**: Be honest. AI provides intelligence. Without it, provide statistics.

---

## Module Responsibilities

### `models.py` — Data Contracts

**Purpose**: Define the shape of all data flowing through the system using Pydantic models.

```python
# Core entities
MediaItem        # Single photo/video with normalized metadata
ParseResult      # Output from a parser
LifeStoryReport  # Final AI-generated report

# AI-generated entities  
LifeChapter              # A chapter in your life story
PlatformBehaviorInsight  # How you used each platform
DataGap                  # Periods of missing data

# Configuration
AnalysisConfig   # Control analysis behavior
```

**Key principles**:

- Immutable after creation
- Full type hints
- Validation via Pydantic
- JSON serializable

---

### `detection.py` — Source Identification

**Purpose**: Identify what kind of export a directory contains.

```python
# Input:  Path to a directory
# Output: DetectionResult with platform, confidence, evidence
```

**Detection strategies**:

1. **Signature files**: `memories_history.json` → Snapchat
2. **Directory structure**: `Takeout/Google Photos/` → Google Photos
3. **File patterns**: Naming conventions, metadata files
4. **Fallback**: Directories with images → Local

**Design decision**: Detection is probabilistic. Returns confidence levels (HIGH, MEDIUM, LOW).

---

### `parsers/` — Data Extraction

**Purpose**: Extract normalized `MediaItem` objects from platform-specific exports.

```text
parsers/
├── __init__.py       # Registry and exports
├── base.py           # BaseParser with common utilities
├── snapchat.py       # Snapchat-specific parsing
├── google_photos.py  # Google Takeout parsing
└── local.py          # Generic local media (fallback)
```

**Parser responsibilities**:

1. Parse platform-specific JSON/metadata
2. Extract timestamps (multiple strategies per platform)
3. Extract location (GPS, place names, country)
4. Extract people (face tags, mentions)
5. Handle duplicates
6. Generate deterministic IDs

**Registry pattern**: Parsers self-register on import:

```python
@ParserRegistry.register(SourcePlatform.SNAPCHAT)
class SnapchatParser(BaseParser):
    ...
```

---

### `ai/` — Intelligence Layer

**Purpose**: All AI-related functionality, isolated from core logic.

```text
ai/
├── __init__.py        # Public API
├── client.py          # Gemini API wrapper
├── life_analyzer.py   # Main analysis engine
└── fallback.py        # Statistics-only fallback
```

**Key classes**:

| Class | Role |
| ----- | ---- |
| `AIClient` | Low-level Gemini wrapper with retry logic |
| `LifeStoryAnalyzer` | Orchestrates full analysis pipeline |
| `FallbackAnalyzer` | Produces reports without AI |

**AI isolation principle**: The rest of the system never calls Gemini directly. All AI goes through this module.

---

### `report.py` — Output Generation

**Purpose**: Transform `LifeStoryReport` into beautiful, shareable outputs.

**Outputs**:

1. **HTML**: Self-contained, interactive report with inline CSS/JS
2. **JSON**: Machine-readable export for integrations

**HTML features**:

- Responsive design
- Dark mode toggle
- Interactive chapter timeline
- Print-friendly
- No external dependencies (fully offline)
- Clear fallback mode warning

---

### `organizer.py` — File Operations

**Purpose**: Optional feature to organize actual files into chapter-based folders.

**Modes**:

- `COPY`: Duplicate files (safe)
- `MOVE`: Relocate files (careful)
- `SYMLINK`: Create links (advanced)

**Safety features**:

- Preview mode by default
- Confirmation before execution
- Undo log for rollback
- Never deletes originals

---

## Data Flow

### MediaItem Lifecycle

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DETECTION                                                           │
│                                                                              │
│   /exports/takeout/               DetectionResult                            │
│   ├── Takeout/          ────►     platform: GOOGLE_PHOTOS                    │
│   │   └── Google Photos/          confidence: HIGH                           │
│   └── ...                         evidence: ["Takeout/Google Photos"]        │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PARSING                                                             │
│                                                                              │
│   GooglePhotosParser.parse()                                                 │
│                                                                              │
│   Raw: IMG_001.jpg + IMG_001.jpg.json        Normalized: MediaItem           │
│   {                                          {                               │
│     "photoTakenTime": {"ts": "1594828800"},    id: UUID                      │
│     "geoData": {"lat": 41.87, "lon": -87.6},   source_platform: GOOGLE       │
│     "people": [{"name": "Alice"}]              media_type: PHOTO             │
│   }                                            timestamp: 2020-07-15         │
│                                                location: Chicago             │
│                                                people: ["Alice"]             │
│                                              }                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: AI PREPARATION                                                      │
│                                                                              │
│   LifeStoryAnalyzer._prepare_items_for_ai()                                  │
│                                                                              │
│   MediaItem                           AI-Safe Summary                        │
│   {                                   {                                      │
│     file_path: "/users/jo..."   ──►     "date": "2020-07-15",               │
│     people: ["Alice Smith"]             "platform": "google_photos",         │
│     ...                                 "location": "Chicago"                │
│   }                                     # paths anonymized                   │
│                                         # names optionally hashed            │
│                                       }                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: AI ANALYSIS                                                         │
│                                                                              │
│   Multiple AI calls with specialized prompts:                                │
│                                                                              │
│   1. Chapter Detection    →  "Identify 5-10 life chapters..."               │
│   2. Narrative Generation →  "Write 2-3 paragraphs about..."                │
│   3. Platform Analysis    →  "Analyze usage patterns..."                    │
│   4. Executive Summary    →  "Weave a cohesive story..."                    │
│                                                                              │
│   Output: LifeStoryReport                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: REPORT GENERATION                                                   │
│                                                                              │
│   ReportGenerator.generate()                                                 │
│                                                                              │
│   LifeStoryReport     →     Jinja2 Template     →     life_story.html       │
│   (Python object)           + CSS/JS                  (Self-contained)       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## AI Integration

### Prompt Engineering Approach

All prompts follow a consistent structure:

```python
SYSTEM_PROMPT = """You are analyzing a personal media collection..."""

USER_PROMPT = """
Given this temporal summary of media items:
{temporal_summary}

And these sample items:
{sampled_items}

Identify {max_chapters} distinct life chapters.
Return valid JSON: {...}
"""
```

**Prompt design principles**:

1. **Context first**: Provide temporal summary before raw data
2. **Explicit structure**: Always specify expected JSON output
3. **Constrained output**: Define exact fields and formats
4. **Guidance not rules**: "Consider..." not "You must..."

### Token Management for Large Datasets

Users may have 50,000+ media items. We can't send all to the API.

**Sampling strategy** (`_sample_items_for_prompt`):

```python
def _sample_items_for_prompt(items: list, max_items: int = 200) -> list:
    # 1. Sort by timestamp
    # 2. Include first and last (boundaries)
    # 3. Evenly sample across time range
    # 4. Prioritize items with rich metadata
    # 5. Ensure platform diversity
```

**Token budget allocation**:

| Component | ~Token Budget |
| --------- | ------------- |
| System prompt | 500 |
| Temporal summary | 1,000 |
| Sampled items | 5,000 |
| Response buffer | 2,000 |
| **Total** | ~8,500 |

### Retry and Fallback Strategies

```python
# Retry hierarchy
1. Transient error → Exponential backoff (up to 3 retries)
2. Rate limit → Longer backoff with jitter
3. Token limit → Reduce sample size, retry
4. API down → Fall back to FallbackAnalyzer
5. Content filtered → Use available partial response
```

**Graceful degradation per chapter**:

- If chapter 3 narrative fails, other chapters still succeed
- Reports are generated even with partial AI failures

---

## Adding New Platforms

### Step-by-Step Guide

#### 1. Add Platform to Enum

```python
# models.py
class SourcePlatform(str, Enum):
    ...
    TIKTOK = "tiktok"  # New platform
```

#### 2. Add Detection Heuristics

```python
# detection.py
def _detect_tiktok(path: Path) -> DetectionResult | None:
    """Detect TikTok export structure."""
    # Look for signature files
    if (path / "user_data.json").exists():
        return DetectionResult(
            platform=SourcePlatform.TIKTOK,
            confidence=Confidence.HIGH,
            evidence=["user_data.json"],
            root_path=path,
        )
    return None
```

#### 3. Create Parser

```python
# parsers/tiktok.py
from organizer.parsers.base import BaseParser, ParserRegistry

@ParserRegistry.register(SourcePlatform.TIKTOK)
class TikTokParser(BaseParser):
    """Parser for TikTok data exports."""
    
    platform = SourcePlatform.TIKTOK
    
    def can_parse(self, path: Path) -> bool:
        return (path / "user_data.json").exists()
    
    def parse(self, path: Path) -> ParseResult:
        items = []
        # Parse TikTok-specific structure
        # ... extraction logic ...
        return ParseResult(items=items, ...)
```

#### 4. Register in `__init__.py`

```python
# parsers/__init__.py
from organizer.parsers.tiktok import TikTokParser  # Auto-registers
```

#### 5. Add Platform Icon/Color

```python
# report.py
PLATFORM_ICONS = {
    ...
    "tiktok": "🎵",
}
```

#### 6. Add Tests

```python
# tests/test_parsers.py
class TestTikTokParser:
    def test_can_parse_valid_export(self, tiktok_export_dir):
        ...
```

---

## Security Considerations

### API Key Storage

Three-tier secure storage system:

```text
┌─────────────────────────────────────────────────────┐
│                  API Key Storage                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Tier 1: Environment Variable (recommended)        │
│  ┌─────────────────────────────────────────────┐   │
│  │  GEMINI_API_KEY=your-key-here               │   │
│  │  - Never in code                            │   │
│  │  - CI/CD friendly                           │   │
│  └─────────────────────────────────────────────┘   │
│                          │                          │
│                          ▼                          │
│  Tier 2: System Keyring                            │
│  ┌─────────────────────────────────────────────┐   │
│  │  Windows: Credential Manager                │   │
│  │  macOS: Keychain                            │   │
│  │  Linux: Secret Service (KDE/GNOME)          │   │
│  └─────────────────────────────────────────────┘   │
│                          │                          │
│                          ▼                          │
│  Tier 3: Encrypted File (fallback)                 │
│  ┌─────────────────────────────────────────────┐   │
│  │  Fernet symmetric encryption                 │   │
│  │  Key derived from machine ID                 │   │
│  │  ~/.config/organizer/api_key.enc            │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Privacy Filtering Before AI

**What IS sent to Gemini**:

- Anonymized timestamps ("2020-07-15", not full datetime)
- General locations ("Chicago", not coordinates)
- Platform names
- Media type (photo/video)
- Optionally hashed people names

**What is NEVER sent**:

- File paths (contain usernames, folder structure)
- Raw GPS coordinates
- Original filenames
- File contents/images themselves
- Captions (truncated, optionally excluded)

```python
# Privacy transformation example
def _prepare_items_for_ai(items, privacy_settings):
    return [
        {
            "date": item.timestamp.strftime("%Y-%m-%d"),
            "platform": item.source_platform.value,
            "location": item.location.place_name if item.location else None,
            "type": item.media_type.value,
            # file_path: EXCLUDED
            # raw coordinates: EXCLUDED
        }
        for item in items
    ]
```

### Data Transmission

```text
┌──────────────────────────────────────────────────────────────────┐
│                    Data Boundaries                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  YOUR MACHINE                            EXTERNAL                 │
│  ┌─────────────────────────────────┐     ┌──────────────────┐    │
│  │                                 │     │                  │    │
│  │  Original Media Files           │     │  Google Gemini   │    │
│  │  ├── photo1.jpg                 │     │                  │    │
│  │  ├── video2.mp4        ─────────────► │  Only receives:  │    │
│  │  └── ...                        │     │  - Dates         │    │
│  │                                 │     │  - Platforms     │    │
│  │  Extracted Metadata             │     │  - Locations     │    │
│  │  ├── timestamps                 │     │  - Statistics    │    │
│  │  ├── locations                  │     │                  │    │
│  │  └── people                     │     └──────────────────┘    │
│  │                                 │                              │
│  │  Generated Reports              │     No other external        │
│  │  ├── life_story.html   ◄───────────── connections              │
│  │  └── life_story.json            │                              │
│  │                                 │                              │
│  └─────────────────────────────────┘                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Local-Only Mode

For maximum privacy, users can run in local-only mode:

```bash
organizer analyze -i ~/exports -o ./report --no-ai
```

This:

- Makes **zero** external network calls
- Uses `FallbackAnalyzer` for statistics
- Clearly marks report as fallback mode
- Still provides useful organization

---

## Development Practices

### Type Safety

- Full type hints throughout
- `mypy --strict` compatibility
- Pydantic validation at boundaries

### Testing Strategy

```text
tests/
├── conftest.py      # Shared fixtures
├── test_models.py   # Data model tests
├── test_parsers.py  # Parser tests
├── test_ai.py       # AI tests (mocked)
└── test_cli.py      # CLI tests
```

### Logging

Centralized logging via `organizer.utils.logging`:

```python
from organizer.utils import get_logger, LogContext

logger = get_logger(__name__)

with LogContext("Parsing Snapchat"):
    ...  # Logs duration automatically
```

---

## Future Considerations

### Planned Platforms

- [ ] TikTok
- [ ] Twitter/X
- [ ] iCloud Photos
- [ ] WhatsApp

### Potential Features

- [ ] Multi-language narrative generation
- [ ] Photo embedding for visual similarity
- [ ] Interactive chapter editing
- [ ] Export to other formats (PDF, EPUB)
