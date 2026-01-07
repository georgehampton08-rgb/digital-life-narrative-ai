# Digital Life Narrative AI — Gemini Hackathon Submission

---

## 📌 One-Sentence Pitch

**An AI-first personal archive analyzer that transforms scattered digital memories into coherent life narratives using Gemini's reasoning capabilities, enabling people to understand and share their own stories.**

---

## 📝 Project Summary

Digital Life Narrative AI solves the "data sprawl" problem: your photos are scattered across Snapchat, Google Photos, local drives, and old phones with no unified story. We ingest multi-platform exports, normalize the metadata, and use **Gemini 1.5 Pro** to detect meaningful life chapters, generate warm narratives for each period, and produce beautiful interactive HTML reports. What makes this unique is that it's **AI-first by design** — without Gemini's reasoning, it degrades to a basic statistics tool with no real value. The AI doesn't just enhance the experience; it *is* the experience.

---

## 🚨 The Problem

### The Data Sprawl Crisis

Your digital life is fragmented:

- 📸 **Google Photos Takeout** archives with cryptic JSON sidecars
- 👻 **Snapchat Memories** exports in proprietary formats
- 💾 **Local photo folders** from old phones and hard drives
- 📱 **iCloud backups** that never quite sync everywhere
- 📘 **Facebook downloads** you requested years ago and never opened

Each platform has different:

- Export formats (ZIP, TAR, nested folders)
- Metadata structures (JSON, EXIF, XML, CSV)
- Naming conventions (timestamps, hashes, platform IDs)
- Completeness levels (missing dates, locations, captions)

### The Meaning Gap

Existing solutions focus on the *wrong problem*:

| Tool Type | What It Does | What It Doesn't Do |
| --------- | ------------ | ------------------ |
| **Photo organizers** | Sort by date, deduplicate | Extract meaning, detect transitions |
| **Cloud backup** | Store files, sync devices | Understand your story |
| **Timeline apps** | Display chronological events | Synthesize narratives, cross-reference |
| **Journaling apps** | Require manual input | Work from existing data |

**The result**: You have terabytes of photos but no coherent understanding of your own history.

### The Human Need

People want to:

- **Understand their own life arc**: "What did my twenties actually look like?"
- **Share meaningful summaries**: Not 10,000 photos, but the *story*
- **Digital legacy**: Leave something interpretable for family
- **Memory support**: "When did I move to Chicago? What was happening then?"
- **Closure and reflection**: Make sense of chaotic periods

### Why This Matters

**Emotional resonance**: Narratives create meaning that photo grids can't  
**Cognitive benefits**: Story structure aids memory and comprehension  
**Practical value**: Digital estate planning, life reviews, family histories  
**Current impossibility**: Manual curation of 10+ years is overwhelming

---

## ✨ The Solution

### What Digital Life Narrative AI Does

**Core Pipeline:**

```text
Platform Exports → Unified Normalization → Gemini Analysis → Interactive Report
```

1. **Intelligent Detection**: Automatically identifies export types (Snapchat, Google Photos, local folders)
2. **Unified Normalization**: Converts all platforms to single `Memory` data model
3. **Gemini-Powered Analysis**: Detects semantic life chapters, generates narratives, identifies patterns
4. **Beautiful Output**: Self-contained HTML report with interactive timeline

### Key Features

| Feature | Description |
| ------- | ----------- |
| 🧠 **Semantic Chapter Detection** | AI identifies meaningful periods: "The Chicago Years", "Starting College", not just "2020" |
| ✍️ **AI-Written Narratives** | 2-3 paragraph stories for each chapter with themes, events, and transitions |
| 📱 **Cross-Platform Insights** | "You used Snapchat for spontaneous moments, Google Photos for memories you wanted to keep" |
| 📭 **Gap Detection** | "No data Mar-May 2020... possibly the pandemic adjustment period" |
| 🎨 **Interactive HTML Report** | Self-contained, shareable, works offline, dark mode |
| 🔒 **Privacy-First** | Files stay local, only metadata sent to AI |
| 📁 **Optional Organization** | Rename/copy files into chapter-based folders |

### What You Get

The output report includes:

- **Interactive Timeline**: Clickable chapters spanning your entire archive
- **Chapter Cards**: Each with:
  - AI-generated title ("The College Years", "New City, New Life")
  - 2-3 paragraph narrative describing the period
  - Key themes and events
  - Date range and memory count
  - Representative images
- **Executive Life Summary**: AI synthesis of your entire journey
- **Platform Behavior Insights**: How you used different apps reflects different life aspects
- **Data Quality Notes**: What's missing, what might be incomplete

### Visual Example

```text
┌─────────────────────────────────────────────────────────────┐
│  📖 Your Life Story                                          │
│  An AI-powered narrative of your journey                    │
│                                                              │
│  📊 2,847 memories · 5 chapters · 6 years · 3 platforms     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🎓 College Senior Year (2018-2019)                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ "The final year of college was marked by a flurry  │    │
│  │ of activity — graduation preparations, last moments │    │
│  │ with roommates, and the anxiety of what comes next. │    │
│  │ Your Snapchat usage spiked during this period,      │    │
│  │ documenting spontaneous dorm moments and late-night │    │
│  │ study sessions..."                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Why Gemini Is Essential

> [!IMPORTANT]
> **This is an AI-first product.** Without Gemini, the application explicitly degrades to a "fallback mode" that provides only basic statistics.

### The Fallback Mode Comparison

| Capability | With Gemini | Without Gemini (Fallback) |
| ---------- | ----------- | ------------------------- |
| **Chapter Detection** | Semantic: "The Chicago Move", "Pandemic at Home" | Calendar-based: "2020", "2021" |
| **Narratives** | AI-written 2-3 paragraph stories | None — just date ranges |
| **Life Summary** | Synthesized arc across years | Basic statistics |
| **Platform Insights** | Behavioral analysis | None |
| **Gap Detection** | AI speculation on meaning | None |
| **Value Proposition** | ✅ Meaningful life story | ❌ Just a metadata parser |

**The fallback mode is intentionally limited.** It exists to prove the point: *without AI, this is just another file organizer.*

### What Gemini Specifically Enables

#### 1. Temporal Reasoning

Gemini understands that:

- A cluster of photos at a new location with new people = **a move**
- Increased photo density in summer months = **travel season**
- Shift from campus locations to city locations = **graduation transition**
- Gap in data during global events = **contextual life changes**

**Why rules can't do this**: Life transitions are too varied. "Starting college" looks different for everyone.

#### 2. Narrative Generation

Translating this:

```json
{
  "period": "2020-03 to 2020-05",
  "location_shift": "college_town → home_town",
  "platform_behavior": "snapchat_frequency: -80%"
}
```

Into this:

```text
"The spring of 2020 marked an abrupt shift. Photos show a sudden 
return home, with Snapchat activity dropping dramatically. This 
aligns with the pandemic's onset — a period of uncertainty and 
adjustment reflected in the sparse digital footprint..."
```

**Why rules can't do this**: Generating warm, insightful prose requires language understanding and generation.

#### 3. Pattern Recognition Across Platforms

Gemini detects:

- **Platform personality**: Snapchat = spontaneous, Google Photos = curated
- **Relationship patterns**: Same person appearing across multiple platforms = significant relationship
- **Seasonal rhythms**: Summer travel, winter holidays, academic calendar patterns
- **Anomalies**: Sudden increase/decrease in activity = something changed

**Why rules can't do this**: The significance of patterns requires judgment and context.

#### 4. Semantic Understanding

Interpreting:

- **Caption snippets**: "graduation day" → chapter boundary
- **Location names**: "Oxford" + UK locations = likely studying abroad
- **File naming patterns**: Snapchat's cryptic filenames vs. user-named events
- **Metadata signals**: Camera model changes, timezone shifts, people tag emergence

**Why rules can't do this**: Understanding meaning requires language comprehension.

### Models Used

- **Primary Model**: `gemini-1.5-pro`
- **Why This Model**:
  - **Large context window**: Can process hundreds of memories in single prompt
  - **Strong reasoning**: Detects subtle life transitions
  - **Output quality**: Generates warm, coherent narratives
  - **Structured output**: Reliable JSON responses with schema

### Integration Approach

```python
# Example prompt structure
{
  "task": "Detect life chapters from memory timeline",
  "context": {
    "memories": [/* sampled memories with metadata */],
    "timeline_stats": {/* density, gaps, distributions */}
  },
  "constraints": {
    "min_chapter_duration": "2 months",
    "max_chapters": 12,
    "require_rationale": true
  },
  "output_schema": {/* Pydantic model as JSON schema */}
}
```

**Key techniques**:

- Stratified sampling to fit token limits
- JSON schema enforcement with Pydantic
- Retry logic with exponential backoff
- Token usage tracking

---

## 🌟 What Makes This Unique

### Differentiation Matrix

| Comparison | Them | Us |
| ---------- | ---- | -- |
| **vs. Photo Organizers** | Sort files by date/folder | Extract meaning, generate stories |
| **vs. Timeline Apps** | Display events chronologically | Synthesize narratives, detect transitions |
| **vs. Journaling Apps** | Require manual input | Work from existing data, retrospective |
| **vs. AI Photo Apps** | Tag objects, faces | Privacy-first narrative focus |
| **vs. Backup Tools** | Store and sync files | Understand and narrate life |

### Unique Technical Elements

1. **Intentional Degradation**: Fallback mode is deliberately limited to prove AI's value
2. **Privacy Architecture**: Files never leave your machine; only metadata sent to AI
3. **Safety Layer**: Content classification with configurable blur/hide actions
4. **Extensible Parsers**: Plugin system for new platforms via base class
5. **Self-Contained Output**: HTML reports work offline, no server needed
6. **Machine-Locked Cache**: Analysis cache tied to your machine for privacy

### The "AI-First" Philosophy

Most tools add AI as a feature. We designed the product assuming AI from day one:

- **Problem definition**: How to extract *meaning*, not how to organize *files*
- **Architecture**: AI orchestration at the core, not bolted on
- **UX**: Narratives are the primary output, not file lists
- **Fallback**: Explicitly worse to demonstrate AI's necessity

---

## 🛠️ Technical Implementation

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│                   CLI Interface (Click)                  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│                Detection & Parsing                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Snapchat   │  │ Google Photos│  │ Local Files  │  │
│  │    Parser    │  │    Parser    │  │    Parser    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └──────────────────┴──────────────────┘          │
│                    Memory Objects                        │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│              Timeline Aggregation                        │
│  • Group by time periods                                │
│  • Calculate density & gaps                             │
│  • Statistical summaries                                │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│          Gemini AI Analysis                             │
│  ┌──────────────────────────────────────────────────┐  │
│  │  LifeStoryAnalyzer                               │  │
│  │  • Stratified memory sampling                    │  │
│  │  • Chapter boundary detection                    │  │
│  │  • Narrative generation                          │  │
│  │  • Executive summary synthesis                   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│            HTML Report Generation                        │
│  • Interactive timeline (JavaScript)                    │
│  • Chapter cards with narratives                        │
│  • Self-contained (embedded CSS/JS)                     │
│  • Responsive design                                    │
└─────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Detection Module (`src/detection.py`)

- Auto-identifies export types by scanning directory structure
- Detects: Snapchat, Google Photos, generic local folders
- Returns confidence scores for each detected source

#### 2. Parser Pipeline (`src/parsers/`)

- **BaseParser**: Abstract class with registry pattern
- **Platform parsers**: Snapchat, Google Photos, LocalFiles
- **Normalization**: All parsers produce `Memory` objects
- **Fault tolerance**: Malformed data logged and skipped

#### 3. LifeStoryAnalyzer (`src/ai/life_analyzer.py`)

- **Orchestrates** the AI analysis pipeline
- **Sampling**: Stratified selection to fit token limits
- **Prompting**: Structured JSON schema enforcement
- **Caching**: SHA-256 fingerprinted results cache
- **Fallback**: Degrades gracefully without API key

#### 4. ContentFilter (`src/ai/content_filter.py`)

- **Safety classification**: Detects sensitive content
- **Metadata + pixel analysis**: Heuristics + Gemini Vision (opt-in)
- **Configurable actions**: Blur, hide, flag, or allow
- **Privacy-conscious**: Requires disclosure for pixel analysis

#### 5. HTMLReportGenerator (`src/output/html_report.py`)

- **Jinja2 templates**: Renders chapter cards, timeline
- **Self-contained**: All CSS/JS embedded (works offline)
- **Responsive**: Desktop and mobile friendly
- **Accessibility**: ARIA labels, keyboard navigation

### Gemini Integration Details

```python
# AIClient wrapper (src/ai/client.py)
class AIClient:
    def analyze_timeline(
        self,
        memories: List[Memory],
        timeline_stats: TimelineStats,
    ) -> LifeStoryReport:
        # 1. Build structured prompt
        prompt = self._build_chapter_detection_prompt(
            sampled_memories=self._stratified_sample(memories),
            stats=timeline_stats,
        )
        
        # 2. Call Gemini with retry logic
        response = self._call_with_retry(
            model="gemini-1.5-pro",
            prompt=prompt,
            response_schema=ChapterDetectionSchema,
        )
        
        # 3. Validate and parse
        chapters = self._parse_chapter_response(response)
        
        # 4. Generate narratives for each chapter
        for chapter in chapters:
            narrative = self._generate_chapter_narrative(
                chapter=chapter,
                memories=self._get_chapter_memories(chapter, memories),
            )
            chapter.narrative = narrative
        
        return LifeStoryReport(chapters=chapters, ...)
```

**Key patterns**:

- Pydantic models for all data (type safety)
- Retry with exponential backoff
- Token usage tracking
- RedactingFilter for security (never log API keys)

### Tech Stack

| Layer | Technology | Purpose |
| ----- | ---------- | ------- |
| **Language** | Python 3.10+ | Modern type hints, async support |
| **AI** | Gemini 1.5 Pro | Core intelligence layer |
| **CLI** | Click | Command-line interface |
| **Validation** | Pydantic | Data models & validation |
| **Templating** | Jinja2 | HTML report generation |
| **Image Processing** | Pillow | EXIF extraction, thumbnails |
| **Packaging** | Poetry | Dependency management |
| **Testing** | pytest | Unit & integration tests |

---

## 🔒 Privacy & Safety

### Privacy Architecture

> [!NOTE]
> Your files **never leave your computer**. Only metadata is sent to Gemini.

**What stays local**:

- ✅ All image and video files
- ✅ Full file paths
- ✅ GPS coordinates
- ✅ Unredacted captions

**What goes to Gemini** (with your API key):

- Dates (e.g., "2020-06-15")
- General locations (e.g., "Chicago", not coordinates)
- Platform names (e.g., "snapchat")
- Media types (e.g., "photo")
- Statistical summaries (e.g., "45% from 2020")

**Privacy modes**:

- **Standard**: City-level locations, truncated captions
- **Strict** (`--privacy-mode`): Hashed names, 50-char captions, no neighborhoods
- **Local-only** (`--no-ai`): Zero network calls, statistics only

### Safety System

**Content classification** to prevent sharing sensitive content:

| Category | Detection | Actions Available |
| -------- | --------- | ----------------- |
| NUDITY | Filename patterns, opt-in vision | Allow, flag, blur, hide |
| VIOLENCE | Caption keywords, folder paths | Allow, flag, blur, hide |
| PRIVATE | "vault", "confidential" folders | Allow, flag, blur, hide |
| SUBSTANCE | Caption keywords | Allow, flag, blur, hide |

**User control**: Configure per-category actions in settings.

📄 **Full details**: See [PRIVACY.md](../PRIVACY.md) and [SECURITY.md](../SECURITY.md)

---

## 🎬 Demo Instructions

### Quick Demo for Judges

#### Option 1: Fallback Demo (No API Key Required)

Experience the intentionally limited non-AI mode:

```bash
# 1. Generate synthetic demo data
python -m demo.generate_demo_data --output ./demo_data

# 2. Run WITHOUT AI (fallback mode)
organizer analyze --input ./demo_data --output ./demo_fallback.html --no-ai

# 3. Open the report
open ./demo_fallback.html  # macOS
# or: start ./demo_fallback.html  # Windows
# or: xdg-open ./demo_fallback.html  # Linux
```

**What to observe**:

- ⚠️ **Fallback warning banner** at the top
- 📅 **Year-based chapters** (e.g., "2020", "2021") instead of semantic ones
- 📊 **Statistics only** — no narratives, no insights
- 💭 **The absence** of what makes this valuable

#### Option 2: Full AI Demo (Requires Gemini API Key)

Experience the complete AI-powered analysis:

```bash
# 1. Configure your API key (one-time)
organizer config set-key
# Enter your Gemini API key when prompted

# 2. Generate demo data (if not already done)
python -m demo.generate_demo_data --output ./demo_data

# 3. Run WITH AI
organizer analyze --input ./demo_data --output ./demo_ai.html

# 4. Open the report
open ./demo_ai.html
```

**What to observe**:

- 🎯 **Semantic chapter titles**: "College Senior Year", "The Chicago Move"
- ✍️ **AI-generated narratives**: Rich 2-3 paragraph stories for each period
- 🔍 **Platform insights**: Behavioral analysis across Snapchat, Google Photos
- 📭 **Gap speculation**: AI reasoning about missing data periods
- 📖 **Executive summary**: Synthesized life arc

#### Side-by-Side Comparison

Open both reports and compare:

| Feature | Fallback | AI-Powered |
| ------- | -------- | ---------- |
| Chapter titles | "2019", "2020" | "Starting College", "The Pandemic Year" |
| Narratives | None | Rich prose |
| Insights | None | Cross-platform patterns |
| Value | Minimal | ✨ The whole point |

---

## 🚀 Future Roadmap

### Near-Term (Next 3 Months)

**Additional Platform Support**:

- 📘 Facebook posts, photos, location history
- 📷 Instagram posts, stories, reels
- ☁️ iCloud backup parsing
- 🎵 Spotify listening history integration

**Enhanced Chapter Detection**:

- Relationship-focused chapters detected from people tags
- Geographic chapters based on location clusters
- Activity-based chapters (fitness, travel, work)

**Improved Safety**:

- More sophisticated pixel analysis
- User feedback loop for classification accuracy
- Public vs. private report modes

### Medium-Term (6-12 Months)

**User Experience**:

- 🖥️ **Desktop GUI**: Electron or Tauri application
- 🌐 **Web interface**: Local server with browser UI
- 👁️ **Photo previews**: Thumbnails in reports
- 🎨 **Multiple themes**: Report customization

**Advanced Features**:

- **Relationship mapping**: Social network visualization
- **Multiple export formats**: PDF, video slideshow, podcast-style narration
- **Collaborative archives**: Multi-person family histories
- **AI-powered search**: Natural language queries across your archive

### Long-Term Vision (1-2 Years)

**Digital Legacy Platform**:

- Estate planning integration
- Time-locked releases ("share my college years when my kids turn 18")
- Multi-generational archives

**Memory Support Applications**:

- Alzheimer's care: Familiar memory retrieval
- PTSD therapy: Gradual exposure to past periods
- Life review: End-of-life legacy creation

**Research Tools**:

- Oral history digitization
- Sociological research on digital behavior
- Longitudinal life pattern studies

**Platform Evolution**:

- Mobile app with camera roll integration
- Real-time chapter updates
- Predictive life insights

---

## 👥 Team & Acknowledgments

### Creator

**George Hampton**  
[georgehampton08@gmail.com](mailto:georgehampton08@gmail.com)  
[GitHub: georgehampton08-rgb](https://github.com/georgehampton08-rgb)

### Acknowledgments

**Powered By**:

- [Google Gemini](https://ai.google.dev/) — The AI engine that makes this possible
- [Gemini API](https://ai.google.dev/api) — For accessible, powerful AI capabilities

**Built With**:

- [Click](https://click.palletsprojects.com/) — CLI framework
- [Pydantic](https://docs.pydantic.dev/) — Data validation
- [Jinja2](https://jinja.palletsprojects.com/) — HTML templating
- [Rich](https://rich.readthedocs.io/) — Beautiful terminal output
- [Pillow](https://python-pillow.org/) — Image processing

**Thanks To**:

- Google Gemini API team for building such a capable model
- Hackathon organizers for the opportunity
- Open-source community for the amazing tools

---

## 🔗 Links & Resources

### Project Resources

| Resource | URL |
| -------- | --- |
| **GitHub Repository** | [github.com/georgehampton08-rgb/digital-life-narrative-ai](https://github.com/georgehampton08-rgb/digital-life-narrative-ai) |
| **Documentation** | See `README.md`, `ARCHITECTURE.md`, `PRIVACY.md`, `SECURITY.md` |
| **Demo Guide** | [demo/DEMO.md](../demo/DEMO.md) |
| **Contributing Guide** | [CONTRIBUTING.md](../CONTRIBUTING.md) |

### Quick Links for Judges

- **README**: [README.md](../README.md) — Project overview and quick start
- **Architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md) — Technical deep dive
- **Privacy Policy**: [PRIVACY.md](../PRIVACY.md) — Data handling details
- **Security Policy**: [SECURITY.md](../SECURITY.md) — Security posture

### Live Demo

> [!NOTE]
> This is a local-first application with no deployed demo. Follow the [Demo Instructions](#-demo-instructions) above to run locally.

### Demo Video

> [!NOTE]
> Demo video coming soon. For now, follow the [Demo Instructions](#-demo-instructions) to experience the application yourself.

---

## 📋 Submission Form Quick Reference

For copy-paste convenience when filling out submission forms:

### Project Name

```text
Digital Life Narrative AI
```

### One-Sentence Description

```text
An AI-first personal archive analyzer that transforms scattered digital memories into coherent life narratives using Gemini's reasoning capabilities, enabling people to understand and share their own stories.
```

### Category

```text
Productivity / Personal Tools / AI-First Applications
```

### Primary Technology

```text
Google Gemini 1.5 Pro
```

### What Problem Does It Solve?

```text
People have thousands of photos scattered across platforms (Snapchat, Google Photos, local drives) but no coherent understanding of their own history. Traditional organizers sort files but don't extract meaning. We use Gemini to detect semantic life chapters, generate narratives for each period, and create beautiful reports that tell your story — not just display your files.
```

### How Does It Use Gemini?

```text
Gemini is the core intelligence layer. It performs temporal reasoning to detect life transitions (moves, graduations, relationships), generates warm 2-3 paragraph narratives for each chapter, identifies cross-platform behavioral patterns, and synthesizes multi-year executive summaries. Without Gemini, the application explicitly degrades to a basic statistics tool with no real value — proving that AI isn't a feature, it's the entire product.
```

### What Makes It Unique?

```text
AI-first design with intentional degradation in fallback mode, privacy-first architecture (files stay local), cross-platform unification, semantic chapter detection (not calendar-based), and beautiful self-contained HTML output. Unlike photo organizers that sort files or timeline apps that display dates, we extract meaning and generate stories.
```

### GitHub URL

```text
https://github.com/georgehampton08-rgb/digital-life-narrative-ai
```

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Fully functional, ready for evaluation

---

> *"Your memories deserve more than folders. They deserve a story."*
