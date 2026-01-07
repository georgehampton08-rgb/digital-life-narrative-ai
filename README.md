# Digital Life Narrative AI 🧠📸

> AI-powered analysis of your scattered media exports to reconstruct your life's narrative

[![CI](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

> [!TIP]
> **Hackathon Judges**: See [`docs/JUDGES_QUICKSTART.md`](./docs/JUDGES_QUICKSTART.md) for the fastest evaluation path (< 2 minutes). A shared API key is pre-configured for you!

## What is this?

### The Problem

Your life is scattered across:

- 📱 Old phones and cloud backups
- 👻 Snapchat Memories exports
- 📸 Google Photos Takeout archives
- 📘 Facebook data downloads
- 💾 Random hard drives and folders

Traditional photo organizers just sort by date. They don't understand *meaning*.

### The Solution

Digital Life Narrative AI uses **Google's Gemini AI** to analyze your media metadata and reconstruct your life story:

- 🎯 **Life Chapters**: "The Chicago Years", "Starting College", "The Pandemic"
- 📝 **AI-Written Narratives**: Rich descriptions of each life period
- 🔍 **Cross-Platform Analysis**: How you used different apps reflects different aspects of your life
- 📊 **Data Gap Detection**: AI speculates on what happened during quiet periods
- 🎨 **Beautiful Reports**: Interactive HTML timeline you can share

### AI-First Design

> ⚠️ **This is an AI-first product.** Without AI, it's just a metadata parser.

The entire value proposition depends on Gemini's ability to:

- Infer context from timestamps, locations, and patterns
- Detect life transitions and turning points
- Generate coherent narratives from fragmented data
- Understand the *meaning* behind the metadata

We don't pretend rules can do what AI does. In fallback mode, you get statistics — not stories.

---

## Key Features

| Feature | Description |
| ------- | ----------- |
| 🧠 **Life Chapter Detection** | AI identifies meaningful periods: moves, jobs, relationships, growth |
| ✍️ **AI-Written Narratives** | 2-3 paragraph stories for each chapter with key events and themes |
| 📱 **Platform Behavior Analysis** | "You used Snapchat for spontaneous moments, Google Photos for memories" |
| 📭 **Data Gap Detection** | "No data Mar-May 2020... possibly the pandemic adjustment period" |
| 🎨 **Beautiful HTML Reports** | Self-contained, interactive, dark mode, shareable |
| 📁 **File Organization** | Optionally organize files into chapter-named folders |
| 🔒 **Privacy-Focused** | Your files never leave your computer |

---

## Supported Platforms

| Platform | Status | Notes |
| -------- | ------ | ----- |
| 👻 Snapchat | ✅ Supported | Memories, chat media, location history |
| 📸 Google Photos | ✅ Supported | Takeout exports with JSON sidecars |
| 💾 Local Photos | ✅ Supported | Any folder with images/videos |
| 📘 Facebook | 🚧 Coming Soon | Posts, photos, location history |
| 📷 Instagram | 🚧 Coming Soon | Posts, stories, reels |
| ☁️ OneDrive | 🚧 Coming Soon | Camera roll backups |

---

## Quick Start

### Installation

For detailed instructions including Docker and system-specific prerequisites, see the **[INSTALL.md](./INSTALL.md)** guide.

```bash
# Fastest path:
git clone --depth=1 https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai
pip install -e .
```

### Set Up Your API Key

**For Hackathon Judges**: You'll receive a shared Gemini API key separately. To configure it:

```bash
cp .env.example .env
# Edit .env and paste the provided key after GEMINI_API_KEY=
```

**For Your Own Use**: Get your personal API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then choose one of these methods:

#### Option 1: Interactive setup (Recommended)

```bash
organizer config set-key
# Enter your API key when prompted (input is hidden)
```

#### Option 2: Environment file (For judges/CI)

```bash
# Copy the template
cp .env.example .env

# Edit .env and replace with your personal key
# GEMINI_API_KEY=AIzaSy...YourPersonalKey...
```

> [!NOTE]
> Without an API key, the application runs in **"statistics-only mode"** with basic file counts and dates, but no AI narratives or semantic analysis.

### One-Shot Demo (Judges 🏆)

If you are on macOS or Linux, the fastest way to see the project in action is our one-shot demo script:

```bash
bash scripts/run_demo.sh
```

This script handles environment detection, dependency verification, demo data generation, and runs the analysis automatically.

### Using With Your Own Data

**See the complete guide**: [`docs/USER_GUIDE.md`](./docs/USER_GUIDE.md)

Quick example with your real media exports:

```bash
# Single source
organizer analyze --input ~/google-photos-export --output ./my_story

# Multiple sources (recommended)
organizer analyze \
  --input ~/google-photos-export \
  --input ~/snapchat-export \
  --input ~/Pictures \
  --output ./complete_story
```

The guide covers:

- How to get exports from Google Photos, Snapchat, etc.
- Privacy modes and configuration
- File organization features
- Understanding your generated report

### Run Analysis

```bash
# Analyze your exports
organizer analyze -i ~/Downloads/takeout -i ~/Snapchat -o ./my_life_story

# View what's detected first
organizer scan ~/Downloads/takeout

# Run without AI (statistics only)
organizer analyze -i ~/exports -o ./report --no-ai
```

### Open Your Report

```bash
# Opens in your default browser
open ./my_life_story.html
```

---

## Example Output

### Sample Chapter Titles

- 🎓 *"2018-2019: College Senior Year"*
- 🏙️ *"2019-2020: The Chicago Move"*
- 🏠 *"2020-2021: Pandemic at Home"*
- ✈️ *"2021-2022: Travel and Recovery"*
- 🚀 *"2023-Present: New Beginnings"*

### Report Preview

```text
┌─────────────────────────────────────────────────────────────────┐
│  📖 Your Life Story                                              │
│  An AI-powered narrative of your journey                        │
│                                                                  │
│  📊 2,847 memories · 5 chapters · 6 years · 3 platforms         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📅 Life Timeline                                                │
│  ●────●────●────●────●                                          │
│  2018  2019  2020  2021  2022                                   │
│                                                                  │
│  📚 Chapters                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 🎓 College Senior Year (2018-2019)                      │    │
│  │ Tags: #education #friends #transition                   │    │
│  │                                                         │    │
│  │ "The final year of college was marked by a flurry      │    │
│  │ of activity — graduation preparations, last moments    │    │
│  │ with roommates, and the anxiety of what comes next..." │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Privacy & Security

### 🔒 Your Data Stays Local

- **Files never leave your computer** — we only read metadata
- **Only anonymized metadata** goes to Gemini (dates, general locations, platform names)
- **No actual images or videos** are ever transmitted
- **No tracking, analytics, or telemetry** — we're open source

### 🛡️ Privacy Mode

For sensitive data, enable strict privacy:

```bash
organizer analyze -i ~/exports -o ./report --privacy-mode
```

This additionally:

- Hashes people's names
- Truncates captions
- Generalizes locations

### 🌐 Local-Only Mode

For maximum privacy, run without any AI:

```bash
organizer analyze -i ~/exports -o ./report --no-ai
```

Zero network calls. Statistical analysis only.

📄 **[Full Privacy Documentation →](PRIVACY.md)**

---

## Requirements

- **Python 3.10+**
- **Gemini API Key** — Free tier available at [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Your data exports** — Download from each platform's data export feature

### Optional Dependencies

- `keyring` — For secure API key storage in system keyring
- `Pillow` — For EXIF extraction from images

---

## CLI Reference

```bash
# Main commands
organizer analyze       # Full AI-powered analysis
organizer scan          # Quick source detection
organizer organize      # Organize files into folders
organizer config        # Manage configuration

# Global options
organizer --version     # Show version
organizer --verbose     # Enable debug output
organizer --help        # Show help

# Analyze options
organizer analyze -i PATH           # Input directory (can specify multiple)
organizer analyze -o PATH           # Output path for report
organizer analyze --format html     # Output format (html/json/both)
organizer analyze --no-ai           # Skip AI, statistics only
organizer analyze --privacy-mode    # Strict privacy filtering
organizer analyze --max-chapters N  # Limit chapter detection

# Config commands
organizer config set-key            # Set Gemini API key
organizer config show               # Show current config
organizer config reset              # Reset to defaults
```

---

## Caching

Analysis results are cached locally for faster repeat runs. The cache:

- Is stored in your system's cache directory (`~/.cache/life-story-reconstructor` on Linux, `~/Library/Caches/...` on macOS, `%LOCALAPPDATA%\...` on Windows)
- Is specific to your machine (won't work if you copy the repo elsewhere)
- Is automatically invalidated when your media changes or analysis settings change
- Can be safely deleted at any time (just triggers recomputation on next run)
- Is never committed to Git

To disable caching, set `ai.cache_enabled: false` in your config file.

---

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai

# Install with dev dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ --override-ini="addopts="

# Run specific test file
python -m pytest tests/test_core_models.py -v --override-ini="addopts="

# Run with verbose output
python -m pytest tests/ -v --tb=short --override-ini="addopts="
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check (optional)
mypy src
```

### Project Structure

```text
digital-life-narrative-ai/
├── src/
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration & API key management
│   ├── detection.py             # Platform detection
│   ├── core/
│   │   ├── memory.py            # Universal Memory data model
│   │   ├── timeline.py          # Timeline aggregation & gap analysis
│   │   ├── privacy.py           # Privacy gate & content filtering
│   │   └── safety.py            # Safety settings & sensitivity levels
│   ├── parsers/
│   │   ├── base.py              # BaseParser & registry
│   │   ├── pipeline.py          # Parsing orchestration
│   │   ├── snapchat.py          # Snapchat Memories parser
│   │   ├── google_photos.py     # Google Takeout parser
│   │   └── local_files.py       # Local photos parser
│   ├── ai/
│   │   ├── client.py            # Gemini API wrapper with retry logic
│   │   ├── life_analyzer.py     # Main AI analysis engine
│   │   ├── fallback.py          # Statistics-only fallback analyzer
│   │   ├── prompts.py           # Prompt templates for Gemini
│   │   ├── cache.py             # Machine-local analysis cache
│   │   ├── content_filter.py    # AI content safety filtering
│   │   ├── disclosure.py        # AI disclosure management
│   │   └── usage_tracker.py     # API usage & cost tracking
│   ├── output/
│   │   └── html_report.py       # Self-contained HTML report generator
│   └── cli/
│       └── main.py              # Click CLI commands
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   ├── test_core_models.py      # Core data model tests
│   ├── test_memory.py           # Memory model tests
│   ├── test_detection_and_parsers.py  # Parser tests
│   ├── test_ai_and_safety.py    # AI & safety tests
│   ├── test_cli_and_report.py   # CLI & report tests
│   └── test_src_ai_client.py    # AI client tests
├── demo/
│   ├── DEMO.md                  # Demo walkthrough
│   └── generate_demo_data.py    # Synthetic data generator
├── pyproject.toml               # Poetry configuration
├── ARCHITECTURE.md              # Technical documentation
├── PRIVACY.md                   # Privacy documentation
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Contributing

Contributions are welcome! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for:

- Development setup instructions
- Code standards and style guide
- Testing guidelines
- Pull request process

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/digital-life-narrative-ai.git
cd digital-life-narrative-ai

# Install and test
poetry install
python -m pytest tests/ --override-ini="addopts="

# Make changes, then submit a PR!
```

### Adding a New Platform

See [ARCHITECTURE.md](ARCHITECTURE.md#adding-new-platforms) for a step-by-step guide.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [Google Gemini](https://ai.google.dev/) for AI-powered narrative generation
- Uses [Click](https://click.palletsprojects.com/) for CLI
- Uses [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- Uses [Pydantic](https://docs.pydantic.dev/) for data validation
- Uses [Jinja2](https://jinja.palletsprojects.com/) for HTML templating

---

## Author

**George Hampton** — [georgehampton08@gmail.com](mailto:georgehampton08@gmail.com)

---

*Turn your scattered memories into a story worth reading.*
