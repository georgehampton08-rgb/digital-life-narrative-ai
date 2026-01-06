# Digital Life Narrative AI 🧠📸

> AI-powered analysis of your scattered media exports to reconstruct your life's narrative

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

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

```bash
# Using pip
pip install digital-life-narrative-ai

# Or using Poetry (recommended for development)
git clone https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai
poetry install
```

### Set Up Your API Key

Get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey), then:

```bash
organizer config set-key
# Enter your API key when prompted (input is hidden)
```

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
pytest

# Run with coverage
pytest --cov=organizer --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black organizer tests

# Lint
ruff check organizer tests

# Type check
mypy organizer
```

### Project Structure

```text
digital-life-narrative-ai/
├── organizer/
│   ├── __init__.py         # Package exports
│   ├── models.py           # Pydantic data models
│   ├── config.py           # Configuration & API key management
│   ├── detection.py        # Platform detection
│   ├── cli.py              # Click CLI
│   ├── report.py           # HTML/JSON report generation
│   ├── organizer.py        # File organization
│   ├── parsers/
│   │   ├── base.py         # BaseParser & registry
│   │   ├── snapchat.py     # Snapchat parser
│   │   ├── google_photos.py # Google Takeout parser
│   │   └── local.py        # Local photos parser
│   ├── ai/
│   │   ├── client.py       # Gemini API wrapper
│   │   ├── life_analyzer.py # Main analysis engine
│   │   └── fallback.py     # Statistics-only fallback
│   └── utils/
│       ├── logging.py      # Logging configuration
│       └── hashing.py      # File hashing utilities
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_models.py      # Model tests
│   ├── test_parsers.py     # Parser tests
│   ├── test_ai.py          # AI tests (mocked)
│   └── test_cli.py         # CLI tests
├── pyproject.toml          # Poetry configuration
├── ARCHITECTURE.md         # Technical documentation
├── PRIVACY.md              # Privacy documentation
└── README.md               # This file
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black . && ruff check .`)
6. Submit a Pull Request

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
