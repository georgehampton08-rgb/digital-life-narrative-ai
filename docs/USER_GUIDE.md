# Digital Life Narrative AI - Complete User Guide

A step-by-step guide to reconstructing your life narrative from scattered media exports.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Getting Your Media Exports](#getting-your-media-exports)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Understanding Your Report](#understanding-your-report)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you start, make sure you have:

- **Python 3.10 or higher** - [Download here](https://www.python.org/downloads/)
- **Your media exports** from platforms like Snapchat, Google Photos, or local photo folders
- **A Gemini API key** (free) - [Get one here](https://aistudio.google.com/app/apikey)
- **10-30 minutes** depending on your collection size

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai
```

### Step 2: Install Dependencies

```bash
# Using pip (recommended for most users)
pip install -e .

# Or using Poetry (for developers)
poetry install
```

### Step 3: Verify Installation

```bash
organizer --version
# Should show: Digital Life Narrative AI version 0.1.0

organizer --help
# Shows available commands
```

---

## Getting Your Media Exports

The application works with exports from various platforms. Here's how to get them:

### Google Photos (Takeout)

1. Go to [Google Takeout](https://takeout.google.com/)
2. Deselect all, then select **Google Photos** only
3. Choose export frequency: **Export once**
4. File type: **.zip**
5. Click **Create export**
6. Download when ready (can take hours for large collections)
7. Extract the ZIP file:

   ```bash
   unzip takeout-*.zip -d ~/google-photos-export
   ```

### Snapchat Memories

1. Go to [Snapchat Account Portal](https://accounts.snapchat.com/)
2. Click **My Data**
3. Request your data (takes 24-48 hours)
4. Download and extract:

   ```bash
   unzip mydata-*.zip -d ~/snapchat-export
   ```

### Local Photo/Video Folders

No export needed! Just point the tool to any folder containing photos/videos:

- `~/Pictures`
- `~/Downloads/phone-backup`
- External drives
- Old hard drives

---

## Basic Usage

### Step 1: Configure Your API Key

**Option A: Interactive Setup (Recommended)**

```bash
organizer config set-key
# Enter your Gemini API key when prompted (input is hidden)
```

**Option B: Environment File**

```bash
cp .env.example .env
# Edit .env and replace with your key:
# GEMINI_API_KEY=AIzaSy...your-key-here...
```

Verify it worked:

```bash
organizer config show
# Should show: "API key is configured ✓"
```

### Step 2: Scan Your Data (Optional but Recommended)

Before running a full analysis, see what the tool detects:

```bash
organizer scan ~/google-photos-export
```

**Example output:**

```text
Detected in google-photos-export
────────────────────────────────────────
Platform       Confidence   Evidence
google_photos  high         Found Takeout/Google Photos/
local_photos   medium       Found 1,247 .jpg files
```

This helps you verify the tool recognizes your data correctly.

### Step 3: Run Your First Analysis

**Single source:**

```bash
organizer analyze \
  --input ~/google-photos-export \
  --output ./my_google_story
```

**Multiple sources (recommended for complete picture):**

```bash
organizer analyze \
  --input ~/google-photos-export \
  --input ~/snapchat-export \
  --input ~/Pictures/Camera \
  --output ./my_complete_story
```

**What happens:**

1. ✅ Detects export types
2. ✅ Parses metadata from all sources
3. ✅ Sends metadata to Gemini for analysis
4. ✅ AI detects semantic life chapters
5. ✅ AI generates narratives for each chapter
6. ✅ Creates interactive HTML report

**Time:** 2-10 minutes depending on collection size

### Step 4: View Your Report

The tool generates `my_complete_story.html` - open it in any browser:

```bash
# macOS
open my_complete_story.html

# Linux
xdg-open my_complete_story.html

# Windows
start my_complete_story.html
```

---

## Advanced Features

### Privacy Modes

Control how much data is sent to Gemini:

```bash
# Standard mode (default) - city-level locations, truncated captions
organizer analyze --input ~/photos --output ./report

# Strict mode - minimal data, hashed names, 50-char captions
organizer analyze --input ~/photos --output ./report --privacy-mode

# No AI mode - local processing only, statistics only
organizer analyze --input ~/photos --output ./report --no-ai
```

### Chapter Configuration

```bash
# Limit the number of chapters detected
organizer analyze \
  --input ~/photos \
  --output ./report \
  --max-chapters 8
```

### File Organization

Organize your messy photo collection into chapter-based folders:

```bash
# First, generate a JSON report
organizer analyze --input ~/photos --output ./analysis --format json

# Then organize files based on the analysis
organizer organize \
  --input ~/messy-photos \
  --output ~/organized-by-chapters \
  --report ./analysis.json
```

**Result:**

```text
organized-by-chapters/
├── 2018-2019_College_Senior_Year/
│   ├── IMG_1234.jpg
│   └── ...
├── 2019-2020_The_Big_Move/
│   └── ...
└── undo_log.json  # Allows you to reverse if needed
```

### Output Formats

```bash
# HTML only (default)
organizer analyze --input ~/photos --output ./report --format html

# JSON only (for programmatic use)
organizer analyze --input ~/photos --output ./report --format json

# Both formats
organizer analyze --input ~/photos --output ./report --format both
```

---

## Understanding Your Report

When you open the HTML report, you'll see:

### 1. **Header Section**

- Total memories analyzed
- Date range covered
- Number of chapters detected
- Platforms found

### 2. **Executive Summary**

AI-generated overview of your entire life story based on all the data.

### 3. **Interactive Timeline**

Visual representation of your life with clickable chapters.

### 4. **Chapter Cards**

Each chapter includes:

- **Semantic title** (e.g., "College Senior Year" not just "2018-2019")
- **2-3 paragraph narrative** written by Gemini
- **Key themes and events** detected by AI
- **Date range** and memory count
- **Platform breakdown** (how you used different apps)

### 5. **Platform Insights**

AI analysis of how you used different platforms:

- "You used Snapchat for spontaneous daily moments"
- "Google Photos captured curated memories and important events"
- "Local photos show a transition in camera usage patterns"

### 6. **Data Quality Notes**

Transparency about:

- Missing data periods
- Incomplete metadata
- AI confidence levels

---

## Troubleshooting

### "No supported export formats detected"

#### Problem

**Solution:**

1. Make sure you extracted ZIP files
2. Check you're pointing to the correct directory
3. Try the `scan` command to see what it found:

   ```bash
   organizer scan ~/your-folder
   ```

### "API key not configured"

#### Problem

**Solution:**

```bash
organizer config set-key
# Or check if it's set:
organizer config show
```

### "Rate limit exceeded"

**Problem:** Too many API calls in a short time.

**Solution:**

1. Wait a few minutes and try again
2. The tool has automatic retry logic built-in
3. Result is cached, so re-running won't waste quota

### "Memory count is low"

**Problem:** Tool found fewer photos than expected.

**Possible causes:**

- Files aren't in a recognized format (only JPG, PNG, MP4, MOV supported)
- Metadata is missing (the tool skips files without dates)
- Files are in nested subdirectories the parser doesn't check

**Solution:**

```bash
# Use scan to diagnose
organizer scan ~/your-folder

# Check logs
ls ~/.life-story/logs/
cat ~/.life-story/logs/latest.log
```

### Report shows "Fallback Mode" warning

**Problem:** Running without AI (statistics only).

**Meaning:** The tool is working, but without Gemini API access you only get:

- Basic file counts
- Date ranges
- Calendar-year chapters (not semantic)
- No narratives or insights

**Solution:** Configure your API key (see [Step 1](#step-1-configure-your-api-key))

---

## Tips for Best Results

### 1. **Combine Multiple Sources**

The AI gets better context when it sees data from different platforms:

```bash
organizer analyze \
  --input ~/google-photos \
  --input ~/snapchat \
  --input ~/old-phone-backup \
  --output ./complete-story
```

### 2. **Include Location Data**

If your photos have GPS coordinates, the AI uses them to detect:

- Moves between cities
- Travel periods
- Routine locations

### 3. **Don't Over-Filter**

Let the AI see all your data - it's smart about:

- Detecting meaningful vs. mundane photos
- Finding patterns in large datasets
- Ignoring screenshots and memes

### 4. **Review Privacy Settings**

Before sharing your report, check:

```bash
organizer config show
# Review privacy settings
```

Use `--privacy-mode` for maximum anonymization.

### 5. **Cache is Your Friend**

AI analysis is cached locally. If you re-run with:

- Same input data
- Same privacy settings
- Same chapter config

The cached result is used instantly (no API cost, no wait time).

---

## Next Steps

- **Share your report:** The HTML file is self-contained and works offline
- **Organize your files:** Use the `organize` command
- **Get your own API key:** [Free tier](https://aistudio.google.com/app/apikey) for personal use
- **Contribute:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Support

- **Documentation:** [README.md](../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Privacy:** [PRIVACY.md](../PRIVACY.md)
- **Security:** [SECURITY.md](../SECURITY.md)
- **Issues:** [GitHub Issues](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues)
- **Email:** <georgehampton08@gmail.com>

---

**Ready to reconstruct your life story?** 🎬

```bash
organizer analyze --input ~/your-photos --output ./my-life-story
```
