# 🧠 Digital Life Narrative AI — Demo Guide

Welcome! This guide walks you through a quick demonstration of the Digital Life
Narrative AI. You'll see how the application transforms scattered media exports
into a cohesive AI-generated life narrative.

**No real personal data required** — we use synthetic demo data.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Generate Demo Data](#step-1-generate-demo-data)
- [Step 2: Run Analysis (Fallback Mode)](#step-2-run-analysis-statistics-only-mode)
- [Step 3: Run Analysis with AI (Optional)](#step-3-run-analysis-with-ai-full-experience)
- [What to Look For](#what-to-look-for)
- [Troubleshooting](#troubleshooting)
- [Privacy Note](#privacy-note)

---

## Prerequisites

- **Python 3.10+** installed
- Install the project:

  ```bash
  # Using poetry
  poetry install
  
  # Or using pip
  pip install -e .
  ```

- **Gemini API key is optional** — the demo works in fallback mode without one

---

## Step 1: Generate Demo Data

Run the demo data generator to create synthetic media exports:

```bash
python -m demo.generate_demo_data --output ./demo_data
```

This creates:

- **~15-20 synthetic "memories"** across multiple years
- Fake Snapchat-like export structure
- Fake Google Photos-like export structure  
- Local photos with EXIF data

The data spans **2018-2023** to give the AI enough material to detect life chapters.

---

## Step 2: Run Analysis (Statistics-Only Mode)

Run the analyzer without AI to see the fallback experience:

```bash
organizer analyze \
  --input ./demo_data \
  --output ./demo_report_fallback.html \
  --no-ai
```

Open `demo_report_fallback.html` in your browser.

**What you'll see:**

- ✅ Year-based "chapters" (not semantic)
- ✅ Statistics and counts
- ✅ Timeline visualization
- ⚠️ Clear messaging that AI is unavailable

This demonstrates **graceful degradation** — the app is still useful without AI,
but the core value (narrative reconstruction) is missing.

---

## Step 3: Run Analysis with AI (Full Experience)

To see the full AI-powered experience:

### 1. Get a Gemini API key

Visit [Google AI Studio](https://makersuite.google.com/app/apikey) and create an API key.

### 2. Configure the key

```bash
organizer config set-key
# Enter your API key when prompted
```

### 3. Run analysis with AI

```bash
organizer analyze \
  --input ./demo_data \
  --output ./demo_report_ai.html
```

### 4. Open the report

Open `demo_report_ai.html` in your browser.

**What you'll see with AI:**

- ✨ Meaningful chapter titles (e.g., "The New Beginnings", "Growth Period")
- ✨ AI-written narratives for each chapter
- ✨ Cross-platform behavior analysis
- ✨ Pattern detection
- ✨ Executive life story summary

**Compare this to the fallback report** — the difference demonstrates why AI is the core value proposition.

---

## What to Look For

### In the HTML Report

#### 1. Header Section

- Title and date range
- Total memories analyzed
- AI model attribution (or fallback indicator)

#### 2. Executive Summary *(AI mode only)*

- Cohesive narrative of the life story
- Themes and patterns identified

#### 3. Timeline

- Visual representation of chapters
- Click markers to navigate

#### 4. Chapters

- **AI mode:** Meaningful titles like "The Adventure Years"
- **Fallback:** Generic "Year 2020" titles
- Narrative paragraphs
- Key events and insights
- Theme tags

#### 5. Platform Insights *(AI mode only)*

- How different platforms were used
- Usage patterns over time

#### 6. Footer Metadata

- Generation timestamp
- Model version used

---

## Troubleshooting

### "No sources detected"

- Ensure demo data was generated: `python -m demo.generate_demo_data --output ./demo_data`
- Check the path is correct
- Run `organizer scan ./demo_data` to verify detection

### "API key not configured"

- Run `organizer config set-key`
- Or use `--no-ai` flag for fallback mode

### Report doesn't open

- Check the output path exists
- Try opening the file directly in browser
- Verify the file was created: `ls -la demo_report*.html`

### Import errors

- Ensure you've run `pip install -e .` from the project root
- Check Python version: `python --version` (needs 3.10+)

---

## Privacy Note

The demo data is **entirely synthetic**:

- ❌ No real personal information
- ❌ No real locations (uses "City A", "City B")
- ❌ No real photos (generated colored squares)
- ✅ Safe to share or commit to repositories
- ✅ Safe to demo publicly

---

## Quick Commands Reference

```bash
# Generate demo data
python -m demo.generate_demo_data --output ./demo_data

# Run fallback analysis (no API key needed)
organizer analyze --input ./demo_data --output demo_fallback.html --no-ai

# Set API key
organizer config set-key

# Run AI analysis
organizer analyze --input ./demo_data --output demo_ai.html

# Scan for sources
organizer scan ./demo_data

# Show config
organizer config show
```

---

*Happy demoing! 🎬*
