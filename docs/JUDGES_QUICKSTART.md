# Quick Start for Hackathon Judges 🏆

Thank you for reviewing **Digital Life Narrative AI**! This guide will get you up and running in under 2 minutes.

---

## ⚡ TL;DR - The Fastest Path

```bash
# Clone the repository
git clone https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai

# Run the one-shot demo (handles everything automatically)
bash scripts/run_demo.sh
```

That's it! The demo script will:

- ✅ Check your Python environment (requires 3.10+)
- ✅ Install the project automatically
- ✅ Set up the pre-configured API key
- ✅ Generate synthetic demo data
- ✅ Run the full AI analysis
- ✅ Open the interactive HTML report in your browser

---

## 🔑 About the API Key

**A shared Gemini API key is pre-configured** in `.env.example` specifically for hackathon evaluation. You don't need to get your own key for testing.

The key is: `AIzaSyCezXkLMSQ0bXFUCMxZqVTvUXafxPQRVxc`

This is already set in the template, so the demo script will automatically use it.

---

## 🎯 What to Look For

### AI-First Design

This project demonstrates that **Gemini is essential infrastructure**, not an optional feature:

- **With AI**: You'll see semantic life chapters ("College Years", "The Big Move"), rich 2-3 paragraph narratives, cross-platform behavioral insights, and gap speculation.
- **Without AI**: Intentionally degraded to basic statistics (file counts, date ranges) to prove the point.

### Key Features to Evaluate

1. **Semantic Chapter Detection** - AI identifies meaningful life periods, not just calendar years
2. **Narrative Generation** - Gemini writes warm, coherent stories for each chapter
3. **Cross-Platform Intelligence** - Understands how you use different apps differently
4. **Privacy Architecture** - Files stay local, only metadata sent to AI
5. **Graceful Degradation** - Clear fallback mode when no API key exists

---

## 📊 Understanding the Demo

The demo generates synthetic media exports simulating:

- **Snapchat Memories** (spontaneous moments)
- **Google Photos Takeout** (curated memories)
- **Local photo folders** (generic media)

The AI analyzes this data and produces an **interactive HTML report** showing your reconstructed life narrative.

---

## 🐛 Troubleshooting

### "Python not found"

Install Python 3.10+ from [python.org](https://www.python.org/downloads/)

### "organizer: command not found"

The demo script auto-installs the project. If it fails, manually run:

```bash
pip install -e .
```

### Want to test without AI?

```bash
bash scripts/run_demo.sh
# The script will detect the missing key and run in fallback mode
```

To see the difference, compare the fallback report with the AI-powered one.

---

## 📁 Key Files to Review

| File | Purpose |
| ---- | ------- |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Technical deep-dive into system design |
| [`docs/HACKATHON_SUBMISSION.md`](../docs/HACKATHON_SUBMISSION.md) | Comprehensive pitch document |
| [`PRIVACY.md`](../PRIVACY.md) | Data handling and privacy commitments |
| [`SECURITY.md`](../SECURITY.md) | Security architecture and threat model |
| [`src/ai/client.py`](../src/ai/client.py) | Gemini API wrapper implementation |
| [`src/ai/life_analyzer.py`](../src/ai/life_analyzer.py) | Core AI narrative engine |

---

## 🎬 Alternative: Manual Setup

If you prefer to set up manually instead of using the demo script:

```bash
# 1. Install dependencies
pip install -e .

# 2. Set up API key
cp .env.example .env
# The evaluation key is already configured!

# 3. Generate demo data
python -m demo.generate_demo_data --output ./demo_data

# 4. Run analysis
organizer analyze --input ./demo_data --output ./my_report

# 5. View report
open my_report.html  # macOS
# or: xdg-open my_report.html  # Linux
# or: start my_report.html  # Windows
```

---

## ❓ Questions or Issues?

- **GitHub Issues**: [Open an issue](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues)
- **Email**: <georgehampton08@gmail.com>
- **Documentation**: See [`README.md`](../README.md) for comprehensive docs

---

## 🌟 Why This Project Stands Out

1. **True AI-First Design**: The application is architected around Gemini from the ground up, not bolted on
2. **Privacy-Conscious**: Local-first processing, metadata-only AI requests
3. **Production-Ready**: Comprehensive error handling, caching, security measures
4. **Beautiful Output**: Self-contained, interactive HTML reports
5. **Hackathon Polish**: CI/CD, Docker, extensive documentation, one-shot demo

Thank you for your time and consideration! 🙏
