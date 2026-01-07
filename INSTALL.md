# Installation Guide 🚀

Get Digital Life Narrative AI up and running on your machine. This guide covers installation for judges, users, and developers.

> [!TIP]
> **Estimated installation time**: 2-5 minutes.

---

## 📋 Overview

Digital Life Narrative AI is a local-first CLI application. It processes your media exports and generates self-contained HTML reports.

- **Minimum Requirement**: Python 3.10+ (3.11 recommended)
- **AI Features**: Requires a [Gemini API Key](https://aistudio.google.com/app/apikey) (Optional: Fallback mode works without one)
- **Platforms**: Windows, macOS, and Linux are all supported.

---

## 🏗️ Prerequisites

Before you begin, ensure you have the following installed:

### For Python Installation (Recommended)

- **Python 3.10+**: [Download here](https://www.python.org/downloads/)
- **Git**: For cloning the repository
- **pip**: Usually comes with Python

### For Docker Installation

- **Docker Desktop** or **Docker Engine**: [Get Docker](https://docs.docker.com/get-docker/)

### System-Specific Notes

- **macOS**: Install Xcode Command Line Tools (`xcode-select --install`) for some dependencies.
- **Windows**: We recommend using **PowerShell** or **WSL2**.
- **Linux (Ubuntu/Debian)**: You may need imaging libraries:

  ```bash
  sudo apt-get update && sudo apt-get install -y libjpeg-dev zlib1g-dev
  ```

---

## 🔌 Option A: Python Installation (Recommended)

This is the fastest and most flexible way to run the application.

### 1. Clone the Repository

```bash
# Clone the repo (shallow clone for speed)
git clone --depth=1 https://github.com/georgehampton08-rgb/digital-life-narrative-ai.git
cd digital-life-narrative-ai
```

### 2. Create a Virtual Environment

We recommend using a virtual environment to keep your system clean.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install the package in editable mode
pip install -e .

# Optional: Install development dependencies (for testing/linting)
# pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
organizer --version
organizer --help
```

---

## 🐳 Option B: Docker Installation

Perfect if you want to run the tool in a sandbox without installing Python locally.

### 1. Build the Image

```bash
docker build -t life-story-reconstructor .
```

### 2. Verify the Build

```bash
docker run --rm life-story-reconstructor --help
```

### 3. Run with Your Data

To process files, you must mount your local data folders into the container.

```bash
# Multi-line version for readability
docker run --rm \
  -v "/path/to/your/input:/data/input" \
  -v "/path/to/save/output:/data/output" \
  -e GEMINI_API_KEY="your-key-here" \
  life-story-reconstructor \
  analyze -i /data/input -o /data/output/report
```

> [!NOTE]
> Replace `/path/to/your/input` and `/path/to/save/output` with absolute paths on your host machine.

---

## 🔑 Gemini API Key Setup

AI-powered narratives require a Google Gemini API key.

### 1. Get a Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a new API key. The **Free Tier** is more than sufficient for testing and personal use.

### 2. Configure the CLI

Choose **one** of the following methods:

#### Method 1: Interactive (Recommended)

```bash
organizer config set-key
# Paste your key when prompted
```

#### Method 2: Environment Variable

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-key-here"
```

### 3. Verify Privacy Settings

```bash
organizer config show
```

This will confirm your key is "configured" without showing the secret value.

---

## 🧪 Quick Smoke Test

Run these tests in order to ensure everything is perfect.

### Method 0: The Quick Path (Recommended for Judges)

If you are on macOS or Linux, you can verify your entire installation with one command:

```bash
bash scripts/run_demo.sh
```

This script will check your environment, generate demo data, and run a full analysis. If you have an API key configured, it will use it; otherwise, it will run in fallback mode.

### Test 1: CLI Accessibility

**Command**: `organizer --help`
**Success**: You see the help text and list of commands (`analyze`, `scan`, `config`, etc.).

### Test 2: Source Detection

**Command**: `organizer scan ./demo`
**Success**: The tool lists detected platforms (like "local_files" or "google_photos").

### Test 3: Demo Data Generation

**Command**: `python -m demo.generate_demo_data --output ./demo_data`
**Success**: A folder named `demo_data` is created with synthetic files.

### Test 4: Fallback Analysis (No AI)

**Command**: `organizer analyze --input ./demo_data --output ./test_report --no-ai`
**Success**: A file named `test_report.html` is generated successfully.

### Test 5: Full AI Analysis (Requires Key)

**Command**: `organizer analyze --input ./demo_data --output ./ai_report`
**Success**: A report is generated with rich narratives and life chapters.

---

## 🛠️ Troubleshooting

### "Command not found: organizer"

- **Solution**: Ensure your virtual environment is activated. If it still fails, try `pip install -e .` again from the root directory.

### Python Version Mismatch

- **Symptoms**: Syntax errors on startup.
- **Solution**: Run `python --version`. If it's below 3.10, install a newer version from [python.org](https://www.python.org/).

### Pillow (Imaging) Installation Fails

- **Symptoms**: Errors mentioning `zlib` or `libjpeg`.
- **Solution**: Install system dependencies mentioned in the [Prerequisites](#system-specific-notes) section.

### Docker volume mount issues

- **Symptoms**: "No such file or directory" inside the container.
- **Solution**: Always use **absolute paths** for volume mounts (e.g., `C:/Users/name/data` or `/home/user/data`).

### Permission Denied (Linux/macOS)

- **Solution**: You may need to grant execution permissions to the `organizer` script if you aren't using a virtual environment (though venv is highy recommended).

---

## 🧼 Uninstallation

### Remove Python Installation

```bash
# Deactivate venv if active
deactivate

# Delete the folder
rm -rf digital-life-narrative-ai
```

### Remove Docker Image

```bash
docker rmi life-story-reconstructor
```

### Cleanup Local Config

Configuration is stored in your user home directory:

- **macOS/Linux**: `~/.life-story/`
- **Windows**: `C:\Users\<User>\.life-story\`
You can delete these folders to completely remove all settings and cached AI results.

---

## 🆘 Need Help?

- Check the detailed logs at: `~/.life-story/logs/`
- Review the [ARCHITECTURE.md](./ARCHITECTURE.md) for technical deep-dives.
- Open an issue on the [GitHub repository](https://github.com/georgehampton08-rgb/digital-life-narrative-ai).
