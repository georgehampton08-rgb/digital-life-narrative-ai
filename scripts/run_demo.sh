#!/usr/bin/env bash

# Digital Life Narrative AI - One-Shot Demo Script
#
# Run this script with: bash scripts/run_demo.sh
# Or make executable: chmod +x scripts/run_demo.sh && ./scripts/run_demo.sh

set -e
set -u
set -o pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helpers ---
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Cleanup ---
cleanup() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        error "Demo script encountered an error (Exit code: $EXIT_CODE). Check the output above for details."
    fi
}
trap cleanup EXIT

# --- Start ---
START_TIME=$(date +%s)
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}       Digital Life Narrative AI - Demo Runner       ${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# --- Environment Detection ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

info "Detecting Python environment..."
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    error "Python not found. Please install Python 3.10+."
fi

PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python version $PY_MAJOR.$PY_MINOR is too old. Please use Python 3.10+."
fi
info "Found Python $PY_MAJOR.$PY_MINOR"

# --- Project Installation ---
info "Checking project installation..."
if ! command -v organizer &> /dev/null; then
    warning "The 'organizer' command is not in your PATH."
    info "Attempting to install the project in editable mode..."
    $PYTHON -m pip install -e . || error "Failed to install the project. Please follow INSTALL.md instructions."
    success "Project installed successfully."
fi

# --- API Key Check ---
HAS_API_KEY=false
info "Checking for Gemini API key..."
if [ -n "${GEMINI_API_KEY:-}" ]; then
    HAS_API_KEY=true
elif organizer config show | grep -q "API key is configured ✓"; then
    HAS_API_KEY=true
fi

if [ "$HAS_API_KEY" = true ]; then
    success "Gemini API key found! Full AI analysis enabled."
else
    warning "No API key found. Falling back to statistics-only mode."
    info "Note: Configure your key in .env or run: organizer config set-key"
fi

# --- Directory Setup ---
OUTPUT_DIR="${PROJECT_ROOT}/output"
DEMO_DATA_DIR="${PROJECT_ROOT}/demo_data"
DEMO_REPORT_BASE="${OUTPUT_DIR}/demo_report"
DEMO_REPORT="${DEMO_REPORT_BASE}.html"
mkdir -p "$OUTPUT_DIR"

# --- Demo Data Generation ---
if [ -d "$DEMO_DATA_DIR" ]; then
    info "Using existing demo data at $DEMO_DATA_DIR"
else
    info "Generating synthetic demo data..."
    $PYTHON -m demo.generate_demo_data --output "$DEMO_DATA_DIR"
    success "Demo data generated at $DEMO_DATA_DIR"
fi

# --- Analysis Execution ---
info "Running life narrative analysis..."
if [ "$HAS_API_KEY" = true ]; then
    organizer analyze --input "$DEMO_DATA_DIR" --output "$DEMO_REPORT_BASE"
else
    organizer analyze --input "$DEMO_DATA_DIR" --output "$DEMO_REPORT_BASE" --no-ai
fi

# --- Results ---
END_TIME=$(date +%s)
echo ""
echo -e "${BLUE}====================================================${NC}"
success "Demo complete! (Took $((END_TIME - START_TIME)) seconds)"
echo ""
echo "Your life narrative report is ready at:"
echo -e "  ${GREEN}$DEMO_REPORT${NC}"
echo ""

if [ "$HAS_API_KEY" = false ]; then
    warning "You are viewing a [fallback mode] report without AI narratives."
    echo "To see the full magic of Gemini:"
    echo "  1. Get a key at: https://aistudio.google.com/app/apikey"
    echo "  2. Run: organizer config set-key"
    echo "  3. Re-run this script: bash scripts/run_demo.sh"
    echo ""
fi

echo "To view the report:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  Run: open $DEMO_REPORT"
    echo ""
    read -p "Open report in browser now? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        open "$DEMO_REPORT"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open &> /dev/null; then
        echo "  Run: xdg-open $DEMO_REPORT"
        echo ""
        read -p "Open report in browser now? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            xdg-open "$DEMO_REPORT"
        fi
    else
        echo "  Open the file in any web browser."
    fi
else
    echo "  Open the file in any web browser."
fi

echo ""
echo -e "${BLUE}====================================================${NC}"
