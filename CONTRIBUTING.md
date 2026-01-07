# Contributing to Digital Life Narrative AI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Adding New Platforms](#adding-new-platforms)
- [Pull Request Process](#pull-request-process)

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Poetry** (recommended) or pip
- **Git**

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/digital-life-narrative-ai.git
cd digital-life-narrative-ai
```

---

## Development Setup

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "from src.config import AppConfig; print('Setup OK!')"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install editable package
pip install -e .
```

---

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

### 2. Make Your Changes

Edit the files in `src/` for source code changes.

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/ --override-ini="addopts=" -v

# Run specific tests
python -m pytest tests/test_core_models.py --override-ini="addopts=" -v
```

### 4. Format and Lint

```bash
# Format code
black src tests

# Lint
ruff check src tests --fix

# Type check (optional)
mypy src
```

---

## Code Standards

### Style Guide

- **Formatter**: [Black](https://github.com/psf/black) with default settings
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for public functions/classes

### Example

```python
def process_memories(
    memories: list[Memory],
    config: AnalysisConfig | None = None,
) -> LifeStoryReport:
    """Process a list of memories into a life story report.
    
    Args:
        memories: List of Memory objects to analyze.
        config: Optional analysis configuration. Uses defaults if not provided.
    
    Returns:
        LifeStoryReport containing the complete analysis.
    
    Raises:
        ValueError: If memories list is empty.
    """
    if not memories:
        raise ValueError("At least one memory is required")
    # ...
```

### Project Structure

```text
src/
├── core/           # Data models (Memory, Timeline, Privacy)
├── parsers/        # Platform-specific parsers
├── ai/             # AI integration (client, analyzer, prompts)
├── output/         # Report generators
├── cli/            # CLI commands
└── config.py       # Configuration management
```

---

## Testing

### Test Organization

| Test File | Coverage |
| --------- | -------- |
| `test_core_models.py` | Core data models and enums |
| `test_memory.py` | Memory model edge cases |
| `test_detection_and_parsers.py` | Source detection and parsers |
| `test_ai_and_safety.py` | AI integration and content safety |
| `test_cli_and_report.py` | CLI commands and HTML reports |
| `test_src_ai_client.py` | AI client with mocked Gemini API |

### Writing Tests

```python
import pytest
from src.core.memory import Memory, SourcePlatform, MediaType

class TestMemoryCreation:
    """Tests for Memory object creation."""
    
    def test_minimal_valid_memory(self):
        """Memory can be created with minimal required fields."""
        memory = Memory(source_platform=SourcePlatform.UNKNOWN)
        assert memory.id is not None
        assert memory.source_platform == SourcePlatform.UNKNOWN
    
    def test_memory_with_location(self, sample_location):
        """Memory correctly stores location data."""
        memory = Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            location=sample_location,
        )
        assert memory.location.city == sample_location.city
```

### Running Tests

```bash
# All tests
python -m pytest tests/ --override-ini="addopts="

# With verbose output
python -m pytest tests/ -v --tb=short --override-ini="addopts="

# Specific test class
python -m pytest tests/test_memory.py::TestMemoryCreation --override-ini="addopts="
```

---

## Adding New Platforms

To add support for a new platform (e.g., TikTok, Instagram), follow these steps:

### 1. Add Platform to Enum

```python
# src/core/memory.py
class SourcePlatform(str, Enum):
    # ... existing platforms ...
    TIKTOK = "tiktok"
```

### 2. Create Parser

```python
# src/parsers/tiktok.py
from src.parsers.base import BaseParser, ParserResult
from src.core.memory import Memory, SourcePlatform

class TikTokParser(BaseParser):
    """Parser for TikTok data exports."""
    
    PLATFORM = SourcePlatform.TIKTOK
    
    @classmethod
    def can_parse(cls, path: Path) -> bool:
        """Check if path looks like a TikTok export."""
        # Look for TikTok-specific files
        return (path / "Video" / "Videos.json").exists()
    
    def parse(self, path: Path) -> ParserResult:
        """Parse TikTok export into Memory objects."""
        # Implementation...
```

### 3. Register Parser

```python
# src/parsers/__init__.py
from src.parsers.tiktok import TikTokParser

PARSERS = [
    # ... existing parsers ...
    TikTokParser,
]
```

### 4. Add Tests

```python
# tests/test_parsers.py
class TestTikTokParser:
    def test_can_parse_valid_export(self, tiktok_export_fixture):
        assert TikTokParser.can_parse(tiktok_export_fixture)
    
    def test_parse_extracts_memories(self, tiktok_export_fixture):
        parser = TikTokParser()
        result = parser.parse(tiktok_export_fixture)
        assert len(result.memories) > 0
```

See [ARCHITECTURE.md](ARCHITECTURE.md#adding-new-platforms) for detailed guidance.

---

## Pull Request Process

### 1. Before Submitting

- [ ] Tests pass: `python -m pytest tests/ --override-ini="addopts="`
- [ ] Code formatted: `black src tests`
- [ ] Linting clean: `ruff check src tests`
- [ ] Docstrings added for new public functions
- [ ] Documentation updated if needed

### 2. PR Title Format

Use conventional commit format:

- `feat: Add TikTok parser`
- `fix: Handle missing EXIF timestamps`
- `docs: Update privacy documentation`
- `refactor: Simplify cache validation`
- `test: Add timeline gap detection tests`

### 3. PR Description

Include:

- **What**: Brief description of changes
- **Why**: Motivation for the change
- **How**: Key implementation details
- **Testing**: How you verified the changes

### 4. Review Process

- PRs require at least one approval
- Address feedback promptly
- Keep PRs focused — prefer multiple small PRs over one large one

---

## Questions?

- **Issues**: [GitHub Issues](https://github.com/georgehampton08-rgb/digital-life-narrative-ai/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Email**: <georgehampton08@gmail.com>

---

**Thank you for contributing! 🎉**
