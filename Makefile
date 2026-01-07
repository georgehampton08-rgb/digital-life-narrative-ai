# Digital Life Narrative AI - Development Commands
#
# Usage:
#   make              Show available commands
#   make install      Install dependencies
#   make test         Run test suite
#   make check        Run all checks
#   make demo         Run demo flow
#   make clean        Remove build artifacts

.DEFAULT_GOAL := help

# --- Variables ---
PYTHON  := python
POETRY  := poetry
PYTEST  := $(POETRY) run pytest
RUFF    := $(POETRY) run ruff
BLACK   := $(POETRY) run black
MYPY    := $(POETRY) run mypy
CLI     := $(POETRY) run organizer
SOURCES := organizer src demo tests

# --- Help ---

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Installation ---

.PHONY: install
install: ## Install project dependencies (including dev)
	$(POETRY) install

.PHONY: install-dev
install-dev: ## Alias for install (installs all dependencies)
	$(POETRY) install

# --- Code Quality ---

.PHONY: lint
lint: ## Run linting checks with ruff
	$(RUFF) check $(SOURCES)

.PHONY: format
format: ## Format code with black
	$(BLACK) $(SOURCES)

.PHONY: format-check
format-check: ## Check formatting without making changes
	$(BLACK) --check $(SOURCES)

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	-$(MYPY) src organizer --ignore-missing-imports

# --- Testing ---

.PHONY: test
test: ## Run test suite
	$(PYTEST) --maxfail=5 -q

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	$(PYTEST) -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	$(PYTEST) --cov=organizer --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

.PHONY: test-fast
test-fast: ## Run tests, stop on first failure
	$(PYTEST) -x

# --- Combinations ---

.PHONY: check
check: lint test ## Run all checks (lint + test)

.PHONY: ci
ci: lint format-check test ## Run CI-equivalent checks (lint + format-check + test)

# --- Demo & Run ---

.PHONY: demo
demo: ## Run the demo flow without AI (no API key required)
	@if [ ! -d "./demo_data" ]; then \
		echo "Generating demo data..."; \
		$(POETRY) run python -m demo.generate_demo_data --output ./demo_data; \
	fi
	@echo "Running analysis in fallback mode..."
	$(CLI) analyze --input ./demo_data --output ./demo_report_fallback --no-ai
	@echo "Report generated: demo_report_fallback.html"

.PHONY: demo-ai
demo-ai: ## Run the demo flow with AI (requires Gemini API key)
	@if [ ! -d "./demo_data" ]; then \
		echo "Generating demo data..."; \
		$(POETRY) run python -m demo.generate_demo_data --output ./demo_data; \
	fi
	@echo "Running AI analysis..."
	@echo "Note: This will fail if your GEMINI_API_KEY is not configured."
	$(CLI) analyze --input ./demo_data --output ./demo_report_ai
	@echo "Report generated: demo_report_ai.html"

.PHONY: run
run: ## Run the CLI (use ARGS="..." to passing arguments)
	$(CLI) $(ARGS)

# --- Cleanup ---

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf `find . -type d -name "__pycache__"`
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

.PHONY: clean-all
clean-all: clean ## Remove all generated files including demo data and reports
	rm -rf demo_data
	rm -rf demo_report_fallback.html
	rm -rf demo_report_ai.html
	rm -rf life_story_report.html

# --- Build & Docker ---

.PHONY: build
build: ## Build distribution packages (wheel and sdist)
	$(POETRY) build

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t life-story-reconstructor .

.PHONY: docker-run
docker-run: ## Run the CLI via Docker (mounts current dir to /data)
	docker run --rm -it -v $(shell pwd):/data life-story-reconstructor $(ARGS)

# --- Development ---

.PHONY: dev
dev: install ## Prepare development environment

.PHONY: shell
shell: ## Open a Python shell with project context
	$(POETRY) run python
