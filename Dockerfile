# =============================================================================
# Digital Life Narrative AI - Dockerfile
# =============================================================================
#
# Build:
#   docker build -t digital-life-narrative-ai .
#
# Run help:
#   docker run --rm digital-life-narrative-ai --help
#
# Run analysis (with AI):
#   docker run --rm \
#     -v /path/to/exports:/data/input:ro \
#     -v /path/to/output:/data/output \
#     -e GEMINI_API_KEY=your-key \
#     digital-life-narrative-ai analyze \
#       --input /data/input \
#       --output /data/output/report.html
#
# Run analysis (fallback mode, no API key needed):
#   docker run --rm \
#     -v /path/to/exports:/data/input:ro \
#     -v /path/to/output:/data/output \
#     digital-life-narrative-ai analyze \
#       --input /data/input \
#       --output /data/output/report.html \
#       --no-ai
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY organizer/ ./organizer/
COPY src/ ./src/

# Install the package into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="George Hampton <georgehampton08@gmail.com>"
LABEL org.opencontainers.image.title="Digital Life Narrative AI"
LABEL org.opencontainers.image.description="AI-first application that reconstructs your life narrative from scattered media exports using Google Gemini."
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/georgehampton08-rgb/digital-life-narrative-ai"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy demo data for quick testing
WORKDIR /app
COPY demo/ ./demo/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /data/input /data/output && \
    chown -R appuser:appuser /data /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Pass your API key at runtime: -e GEMINI_API_KEY=your-key

# Declare volume mount points
VOLUME ["/data/input", "/data/output"]

# Set working directory for runtime
WORKDIR /data

# Health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=5s \
    CMD organizer --version || exit 1

# Entry point: the CLI command
ENTRYPOINT ["organizer"]

# Default command: show help
CMD ["--help"]
