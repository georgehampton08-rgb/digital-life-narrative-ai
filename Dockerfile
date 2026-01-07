# Use Python 3.11 slim image for a small footprint
FROM python:3.11-slim

# Install system dependencies for Pillow and other libs
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the application source
COPY . .

# Install dependencies and the package
RUN pip install --upgrade pip && \
    pip install .

# Create data directories for volume mounting
RUN mkdir -p /data/input /data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CI=true

# Entry point for the CLI
ENTRYPOINT ["organizer"]

# Default command shows help
CMD ["--help"]
