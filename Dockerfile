# Dockerfile for Support Triage Agent
# Uses Python 3.11 slim as base image for lightweight container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for the application
# - build-essential for any compiled dependencies
# - curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for installation
COPY pyproject.toml README.md ./

# Install the project in production mode (no dev dependencies)
RUN pip install --no-cache-dir .

# Copy application source code
COPY src/ ./src/

# Copy the corpus directory (required for ingestion/retrieval)
COPY public/corpus/ ./public/corpus/

# Copy the samples directory (contains sample CSV files)
COPY public/samples/ ./public/samples/

# Create non-root user for security
RUN useradd --no-create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose HTTP port (Render will set PORT environment variable)
EXPOSE 8000

# Run the FastAPI application with Uvicorn
# Bind to 0.0.0.0 and use PORT environment variable (default 8000)
CMD ["sh", "-c", "uvicorn support_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}"]