FROM python:3.12-slim

WORKDIR /app

# Install system build dependencies and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Install dependencies into system environment or container venv
RUN uv sync --frozen --no-dev

# Copy application source code, configurations, and trained artifacts
COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/ ./artifacts/

# Expose HTTP port
EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Run FastAPI serving via uvicorn
CMD ["uvicorn", "fraudguard.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
