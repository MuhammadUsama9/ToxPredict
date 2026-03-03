# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="QSAR Toxicity Prediction Project"
LABEL description="FastAPI service for molecular toxicity prediction (Tox21/GCN)"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Runtime system libs for RDKit and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project source
COPY src/ ./src/
COPY tests/ ./tests/
COPY checkpoints/ ./checkpoints/
COPY train.py evaluate.py ./

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
