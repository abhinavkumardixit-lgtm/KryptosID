# ==============================================================================
# KryptosID: Production Multi-Stage Dockerfile (docker/Dockerfile)
# Implements Section 10.3 of DESIGN.md:
# - Slim Python 3.11+ base
# - CPU inference optimized
# - Healthcheck on /v1/health
# - Non-root execution for compliance
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install minimal OS dependencies for OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for zero-PII security compliance
RUN groupadd -r kryptos && useradd -r -g kryptos -d /app -s /sbin/nologin kryptos

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/kryptos/.local
ENV PATH=/home/kryptos/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code
COPY --chown=kryptos:kryptos . /app

# Create persistent folders for database and reports
RUN mkdir -p /app/generated_reports /app/models \
    && chown -R kryptos:kryptos /app

USER kryptos

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Healthcheck wired to /v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default runtime entrypoint: FastAPI inference service
CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
