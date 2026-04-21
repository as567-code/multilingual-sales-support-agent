# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the multilingual sales/support agent.
#
# Design goals:
#   * Cold start < 15s: pre-download spaCy en_core_web_lg and the e5 embedding
#     model into the builder layer so the runtime image has them on disk.
#   * Reproducibility: everything installed from pinned versions in
#     pyproject.toml; no implicit model downloads at container start.
#   * Small-ish runtime: builder carries compilers + build deps; the runtime
#     stage ships only the populated virtualenv, HF cache, and app code.
#   * Non-root at runtime: uvicorn runs as an unprivileged UID.
#
# Size-wise this image is ~3GB — dominated by torch (~800MB), sentence-transformers
# weights (~1GB), and spaCy en_core_web_lg (~500MB). Shrinking further means
# swapping to CPU-only torch wheels or quantized embedding models; out of scope
# for this stage.

ARG PYTHON_VERSION=3.12-slim

# --------------------------------------------------------------------------
# builder: install deps + warm model caches
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/opt/hf-cache

# Build-time deps: gcc + libgomp for FAISS, curl for healthchecks later.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python deps first so the layer caches across code changes.
WORKDIR /build
COPY pyproject.toml README.md ./
# hatchling builds a wheel from pyproject's declared packages (app, ingest,
# evals) — copy the minimum file tree needed for discovery so the deps layer
# caches independently of the real source trees.
COPY app/__init__.py app/__init__.py
COPY ingest/__init__.py ingest/__init__.py
COPY evals/__init__.py evals/__init__.py
RUN pip install --upgrade pip && pip install .

# Pre-download the two heaviest runtime artifacts so the image is self-contained:
#   1. spaCy English pipeline (Presidio PERSON/LOC NER)
#   2. multilingual-e5-base (retrieval embeddings)
RUN python -m spacy download en_core_web_lg
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# --------------------------------------------------------------------------
# runtime: slim image, non-root, uvicorn on :8000
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/opt/hf-cache \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    # Disable HF telemetry + ensure offline use of pre-downloaded weights.
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# curl is needed for HEALTHCHECK; libgomp1 is a torch/faiss runtime dep.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app -u 10001 app

# Bring the populated venv + HF cache from the builder stage.
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

WORKDIR /app
# Ship only what's needed to run the API + orchestrator + ingested index.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app data/processed/ ./data/processed/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
