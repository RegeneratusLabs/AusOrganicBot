# OrgStandardsBot backend
#
# Build:  docker build -t orgstandardsbot-backend -f backend/Dockerfile backend
# Run:    docker run -p 8080:8080 -v $(pwd)/data:/app/data --env-file backend/.env orgstandardsbot-backend

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf \
    PORT=8080

WORKDIR /app

# Which embedding family to bake into the image. Must match EMBEDDING_FAMILY at
# runtime, or the container pays a large model download on first request.
# minicpm is the documented default (see backend/.env.example); build with
# --build-arg EMBEDDING_FAMILY=all-minilm for the 384-dim fallback.
ARG EMBEDDING_FAMILY=minicpm

# Persist the choice so the image warms and runs the same embedding family.
ENV EMBEDDING_FAMILY=${EMBEDDING_FAMILY}

# System deps for pdfplumber/chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the selected embedding model so containers start without a
# network fetch. Both families are cheap to download relative to first-request
# latency; only the selected one is kept meaningful at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
    import os; fam = os.environ.get('EMBEDDING_FAMILY', 'minicpm'); \
    name = 'sentence-transformers/all-MiniLM-L6-v2' if fam == 'all-minilm' else 'openbmb/MiniCPM-Embedding-Light'; \
    S(name, trust_remote_code=True); print('preloaded', name)"

COPY app ./app

RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
