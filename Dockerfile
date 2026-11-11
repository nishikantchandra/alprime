FROM python:3.12-slim

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL}

# Unprivileged runtime user.
RUN groupadd --system app && useradd --system --gid app --home-dir /app --no-create-home app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-cache the ONNX embedding + cross-encoder models so startup needs no downloads.
RUN python -c "import os; from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; c=os.environ['FASTEMBED_CACHE_PATH']; TextEmbedding(os.environ['EMBEDDING_MODEL'], cache_dir=c); TextCrossEncoder(os.environ['RERANKER_MODEL'], cache_dir=c)"

COPY app ./app
COPY static ./static
COPY sample_docs ./sample_docs

RUN mkdir -p /app/data/qdrant && chown -R app:app /app/data /app/.fastembed_cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
