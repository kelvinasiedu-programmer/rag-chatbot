# ---------- build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY pyproject.toml .

RUN mkdir -p data/vector_store

# Pin HuggingFace cache to a writable directory inside the image
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Pre-download models at build time so startup is instant (no timeout)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L12-v2'); \
print('Embedding model ready')"

# Pre-download local LLM (used when LLM_BACKEND=local, the free default)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
AutoTokenizer.from_pretrained('google/flan-t5-base'); \
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base'); \
print('Local LLM ready')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/v1/health')"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
