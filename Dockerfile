FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model.pkl

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY artifacts ./artifacts

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD uvicorn churnstream.api.main:app --host 0.0.0.0 --port 8000