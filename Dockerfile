FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

# scikit-learn and scipy ship manylinux wheels, but keeping libgomp1 avoids a
# silent fallback to a slow pure-Python path on some slim bases.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY pyproject.toml ./

RUN mkdir -p /srv/app/data /srv/app/artifacts

# Run unprivileged.
RUN useradd --create-home --uid 10001 reco \
 && chown -R reco:reco /srv/app
USER reco

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
