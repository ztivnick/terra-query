FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# system deps for psycopg + rasterio + pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /usr/local/bin/uv

WORKDIR /app

# deps first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY experiments ./experiments
RUN uv sync --frozen --no-dev

# config + DSN come from env at run time
ENV PATH="/opt/venv/bin:${PATH}"
EXPOSE 8000

CMD ["uvicorn", "terra_query.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
