# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into /app/.venv ----
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

# ---- runtime: no build tools, non-root ----
FROM python:3.13-slim
RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir -p /data && chown app:app /data
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY docs ./docs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data
USER app
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

# one worker on purpose: per-session locks live in process memory and SQLite is the store
# client IPs are not used, so proxy headers are not trusted at all
CMD ["uvicorn", "taxi_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
