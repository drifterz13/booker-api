# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 booker

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder --chown=booker:booker /app /app
COPY --chown=booker:booker alembic.ini ./alembic.ini
COPY --chown=booker:booker migrations ./migrations
RUN chown booker:booker /app
USER booker

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
