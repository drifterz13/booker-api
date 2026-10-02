.PHONY: dev test eval format migrate migration db-check

dev:
	uv run fastapi dev

db-check:
	uv run alembic check

migrate:
	uv run alembic upgrade head

migration:
	@test -n "$(message)" || (echo 'Usage: make migration message="description"' >&2; exit 1)
	uv run alembic revision --autogenerate -m "$(message)"

format:
	uv run ruff format app

test:
	uv run python -m unittest discover -s tests -v

eval:
	DEEPEVAL_TELEMETRY_OPT_OUT=1 uv run --group eval deepeval test run eval/test_rag.py
