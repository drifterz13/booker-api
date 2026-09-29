.PHONY: dev chainlit test format

dev:
	uv run fastapi dev

format:
	uv run ruff format app

chainlit:
	uv run chainlit run app/chainlit_app.py

test:
	uv run python -m unittest discover -s tests -v
