.PHONY: dev, chainlit

dev: 
	uv run uvicorn app.main:app --reload

chainlit:
	uv run chainlit run app/chainlit_app.py


