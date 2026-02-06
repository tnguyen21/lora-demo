.PHONY: setup lint format test demo-text demo-vision clean

setup:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

lint:
	uvx ruff check .

format:
	uvx ruff format --line-length=144 .

format-check:
	uvx ruff format --check --line-length=144 .

test:
	python -m pytest tests/ -v

demo-text:
	python use_cases/text/01_ticket_routing/cost_comparison.py

demo-vision:
	python use_cases/vision/01_document_type/cost_comparison.py

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
