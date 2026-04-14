.PHONY: install dev test lint format run docker-build docker-run clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

run:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t rag-chatbot .

docker-run:
	docker compose up -d

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov data/vector_store
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
