.PHONY: help install install-dev lint format test cov generate train serve realtime clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	pip install -e .

install-dev: ## Install runtime + dev dependencies
	pip install -e ".[dev]"

lint: ## Run ruff lint checks
	ruff check .

format: ## Auto-format and fix with ruff
	ruff check --fix .
	ruff format .

test: ## Run the test suite
	pytest

cov: ## Run tests with coverage report
	pytest --cov --cov-report=term-missing

generate: ## Generate the synthetic dataset
	ran6g generate

train: ## Train all models
	ran6g train

serve: ## Launch the FastAPI server (development)
	ran6g serve --reload

realtime: ## Run the real-time inference demo
	ran6g realtime

clean: ## Remove caches and generated outputs
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
