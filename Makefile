# SOC-Triage-Gym — developer task runner.
# Usage: `make <target>`. On Windows, run these under Git Bash / WSL, or
# invoke the underlying commands directly (see each recipe).

PYTHON ?= python
PORT   ?= 7860

.DEFAULT_GOAL := help

.PHONY: help install lint fmt fmt-check test test-cov serve demo plots clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff lint checks
	ruff check .

fmt: ## Auto-fix lint issues and format
	ruff check . --fix
	ruff format .

fmt-check: ## Verify formatting without writing
	ruff format --check .

test: ## Run the test suite
	pytest -q

test-cov: ## Run tests with coverage (needs pytest-cov)
	pytest -q --cov=. --cov-report=term-missing

serve: ## Start the OpenEnv server on $(PORT)
	uvicorn server.app:app --host 0.0.0.0 --port $(PORT)

demo: ## Run the one-command judge demo
	$(PYTHON) demo.py

plots: ## Regenerate README charts from repo metadata (no GPU)
	$(PYTHON) scripts/gen_readme_assets.py

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ */__pycache__ .pytest_cache *.egg-info build dist
