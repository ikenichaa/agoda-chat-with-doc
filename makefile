.PHONY: help install run build up down logs clean deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync --all-extras

install-prod: ## Install production dependencies only
	uv sync

run: ## Run the app locally (development)
	chainlit run app.py --watch

build: ## Build Docker image
	docker compose build app

up: ## Start all services with Docker Compose
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Show application logs
	docker compose logs -f app

clean: ## Clean up containers and volumes
	docker compose down -v
	rm -rf volumes/

test: ## Run all tests
	uv run --extra dev pytest

test-verbose: ## Run tests with verbose output
	uv run --extra dev pytest -v

test-coverage: ## Run tests with coverage report
	uv run --extra dev pytest --cov=. --cov-report=html --cov-report=term

format: ## Format code with black and isort
	@echo "Formatting code..."
	@command -v black >/dev/null 2>&1 || { echo "black not installed"; exit 1; }
	uv run black .
	uv run isort .
