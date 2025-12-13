.PHONY: help install run build up down logs clean deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
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

deploy: ## Deploy the application
	./deploy.sh

test: ## Run tests (placeholder)
	@echo "Running tests..."
	uv run pytest tests/ || echo "No tests configured yet"

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

