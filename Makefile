# Makefile for HAILO RPI5 Development Project
# 
# This provides convenient shortcuts for development tasks.
# For installation, use: ./install.sh

.PHONY: help test lint format clean docs

# Default target
help:
	@echo "HAILO RPI5 Development Commands:"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  ./install.sh         - Complete project installation (recommended)"
	@echo "  ./install.sh --help  - Show installation options"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests (requires hardware)"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  python examples/detection_example.py --help"
	@echo "  python examples/classification_example.py --help"
	@echo ""

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src examples tools tests
	mypy src

format:
	black src examples tools tests
	isort src examples tools tests

format-check:
	black --check src examples tools tests
	isort --check-only src examples tools tests

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "Documentation generation not implemented yet"
	@echo "Consider using sphinx-build docs/ docs/_build/"
