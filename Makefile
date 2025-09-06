# Makefile for HAILO RPI5 Development Project

.PHONY: help install install-dev test lint format clean build docs setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  format       - Format code (black)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  setup-dev    - Set up development environment"
	@echo "  docs         - Generate documentation"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,hailo]"

install-hailo:
	pip install ".[hailo]"

# Development setup
setup-dev: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'python tools/device_info.py' to test HAILO device connectivity"

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=hailo_rpi5 --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 hailo_rpi5 examples tools tests
	mypy hailo_rpi5

format:
	black hailo_rpi5 examples tools tests

format-check:
	black --check hailo_rpi5 examples tools tests

# Building
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation (placeholder)
docs:
	@echo "Documentation generation not implemented yet"

# Development helpers
run-device-check:
	python tools/device_info.py

run-example-classification:
	@echo "Usage: make run-example-classification HEF=path/to/model.hef IMAGE=path/to/image.jpg"
	@if [ -z "$(HEF)" ] || [ -z "$(IMAGE)" ]; then \
		echo "Please provide HEF and IMAGE parameters"; \
	else \
		python examples/classification_example.py --hef $(HEF) --image $(IMAGE); \
	fi

run-example-detection:
	@echo "Usage: make run-example-detection HEF=path/to/model.hef IMAGE=path/to/image.jpg"
	@if [ -z "$(HEF)" ] || [ -z "$(IMAGE)" ]; then \
		echo "Please provide HEF and IMAGE parameters"; \
	else \
		python examples/detection_example.py --hef $(HEF) --image $(IMAGE); \
	fi

benchmark:
	@echo "Usage: make benchmark HEF=path/to/model.hef"
	@if [ -z "$(HEF)" ]; then \
		echo "Please provide HEF parameter"; \
	else \
		python tools/benchmark.py --hef $(HEF); \
	fi

analyze-model:
	@echo "Usage: make analyze-model HEF=path/to/model.hef"
	@if [ -z "$(HEF)" ]; then \
		echo "Please provide HEF parameter"; \
	else \
		python tools/model_analyzer.py $(HEF) --validate; \
	fi
