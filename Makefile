.PHONY: install dev lint format type test

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	black src tests

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest
