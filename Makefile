.PHONY: env init clean format lint test

# Environment Management
env:
	conda env create -f env.yml

init: env
	conda run -n ryoma pip install poetry
	conda run -n ryoma poetry install

update:
	conda run -n ryoma poetry update

clean:
	conda env remove -n ryoma

# Development Tools
format:
	conda run -n ryoma poetry run black .
	conda run -n ryoma poetry run isort .

lint:
	conda run -n ryoma poetry run flake8 .
	conda run -n ryoma poetry run mypy .

test:
	conda run -n ryoma poetry run pytest

# Shortcuts
shell:
	conda run -n ryoma ipython

.DEFAULT_GOAL := help
help:
	@echo "Management Commands:"
	@echo "  make env      Create conda environment"
	@echo "  make init     Initialize project dependencies"
	@echo "  make update   Update project dependencies"
	@echo "  make clean    Remove environment"
	@echo "Development Commands:"
	@echo "  make format   Format code"
	@echo "  make lint     Check code"
	@echo "  make test     Run tests"
	@echo "  make shell    Open IPython"
