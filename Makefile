# Convenience targets. Run `make help` for the list.
PYTHON ?= python

.PHONY: help install test lint format fortran figures quick clean

help:
	@echo "install   install the package in editable mode with dev tools"
	@echo "test      run the test-suite (pytest)"
	@echo "lint      ruff lint + format check"
	@echo "format    auto-format with ruff"
	@echo "fortran   build the Fortran solver"
	@echo "figures   regenerate every figure in figures/"
	@echo "quick     run every script in --quick mode (smoke test)"
	@echo "clean     remove build artefacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

fortran:
	$(MAKE) -C fortran

figures: fortran
	@for s in scripts/0*.py; do echo "== $$s"; $(PYTHON) $$s || exit 1; done

quick: fortran
	@for s in scripts/0*.py; do echo "== $$s"; $(PYTHON) $$s --quick --outdir build/quick-figures || exit 1; done

clean:
	$(MAKE) -C fortran clean
	rm -rf build .pytest_cache .ruff_cache src/*.egg-info
