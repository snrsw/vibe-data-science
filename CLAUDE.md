# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vibe-data-science is a data science project focused on analyzing penguin datasets and building ML pipelines. The project processes the Palmer Penguins dataset, which contains information about different penguin species, their physical measurements, and demographic details.

## Key Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Download penguin dataset
uv run kaggle datasets download amulyas/penguin-size-dataset
```

### Development

```bash
# Format code with ruff
uvx ruff format .

# Run linting with auto-fixes
uvx ruff check . --fix

# Type checking
uvx ty check .

# Run tests
uv run pytest tests

# Run single test
uv run pytest tests/path_to_test.py::test_function_name
```

## Project Structure

- `data/`: Contains datasets including `penguins_size.csv`
- `src/vibe_data_science/`: Main package code
- `docs/`: Documentation including dataset descriptions

## Data Processing

The main dataset (`penguins_size.csv`) contains the following columns:
- species: Penguin species (Adelie, Chinstrap, Gentoo)
- island: Location (Torgersen, Biscoe, Dream)
- culmen_length_mm: Culmen length in millimeters
- culmen_depth_mm: Culmen depth in millimeters
- flipper_length_mm: Flipper length in millimeters
- body_mass_g: Body mass in grams
- sex: Sex of the penguin (MALE, FEMALE, NA)

See detailed column descriptions in `docs/datasets.md`.

## Problem Definition

See `docs/problem.md` for the problem statement, inputs, outputs, and evaluation metrics.

## Technology Stack

- Python 3.11+
- DuckDB: SQL database for data processing
- Polars: Data manipulation library
- UV: Python package manager and virtual environment
- Ruff: Code linting and formatting
- Typer (ty): Type checking tool
- Pytest: Testing framework

## Cording Style

* Use pydantic.BaseModel rather than dataclasses and classes.
* Use immutable data structures as much as possible.
* Use functional programming techniques where appropriate.
* Use mlflow for model tracking and management.
* Use python 3.10+ features such as type hints, f-strings, and match statements (Do not use List and Dict like annotations).
* Do not use comments to explain code. Instead, use descriptive variable and function names.
* Use explicit named arguments for function calls.
* Use structured logging with the `structlog` module, do not use print statements for debugging.

## Development Workflow

* Use t-wada's TDD style (see Japanese resources for more tdetails).
* Plan first, then implement if the owner agrees with the plan.
* Check linting and type checking before committing code.
