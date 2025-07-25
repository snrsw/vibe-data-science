# vibe-data-science

## Getting Started

### Install Dependencies

```bash
uv sync
```

### Download datasets

```bash
uv run kaggle datasets download amulyas/penguin-size-dataset
```

## How to develop

### Linting

```bash
uvx ruff format .
```
```bash
uvx ruff check . --fix
```

### Type Checking

```bash
uvx ty check .
```

### Testing

```bash
uvx pytest tests
```
