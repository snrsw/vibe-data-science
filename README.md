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

## Running ML Pipelines

The project provides a command-line interface for running ML pipelines with various configurations.

### Running a Predefined Configuration

To list all available configurations:

```bash
uv run python -m vibe_data_science.ml.main list
```

To list specific configuration types:

```bash
uv run python -m vibe_data_science.ml.main list model
uv run python -m vibe_data_science.ml.main list pipeline
```

To run a predefined pipeline configuration:

```bash
uv run python -m vibe_data_science.ml.main run default
uv run python -m vibe_data_science.ml.main run deep_forest
uv run python -m vibe_data_science.ml.main run optimized_forest
```

### Running with YAML Configuration Files

The project supports configuration via YAML files located in `configs/pipelines/`:

```bash
uv run python -m vibe_data_science.ml.main run configs/pipelines/optimized_random_forest.yaml
uv run python -m vibe_data_science.ml.main run configs/pipelines/svm_linear.yaml
```

Available configuration files:
- `optimized_random_forest.yaml`: Random forest with optimized hyperparameters
- `svm_linear.yaml`: SVM with linear kernel
- `feature_exploration.yaml`: Configuration for feature exploration
- `reduced_feature_set.yaml`: Pipeline with a reduced feature set
- `cross_validation.yaml`: Configuration with cross-validation setup
- `grid_search_rf.yaml`: Grid search for random forest hyperparameters
- `grid_search_svm.yaml`: Grid search for SVM hyperparameters

See `docs/configs.md` for detailed information about the configuration system.

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
uv run pytest tests
```
