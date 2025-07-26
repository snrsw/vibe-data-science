# Configuration Management in vibe-data-science

This document outlines the configuration management strategy for ML pipelines in the vibe-data-science project, focusing on functional programming principles and immutable data structures.

## Introduction

The vibe-data-science project uses a comprehensive configuration system to manage ML pipeline parameters. The system prioritizes:

- Immutability and functional programming principles
- Type safety using Pydantic models
- Centralized configuration management
- Flexibility for experimentation
- Descriptive naming over comments

## Functional Configuration Registry

A module-level configuration registry implemented with pure functions:

```python
from typing import Dict, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

# Module-level registry storage
_CONFIG_REGISTRY: Dict[str, Dict[str, BaseModel]] = {}

def register_config(config_type: str, name: str, config: BaseModel) -> None:
    """Register a configuration in the registry."""
    if config_type not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[config_type] = {}
    _CONFIG_REGISTRY[config_type][name] = config

def get_config(config_type: str, name: str) -> Optional[BaseModel]:
    """Retrieve a configuration from the registry."""
    return _CONFIG_REGISTRY.get(config_type, {}).get(name)

def list_configs(config_type: Optional[str] = None) -> Dict:
    """List available configurations in the registry."""
    if config_type:
        return {name: type(config).__name__
                for name, config in _CONFIG_REGISTRY.get(config_type, {}).items()}

    return {ct: {name: type(config).__name__ for name, config in configs.items()}
            for ct, configs in _CONFIG_REGISTRY.items()}
```

## Factory Functions

Factory functions create standard configurations with descriptive names:

```python
def create_random_forest_config(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> ModelConfig:
    return ModelConfig(
        model_type="random_forest",
        hyperparameters={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
    )

def create_svm_config(
    C: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42
) -> ModelConfig:
    return ModelConfig(
        model_type="svm",
        hyperparameters={
            "C": C,
            "kernel": kernel,
            "random_state": random_state,
        },
    )

def create_default_pipeline_config(
    dataset_path: str = "data/penguins_size.csv",
    target_column: str = "species",
    model_config: Optional[ModelConfig] = None,
) -> PipelineConfig:
    if model_config is None:
        model_config = create_random_forest_config()

    return PipelineConfig(
        dataset=DatasetConfig(
            filepath=dataset_path,
            target_column=target_column,
            null_values=("NA", "."),
        ),
        preprocessing=PreprocessingConfig(
            categorical_features=("island", "sex"),
            numeric_features=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
            fill_strategy="median",
        ),
        features=FeatureConfig(
            features_to_use=(
                "island_encoded",
                "sex_encoded",
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
            normalize=True,
        ),
        model=model_config,
        evaluation=EvaluationConfig(
            metrics=("accuracy", "precision", "recall", "f1"),
            average="macro",
        ),
        split=SplitConfig(
            test_ratio=0.2,
            validation_ratio=0.0,
            random_seed=42,
        ),
        experiment=ExperimentConfig(
            experiment_name="penguin_classification",
            run_name=f"{model_config.model_type}_pipeline",
        ),
    )
```

## External Configuration Files

Configuration loading from YAML/JSON files:

```python
import yaml
from pathlib import Path

def load_config_from_yaml(
    filepath: str,
    config_class: Type[T]
) -> Optional[T]:
    path = Path(filepath)
    if not path.exists():
        return None

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_class.model_validate(config_dict)
```

Example YAML configuration file (`configs/pipelines/feature_exploration.yaml`):

```yaml
dataset:
  filepath: "data/penguins_size.csv"
  target_column: "species"
  null_values: ["NA", "."]

preprocessing:
  categorical_features: ["island", "sex"]
  numeric_features:
    - "culmen_length_mm"
    - "culmen_depth_mm"
    - "flipper_length_mm"
    - "body_mass_g"
  fill_strategy: "mean"

features:
  features_to_use:
    - "island_encoded"
    - "sex_encoded"
    - "culmen_length_mm"
    - "culmen_depth_mm"
    - "flipper_length_mm"
    - "body_mass_g"
  normalize: true

model:
  model_type: "random_forest"
  hyperparameters:
    n_estimators: 150
    max_depth: 12
    random_state: 42

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  average: "macro"

split:
  test_ratio: 0.2
  validation_ratio: 0.0
  random_seed: 42

experiment:
  experiment_name: "penguin_feature_exploration"
  run_name: "normalized_features"
```

## Configuration Transformation

Functions to modify configurations while maintaining immutability:

```python
def update_config(base_config: T, **updates) -> T:
    data = base_config.model_dump()
    for key, value in updates.items():
        if key in data:
            data[key] = value

    return type(base_config).model_validate(data)
```

## Registry Initialization

Setting up the registry with standard configurations:

```python
def initialize_configs() -> None:
    """Initialize the configuration registry with standard configurations."""
    # Create model configurations
    rf_default = create_random_forest_config()
    rf_deep = create_random_forest_config(n_estimators=200, max_depth=20)
    svm_default = create_svm_config()

    # Register model configurations
    register_config("model", "random_forest_default", rf_default)
    register_config("model", "random_forest_deep", rf_deep)
    register_config("model", "svm_default", svm_default)

    # Create and register pipeline configurations
    default_pipeline = create_default_pipeline_config(model_config=rf_default)
    deep_pipeline = create_default_pipeline_config(model_config=rf_deep)
    svm_pipeline = create_default_pipeline_config(model_config=svm_default)

    register_config("pipeline", "default", default_pipeline)
    register_config("pipeline", "deep_forest", deep_pipeline)
    register_config("pipeline", "svm", svm_pipeline)
```

## Usage Patterns

### Running Predefined Configurations

```python
initialize_configs()
pipeline_config = get_config("pipeline", "deep_forest")
model, metrics = run_pipeline(pipeline_config)
```

### Loading from YAML

```python
config = load_config_from_yaml(
    "configs/pipelines/feature_exploration.yaml",
    PipelineConfig
)
model, metrics = run_pipeline(config)
```

### Creating Modified Configurations

```python
initialize_configs()
base_config = get_config("pipeline", "default")
modified_config = update_config(
    base_config,
    split=SplitConfig(test_ratio=0.3, random_seed=99)
)
model, metrics = run_pipeline(modified_config)
```

### Comparing Multiple Configurations

```python
initialize_configs()
results = {}

for config_name in ["default", "deep_forest", "svm"]:
    pipeline_config = get_config("pipeline", config_name)
    if pipeline_config:
        _, metrics = run_pipeline(pipeline_config)
        results[config_name] = {
            "accuracy": metrics.metrics.get("accuracy"),
            "precision": metrics.metrics.get("precision"),
            "recall": metrics.metrics.get("recall"),
        }

best_config = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
```

## Best Practices

1. **Use Factory Functions**: Create configurations with semantic factory functions rather than direct instantiation
2. **Register All Configurations**: Store configurations in the registry for centralized management
3. **Maintain Immutability**: Use `update_config` instead of modifying configurations in-place
4. **Descriptive Names**: Use descriptive names for configurations rather than comments
5. **Environment-Based Configurations**: Load different configurations based on the environment (dev/test/prod)
6. **Document Configurations**: Include metadata in configurations to explain their purpose
7. **Version Configurations**: Include version numbers in saved configuration files
8. **Validate Configs Early**: Check configurations at loading time rather than runtime
9. **Default Values**: Provide sensible defaults in factory functions
10. **Composition Over Inheritance**: Compose configurations from smaller parts

## Environment-Specific Configurations

For different environments (dev/test/prod):

```python
import os

def get_env_config_path(config_name: str) -> str:
    env = os.getenv("APP_ENV", "dev")
    return f"configs/{env}/{config_name}.yaml"

def load_env_pipeline_config(config_name: str) -> Optional[PipelineConfig]:
    path = get_env_config_path(config_name)
    return load_config_from_yaml(path, PipelineConfig)
```

## Command-Line Interface

Run pipelines with different configurations from the command line:

```python
#!/usr/bin/env python3
import sys
import structlog
from vibe_data_science.ml.pipeline import (
    initialize_configs,
    get_config,
    load_config_from_yaml,
    run_pipeline,
    PipelineConfig
)

logger = structlog.get_logger()

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_config.py <config_name_or_path>")
        sys.exit(1)

    config_name_or_path = sys.argv[1]
    initialize_configs()

    if config_name_or_path.endswith((".yaml", ".yml")):
        config = load_config_from_yaml(config_name_or_path, PipelineConfig)
        if not config:
            logger.error("config_load_failed", filepath=config_name_or_path)
            sys.exit(1)
    else:
        config = get_config("pipeline", config_name_or_path)
        if not config:
            logger.error("config_not_found", name=config_name_or_path)
            sys.exit(1)

    logger.info("running_pipeline", config_name=config_name_or_path)
    model, metrics = run_pipeline(config)

    logger.info(
        "pipeline_complete",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
    )

if __name__ == "__main__":
    main()
```
