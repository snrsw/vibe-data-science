from typing import Any, Dict, Optional
import structlog
from pydantic import BaseModel

from vibe_data_science.ml.data.loader import DatasetConfig
from vibe_data_science.ml.data.preprocessing import PreprocessingConfig
from vibe_data_science.ml.features.extractors import FeatureConfig
from vibe_data_science.ml.models.classification import ModelConfig
from vibe_data_science.ml.models.evaluation import EvaluationConfig
from vibe_data_science.ml.pipeline.experiment import ExperimentConfig
from vibe_data_science.ml.pipeline.training import PipelineConfig, SplitConfig


logger = structlog.get_logger()

# Module-level registry storage
_CONFIG_REGISTRY: Dict[str, Dict[str, BaseModel]] = {}


def register_config(config_type: str, name: str, config: BaseModel) -> None:
    if config_type not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[config_type] = {}
    _CONFIG_REGISTRY[config_type][name] = config
    logger.info("config_registered", config_type=config_type, name=name)


def get_config(config_type: str, name: str) -> Optional[BaseModel]:
    if config_type in _CONFIG_REGISTRY and name in _CONFIG_REGISTRY[config_type]:
        return _CONFIG_REGISTRY[config_type][name]
    logger.error("config_not_found", config_type=config_type, name=name)
    return None


def get_typed_config(config_type: str, name: str, config_class: type) -> Any:
    config = get_config(config_type, name)
    if config and isinstance(config, config_class):
        return config
    return None


def list_configs(config_type: Optional[str] = None) -> Dict:
    if config_type:
        return {
            name: type(config).__name__
            for name, config in _CONFIG_REGISTRY.get(config_type, {}).items()
        }

    return {
        ct: {name: type(config).__name__ for name, config in configs.items()}
        for ct, configs in _CONFIG_REGISTRY.items()
    }


def create_random_forest_config(
    n_estimators: int = 100, max_depth: int = 10, random_state: int = 42
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
    C: float = 1.0, kernel: str = "rbf", random_state: int = 42
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


def update_config(base_config: BaseModel, **updates) -> BaseModel:
    data = base_config.model_dump()
    for key, value in updates.items():
        if key in data:
            data[key] = value

    return type(base_config).model_validate(data)


def initialize_configs() -> None:
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

    logger.info("configs_initialized")
