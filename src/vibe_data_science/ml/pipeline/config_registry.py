from typing import Any, Dict, Literal, Optional
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


def create_random_forest_optimized_config() -> ModelConfig:
    return ModelConfig(
        model_type="random_forest",
        hyperparameters={
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "bootstrap": True,
            "class_weight": "balanced",
            "random_state": 42,
        },
    )


def create_random_forest_fast_config() -> ModelConfig:
    return ModelConfig(
        model_type="random_forest",
        hyperparameters={
            "n_estimators": 50,
            "max_depth": 8,
            "max_features": "sqrt",
            "random_state": 42,
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


def create_svm_linear_config() -> ModelConfig:
    return ModelConfig(
        model_type="svm",
        hyperparameters={
            "C": 1.0,
            "kernel": "linear",
            "random_state": 42,
        },
    )


def create_svm_optimized_config() -> ModelConfig:
    return ModelConfig(
        model_type="svm",
        hyperparameters={
            "C": 10.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42,
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


def create_dataset_config(
    filepath: str = "data/penguins_size.csv",
    target_column: str = "species",
    null_values: tuple[str, ...] = ("NA", "."),
) -> DatasetConfig:
    return DatasetConfig(
        filepath=filepath,
        target_column=target_column,
        null_values=null_values,
    )


def create_preprocessing_config(
    categorical_features: tuple[str, ...] = ("island", "sex"),
    numeric_features: tuple[str, ...] = (
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ),
    fill_strategy: Literal["mean", "median", "mode"] = "median",
) -> PreprocessingConfig:
    return PreprocessingConfig(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        fill_strategy=fill_strategy,
    )


def create_feature_config(
    normalize: bool = True,
) -> FeatureConfig:
    return FeatureConfig(
        features_to_use=(
            "island_encoded",
            "sex_encoded",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ),
        normalize=normalize,
    )


def create_evaluation_config(
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
    average: Literal["micro", "macro", "weighted"] = "macro",
) -> EvaluationConfig:
    return EvaluationConfig(
        metrics=metrics,
        average=average,
    )


def initialize_configs() -> None:
    # Create model configurations
    rf_default = create_random_forest_config()
    rf_deep = create_random_forest_config(n_estimators=200, max_depth=20)
    rf_optimized = create_random_forest_optimized_config()
    rf_fast = create_random_forest_fast_config()
    svm_default = create_svm_config()
    svm_linear = create_svm_linear_config()
    svm_optimized = create_svm_optimized_config()

    # Register model configurations
    register_config("model", "random_forest_default", rf_default)
    register_config("model", "random_forest_deep", rf_deep)
    register_config("model", "random_forest_optimized", rf_optimized)
    register_config("model", "random_forest_fast", rf_fast)
    register_config("model", "svm_default", svm_default)
    register_config("model", "svm_linear", svm_linear)
    register_config("model", "svm_optimized", svm_optimized)

    # Create dataset configurations
    default_dataset = create_dataset_config()
    test_dataset = create_dataset_config(filepath="data/penguins_test.csv")

    # Register dataset configurations
    register_config("dataset", "default", default_dataset)
    register_config("dataset", "test", test_dataset)

    # Create preprocessing configurations
    default_preprocessing = create_preprocessing_config()
    mean_imputation = create_preprocessing_config(fill_strategy="mean")
    mode_imputation = create_preprocessing_config(fill_strategy="mode")

    # Register preprocessing configurations
    register_config("preprocessing", "default", default_preprocessing)
    register_config("preprocessing", "mean_imputation", mean_imputation)
    register_config("preprocessing", "mode_imputation", mode_imputation)

    # Create feature configurations
    default_features = create_feature_config()
    no_normalization = create_feature_config(normalize=False)

    # Register feature configurations
    register_config("feature", "default", default_features)
    register_config("feature", "no_normalization", no_normalization)

    # Create evaluation configurations
    default_evaluation = create_evaluation_config()
    full_metrics = create_evaluation_config(
        metrics=("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")
    )

    # Register evaluation configurations
    register_config("evaluation", "default", default_evaluation)
    register_config("evaluation", "full_metrics", full_metrics)

    # Create and register pipeline configurations
    default_pipeline = create_default_pipeline_config(model_config=rf_default)
    deep_pipeline = create_default_pipeline_config(model_config=rf_deep)
    fast_pipeline = create_default_pipeline_config(model_config=rf_fast)
    optimized_rf_pipeline = create_default_pipeline_config(model_config=rf_optimized)
    svm_pipeline = create_default_pipeline_config(model_config=svm_default)
    optimized_svm_pipeline = create_default_pipeline_config(model_config=svm_optimized)

    # Pipeline with different feature configurations
    no_norm_pipeline = update_config(default_pipeline, features=no_normalization)

    # Pipeline with different evaluation configurations
    full_eval_pipeline = update_config(default_pipeline, evaluation=full_metrics)

    # Register all pipeline configurations
    register_config("pipeline", "default", default_pipeline)
    register_config("pipeline", "deep_forest", deep_pipeline)
    register_config("pipeline", "fast_forest", fast_pipeline)
    register_config("pipeline", "optimized_forest", optimized_rf_pipeline)
    register_config("pipeline", "svm", svm_pipeline)
    register_config("pipeline", "optimized_svm", optimized_svm_pipeline)
    register_config("pipeline", "no_normalization", no_norm_pipeline)
    register_config("pipeline", "full_evaluation", full_eval_pipeline)

    logger.info("configs_initialized")
