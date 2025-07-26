from typing import Any

import polars as pl
import structlog
from pydantic import BaseModel

from vibe_data_science.ml.data.loader import load_dataset, DatasetConfig
from vibe_data_science.ml.data.preprocessing import (
    preprocess_dataset,
    PreprocessingConfig,
)
from vibe_data_science.ml.features.extractors import extract_features, FeatureConfig
from vibe_data_science.ml.models.classification import train_model, ModelConfig
from vibe_data_science.ml.models.evaluation import (
    evaluate_model,
    EvaluationConfig,
    MetricsResult,
)
from vibe_data_science.ml.pipeline.experiment import (
    log_experiment,
    log_model,
    ExperimentConfig,
)


logger = structlog.get_logger()


class SplitConfig(BaseModel):
    test_ratio: float = 0.2
    validation_ratio: float = 0.2
    random_seed: int = 42

    model_config = {"frozen": True}


class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    features: FeatureConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    split: SplitConfig
    experiment: ExperimentConfig

    model_config = {"frozen": True}


def train_test_split(
    df: pl.DataFrame, target_column: str, config: SplitConfig
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    logger.info(
        "splitting_dataset",
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )

    shuffled_df = df.sample(fraction=1.0, seed=config.random_seed)

    target = shuffled_df[target_column]
    features_df = shuffled_df.drop(target_column)

    total_rows = features_df.shape[0]
    test_size = int(total_rows * config.test_ratio)

    X_train = features_df.slice(test_size, total_rows - test_size)
    y_train = target.slice(test_size, total_rows - test_size)
    X_test = features_df.slice(0, test_size)
    y_test = target.slice(0, test_size)

    logger.info(
        "dataset_split_complete", train_size=len(X_train), test_size=len(X_test)
    )

    return X_train, y_train, X_test, y_test


def run_pipeline(config: PipelineConfig) -> tuple[Any, MetricsResult]:
    logger.info(
        "starting_ml_pipeline",
        experiment_name=config.experiment.experiment_name,
        model_type=config.model.model_type,
    )

    dataset = load_dataset(config.dataset)

    X_train, y_train, X_test, y_test = train_test_split(
        dataset, config.dataset.target_column, config.split
    )

    X_train_processed = preprocess_dataset(X_train, config.preprocessing)
    X_test_processed = preprocess_dataset(X_test, config.preprocessing)

    X_train_features = extract_features(X_train_processed, config.features)
    X_test_features = extract_features(X_test_processed, config.features)

    model = train_model(X_train_features, y_train, config.model)

    metrics = evaluate_model(model, X_test_features, y_test, config.evaluation)

    log_experiment(config.experiment, config.model.hyperparameters, metrics)

    model_name = f"{config.model.model_type}_{config.experiment.run_name}"
    log_model(model, model_name)

    logger.info(
        "pipeline_complete",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
    )

    return model, metrics
