from typing import Any, Literal

import numpy as np
import polars as pl
import structlog
from pydantic import BaseModel, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


logger = structlog.get_logger()


class ModelConfig(BaseModel):
    model_type: Literal["random_forest", "svm"]
    hyperparameters: dict[str, Any] = {}

    model_config = {"frozen": True}

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed_models = ["random_forest", "svm"]
        if v not in allowed_models:
            raise ValueError(f"Model type must be one of: {', '.join(allowed_models)}")
        return v


def _create_model_instance(config: ModelConfig) -> Any:
    if config.model_type == "random_forest":
        return RandomForestClassifier(**config.hyperparameters)
    elif config.model_type == "svm":
        return SVC(**config.hyperparameters)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


def train_model(features: pl.DataFrame, target: pl.Series, config: ModelConfig) -> Any:
    logger.info(
        "training_model",
        model_type=config.model_type,
        feature_count=features.width,
        sample_count=features.height,
        hyperparameters=config.hyperparameters,
    )

    # Convert polars dataframe to numpy arrays
    X = features.to_numpy()
    y = target.to_numpy()

    # Create and train the model
    model = _create_model_instance(config)
    model.fit(X, y)

    logger.info("model_training_complete", model_type=config.model_type)

    return model


def predict(model: Any, features: pl.DataFrame) -> np.ndarray:
    logger.info(
        "making_predictions",
        model_type=type(model).__name__,
        feature_count=features.width,
        sample_count=features.height,
    )

    # Convert polars dataframe to numpy array
    X = features.to_numpy()

    # Make predictions
    predictions = model.predict(X)

    logger.info(
        "predictions_complete",
        prediction_count=len(predictions),
        unique_predictions=len(np.unique(predictions)),
    )

    return predictions
