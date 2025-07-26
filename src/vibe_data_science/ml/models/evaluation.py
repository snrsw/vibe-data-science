from typing import Any, Literal

import numpy as np
import polars as pl
import structlog
from pydantic import BaseModel, field_validator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


logger = structlog.get_logger()


class EvaluationConfig(BaseModel):
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
    average: Literal["micro", "macro", "weighted"] = "macro"

    model_config = {"frozen": True}

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        allowed_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
        ]
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(
                    f"Metric '{metric}' not supported. Must be one of: {', '.join(allowed_metrics)}"
                )
        return v


class MetricsResult(BaseModel):
    metrics: dict[str, float]
    confusion_matrix: list[list[int]] | None = None
    class_mapping: dict[Any, str] | None = None
    classification_report: str | None = None

    model_config = {"frozen": True}


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, config: EvaluationConfig
) -> MetricsResult:
    results = {}

    if "accuracy" in config.metrics:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))

    if "precision" in config.metrics:
        results["precision"] = float(
            precision_score(y_true, y_pred, average=config.average, zero_division=0)
        )

    if "recall" in config.metrics:
        results["recall"] = float(
            recall_score(y_true, y_pred, average=config.average, zero_division=0)
        )

    if "f1" in config.metrics:
        results["f1"] = float(
            f1_score(y_true, y_pred, average=config.average, zero_division=0)
        )

    if "roc_auc" in config.metrics:
        try:
            # For binary classification or multi-class with OvR strategy
            results["roc_auc"] = float(
                roc_auc_score(y_true, y_pred, multi_class="ovr", average=config.average)
            )
        except (ValueError, TypeError):
            # If ROC AUC can't be calculated
            logger.warning("roc_auc_calculation_failed")
            results["roc_auc"] = float(0.0)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=False)

    # Create class mapping
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_map = {}
    for i in unique_classes:
        try:
            key = int(i) if isinstance(i, (int, float, np.number)) else i
            class_map[key] = str(i)
        except (ValueError, TypeError):
            class_map[i] = str(i)

    logger.info(
        "metrics_calculated",
        metrics={k: round(v, 4) for k, v in results.items()},
        num_samples=len(y_true),
    )

    return MetricsResult(
        metrics=results,
        confusion_matrix=conf_matrix,
        class_mapping=class_map,
        classification_report=report,
    )


def evaluate_model(
    model: Any, features: pl.DataFrame, target: pl.Series, config: EvaluationConfig
) -> MetricsResult:
    logger.info(
        "evaluating_model",
        feature_count=features.width,
        sample_count=features.height,
        metrics=config.metrics,
        average=config.average,
    )

    # Convert polars dataframe/series to numpy arrays
    X = features.to_numpy()
    y_true = target.to_numpy()

    # Get predictions from model
    y_pred = model.predict(X)

    # Check if model supports probability predictions for ROC AUC
    if "roc_auc" in config.metrics:
        try:
            model.predict_proba
        except (AttributeError, NotImplementedError):
            logger.warning("model_does_not_support_predict_proba")

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, config)

    logger.info(
        "model_evaluation_complete",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
        f1=metrics.metrics.get("f1"),
    )

    return metrics
