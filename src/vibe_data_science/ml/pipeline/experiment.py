from typing import Any, Optional

import json
import mlflow
import mlflow.sklearn
import structlog
from pydantic import BaseModel

from vibe_data_science.ml.models.evaluation import MetricsResult


logger = structlog.get_logger()


class ExperimentConfig(BaseModel):
    experiment_name: str
    run_name: str
    tracking_uri: Optional[str] = None

    model_config = {"frozen": True}


def log_experiment(
    config: ExperimentConfig,
    params: dict[str, Any],
    metrics_result: MetricsResult,
) -> None:
    logger.info(
        "logging_experiment", experiment=config.experiment_name, run=config.run_name
    )

    # Set tracking URI if provided
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    # Set experiment
    mlflow.set_experiment(config.experiment_name)

    # Start a run
    with mlflow.start_run(run_name=config.run_name):
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics_result.metrics.items():
            mlflow.log_metric(key, value)

        # Log artifacts
        if metrics_result.confusion_matrix:
            confusion_matrix_json = json.dumps(metrics_result.confusion_matrix)
            with open("confusion_matrix.json", "w") as f:
                f.write(confusion_matrix_json)
            mlflow.log_artifact("confusion_matrix.json")

        if metrics_result.classification_report:
            with open("classification_report.txt", "w") as f:
                f.write(metrics_result.classification_report)
            mlflow.log_artifact("classification_report.txt")

    logger.info(
        "experiment_logged", experiment=config.experiment_name, run=config.run_name
    )


def log_model(model: Any, name: str) -> None:
    logger.info("logging_model", model_name=name)

    with mlflow.start_run(run_name=f"{name}_log"):
        mlflow.sklearn.log_model(model, name)

    logger.info("model_logged", model_name=name)
