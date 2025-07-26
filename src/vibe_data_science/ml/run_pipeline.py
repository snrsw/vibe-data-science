#!/usr/bin/env python3
import structlog
from vibe_data_science.ml.data.loader import DatasetConfig
from vibe_data_science.ml.data.preprocessing import PreprocessingConfig
from vibe_data_science.ml.features.extractors import FeatureConfig
from vibe_data_science.ml.models.classification import ModelConfig
from vibe_data_science.ml.models.evaluation import EvaluationConfig
from vibe_data_science.ml.pipeline.experiment import ExperimentConfig
from vibe_data_science.ml.pipeline.training import (
    PipelineConfig,
    SplitConfig,
    run_pipeline,
)


# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


def main() -> None:
    logger.info("starting_penguin_classification_pipeline")

    # Define configuration
    pipeline_config = PipelineConfig(
        dataset=DatasetConfig(
            filepath="data/penguins_size.csv",
            target_column="species",
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
        model=ModelConfig(
            model_type="random_forest",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
        ),
        evaluation=EvaluationConfig(
            metrics=("accuracy", "precision", "recall", "f1"),
            average="macro",
        ),
        split=SplitConfig(
            test_ratio=0.2,
            validation_ratio=0.0,  # Not using validation set in this example
            random_seed=42,
        ),
        experiment=ExperimentConfig(
            experiment_name="penguin_classification",
            run_name="random_forest_baseline",
        ),
    )

    # Run the pipeline
    _, metrics = run_pipeline(pipeline_config)

    # Print results
    logger.info(
        "pipeline_results",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
        f1=metrics.metrics.get("f1"),
    )


if __name__ == "__main__":
    main()
