from unittest.mock import patch

import pytest

from vibe_data_science.ml.pipeline.experiment import (
    ExperimentConfig,
    log_experiment,
    log_model,
)
from vibe_data_science.ml.models.evaluation import MetricsResult


class TestExperiment:
    @pytest.fixture
    def sample_metrics(self) -> MetricsResult:
        return MetricsResult(
            metrics={"accuracy": 0.85, "precision": 0.82, "recall": 0.81, "f1": 0.80},
            confusion_matrix=[[5, 1, 0], [1, 4, 0], [0, 1, 4]],
            class_mapping={0: "Adelie", 1: "Gentoo", 2: "Chinstrap"},
            classification_report="Classification report",
        )

    def test_experiment_config_validation(self) -> None:
        valid_config = ExperimentConfig(
            experiment_name="penguin_classification",
            run_name="random_forest_run",
            tracking_uri=None,
        )
        assert valid_config.experiment_name == "penguin_classification"
        assert valid_config.run_name == "random_forest_run"
        assert valid_config.tracking_uri is None

    @patch("mlflow.log_metric")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    def test_log_experiment(
        self, mock_start_run, mock_set_experiment, mock_log_metric, sample_metrics
    ) -> None:
        config = ExperimentConfig(
            experiment_name="penguin_classification_test", run_name="test_run"
        )

        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = lambda self: None
        mock_start_run.return_value.__exit__ = lambda self, *args: None

        params = {"n_estimators": 100, "max_depth": 5}

        log_experiment(config, params, sample_metrics)

        # Check that mlflow was used correctly
        mock_set_experiment.assert_called_once_with("penguin_classification_test")
        mock_start_run.assert_called_once_with(run_name="test_run")

        # Check that metrics were logged
        expected_calls = [
            (("accuracy", 0.85), {}),
            (("precision", 0.82), {}),
            (("recall", 0.81), {}),
            (("f1", 0.80), {}),
        ]
        assert mock_log_metric.call_count == 4
        for i, (args, kwargs) in enumerate(mock_log_metric.call_args_list):
            assert args == expected_calls[i][0]

    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.start_run")
    def test_log_model(self, mock_start_run, mock_log_model) -> None:
        # Mock model
        class MockModel:
            def predict(self, X):
                return [0, 1, 2]

        model = MockModel()

        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = lambda self: None
        mock_start_run.return_value.__exit__ = lambda self, *args: None

        # Call function
        log_model(model, "random_forest_model")

        # Check that model was logged
        mock_log_model.assert_called_once()
