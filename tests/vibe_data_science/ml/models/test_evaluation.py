import pytest
import numpy as np
import polars as pl

from vibe_data_science.ml.models.evaluation import (
    EvaluationConfig,
    evaluate_model,
    calculate_metrics,
)


class TestEvaluation:
    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Create true labels and predictions
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 0])
        return y_true, y_pred

    def test_evaluation_config_validation(self) -> None:
        valid_config = EvaluationConfig(
            metrics=("accuracy", "precision", "recall"), average="macro"
        )
        assert valid_config.metrics == ("accuracy", "precision", "recall")
        assert valid_config.average == "macro"

    def test_calculate_metrics(
        self, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = sample_data

        config = EvaluationConfig(
            metrics=("accuracy", "precision", "recall"), average="macro"
        )

        metrics = calculate_metrics(y_true, y_pred, config)

        assert "accuracy" in metrics.metrics
        assert "precision" in metrics.metrics
        assert "recall" in metrics.metrics

        correct = (y_true == y_pred).sum()
        expected_accuracy = correct / len(y_true)
        assert metrics.metrics["accuracy"] == pytest.approx(expected_accuracy)

        assert 0 <= metrics.metrics["precision"] <= 1
        assert 0 <= metrics.metrics["recall"] <= 1

    def test_evaluate_model(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = sample_data

        # Convert to polars series
        y_true_series = pl.Series("true", y_true)

        config = EvaluationConfig(
            metrics=("accuracy", "precision", "recall"), average="macro"
        )

        class MockModel:
            def predict(self, X):
                return y_pred

        mock_model = MockModel()

        features = pl.DataFrame(
            {"feature1": [0] * len(y_true), "feature2": [1] * len(y_true)}
        )

        metrics = evaluate_model(mock_model, features, y_true_series, config)

        assert "accuracy" in metrics.metrics
        assert "precision" in metrics.metrics
        assert "recall" in metrics.metrics

        assert metrics.confusion_matrix is not None
        assert len(metrics.confusion_matrix) == 3
        assert all(len(row) == 3 for row in metrics.confusion_matrix)

        assert metrics.class_mapping is not None
        assert len(metrics.class_mapping) == 3
