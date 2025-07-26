import pytest
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from vibe_data_science.ml.models.classification import (
    ModelConfig,
    train_model,
    predict,
    _create_model_instance,
)


class TestModelTraining:
    @pytest.fixture
    def sample_data(self) -> tuple[pl.DataFrame, pl.Series]:
        # Create a simple dataset with features and target
        features_data = {
            "culmen_length_mm": [
                39.1,
                46.1,
                39.5,
                46.5,
                39.5,
                42.0,
                37.9,
                45.6,
                46.1,
                41.5,
            ],
            "culmen_depth_mm": [
                18.7,
                13.2,
                17.4,
                17.9,
                17.4,
                15.5,
                17.8,
                14.8,
                15.2,
                16.3,
            ],
            "flipper_length_mm": [181, 211, 186, 192, 186, 201, 174, 208, 213, 196],
            "body_mass_g": [3750, 4500, 3800, 3500, 3800, 4000, 3400, 4200, 4500, 3900],
        }
        features_df = pl.DataFrame(features_data)

        # Target: 0=Adelie, 1=Gentoo, 2=Chinstrap
        target_data = [0, 1, 0, 2, 0, 1, 0, 1, 1, 2]
        target_series = pl.Series("species", target_data)

        return features_df, target_series

    def test_model_config_validation(self) -> None:
        valid_config = ModelConfig(
            model_type="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
        )
        assert valid_config.model_type == "random_forest"
        assert valid_config.hyperparameters["n_estimators"] == 100

        config = ModelConfig(model_type="random_forest", hyperparameters={})
        with pytest.raises(ValueError):
            config_dict = config.model_dump()
            config_dict["model_type"] = "invalid_model_type"
            _create_model_instance(ModelConfig.model_validate(config_dict))

    def test_train_model(self, sample_data: tuple[pl.DataFrame, pl.Series]) -> None:
        features_df, target_series = sample_data

        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 5},
        )

        trained_model = train_model(features_df, target_series, config)

        # Check that the model is of the correct type
        assert isinstance(trained_model, RandomForestClassifier)

        # Check that hyperparameters were set correctly
        assert trained_model.n_estimators == 10
        assert trained_model.max_depth == 5

        # Check that the model was actually trained (has feature importances)
        assert hasattr(trained_model, "feature_importances_")
        assert len(trained_model.feature_importances_) == features_df.width

    def test_predict(self, sample_data: tuple[pl.DataFrame, pl.Series]) -> None:
        features_df, target_series = sample_data

        config = ModelConfig(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "random_state": 42},
        )

        # Train the model
        model = train_model(features_df, target_series, config)

        # Test prediction on the same data
        predictions = predict(model, features_df)

        # Check prediction shape and content
        assert len(predictions) == len(features_df)
        assert all(p in [0, 1, 2] for p in predictions)
