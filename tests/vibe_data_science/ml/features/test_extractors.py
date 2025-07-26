import pytest
import polars as pl

from vibe_data_science.ml.features.extractors import (
    FeatureConfig,
    extract_features,
    normalize_features,
)


class TestFeatureExtraction:
    @pytest.fixture
    def preprocessed_df(self) -> pl.DataFrame:
        data = {
            "species": ["Adelie", "Gentoo", "Adelie", "Chinstrap", "Adelie"],
            "species_encoded": [0, 1, 0, 2, 0],
            "island": ["Torgersen", "Biscoe", "Dream", "Dream", "Torgersen"],
            "island_encoded": [0, 1, 2, 2, 0],
            "culmen_length_mm": [39.1, 46.1, 39.5, 46.5, 39.5],
            "culmen_depth_mm": [18.7, 13.2, 17.4, 17.9, 17.4],
            "flipper_length_mm": [181, 211, 186, 192, 186],
            "body_mass_g": [3750, 4500, 3800, 3500, 3800],
            "sex": ["MALE", "MALE", "FEMALE", "FEMALE", "FEMALE"],
            "sex_encoded": [0, 0, 1, 1, 1],
        }
        return pl.DataFrame(data)

    def test_feature_config_validation(self) -> None:
        valid_config = FeatureConfig(
            features_to_use=("culmen_length_mm", "culmen_depth_mm", "species_encoded"),
            normalize=True,
        )
        assert valid_config.features_to_use == (
            "culmen_length_mm",
            "culmen_depth_mm",
            "species_encoded",
        )
        assert valid_config.normalize is True

    def test_normalize_features(self, preprocessed_df: pl.DataFrame) -> None:
        numeric_cols = (
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        )

        normalized_df = normalize_features(preprocessed_df, numeric_cols)

        for col in numeric_cols:
            assert f"{col}_norm" in normalized_df.columns

        for col in numeric_cols:
            norm_col = normalized_df[f"{col}_norm"]
            mean_val = norm_col.mean()
            std_val = norm_col.std()

            if mean_val is not None:
                assert abs(float(mean_val)) < 1e-10

            if std_val is not None:
                assert abs(float(std_val) - 1.0) < 1e-10

        for col in numeric_cols:
            assert col in normalized_df.columns

    def test_extract_features(self, preprocessed_df: pl.DataFrame) -> None:
        config = FeatureConfig(
            features_to_use=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
                "species_encoded",
                "island_encoded",
                "sex_encoded",
            ),
            normalize=True,
        )

        feature_df = extract_features(preprocessed_df, config)

        assert "culmen_length_mm_norm" in feature_df.columns

        all_expected_columns = [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "species_encoded",
            "island_encoded",
            "sex_encoded",
            "culmen_length_mm_norm",
            "culmen_depth_mm_norm",
            "flipper_length_mm_norm",
            "body_mass_g_norm",
        ]

        for col in all_expected_columns:
            assert col in feature_df.columns

        assert "species" not in feature_df.columns
