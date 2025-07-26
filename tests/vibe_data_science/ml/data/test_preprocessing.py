import pytest
import polars as pl

from vibe_data_science.ml.data.preprocessing import (
    PreprocessingConfig,
    preprocess_dataset,
    handle_missing_values,
    encode_categorical_features,
)


class TestPreprocessing:
    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        data = {
            "species": ["Adelie", "Gentoo", None, "Chinstrap", "Adelie"],
            "island": ["Torgersen", "Biscoe", "Dream", None, "Torgersen"],
            "culmen_length_mm": [39.1, 46.1, None, 46.5, 39.5],
            "culmen_depth_mm": [18.7, 13.2, None, 17.9, 17.4],
            "flipper_length_mm": [181, 211, None, 192, 186],
            "body_mass_g": [3750, 4500, None, 3500, 3800],
            "sex": ["MALE", "MALE", None, "FEMALE", "FEMALE"],
        }
        return pl.DataFrame(data)

    def test_preprocessing_config_validation(self) -> None:
        valid_config = PreprocessingConfig(
            categorical_features=("species", "island", "sex"),
            numeric_features=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
            fill_strategy="median",
        )
        assert valid_config.categorical_features == ("species", "island", "sex")
        assert valid_config.numeric_features == (
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        )
        assert valid_config.fill_strategy == "median"

        with pytest.raises(ValueError):
            PreprocessingConfig.validate_fill_strategy("invalid_strategy")

    def test_handle_missing_values(self, sample_df: pl.DataFrame) -> None:
        config = PreprocessingConfig(
            categorical_features=("species", "island", "sex"),
            numeric_features=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
            fill_strategy="median",
        )

        result_df = handle_missing_values(sample_df, config)

        # Check that numeric columns have no nulls
        for col in config.numeric_features:
            assert result_df[col].null_count() == 0

        # Check that categorical columns have no nulls
        for col in config.categorical_features:
            assert result_df[col].null_count() == 0

        null_row_index = 2
        filled_value = result_df[null_row_index, "culmen_length_mm"]
        assert filled_value is not None
        assert isinstance(filled_value, (int, float))

    def test_encode_categorical_features(self, sample_df: pl.DataFrame) -> None:
        config = PreprocessingConfig(
            categorical_features=("species", "island", "sex"),
            numeric_features=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
        )

        # First handle missing values to avoid encoding issues
        filled_df = handle_missing_values(sample_df, config)
        result_df = encode_categorical_features(filled_df, config)

        # Check that we have encoded categorical columns
        for col in config.categorical_features:
            assert f"{col}_encoded" in result_df.columns
            assert result_df[f"{col}_encoded"].dtype == pl.Int32

        # Check encoding is consistent
        species_encoded_values = result_df.filter(pl.col("species") == "Adelie")[
            "species_encoded"
        ]
        assert species_encoded_values[0] == species_encoded_values[1]

        # Original columns should still exist
        for col in config.categorical_features:
            assert col in result_df.columns

    def test_preprocess_dataset(self, sample_df: pl.DataFrame) -> None:
        config = PreprocessingConfig(
            categorical_features=("species", "island", "sex"),
            numeric_features=(
                "culmen_length_mm",
                "culmen_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ),
        )

        result_df = preprocess_dataset(sample_df, config)

        for col in config.numeric_features + config.categorical_features:
            assert result_df[col].null_count() == 0

        # Check that we have encoded categorical columns
        for col in config.categorical_features:
            assert f"{col}_encoded" in result_df.columns

        # Check that the result has more columns than the original due to encoding
        assert len(result_df.columns) > len(sample_df.columns)
