from typing import Any, Literal

import polars as pl
import structlog
from pydantic import BaseModel, Field, field_validator


logger = structlog.get_logger()


class PreprocessingConfig(BaseModel):
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    fill_strategy: Literal["mean", "median", "mode"] = Field(default="median")

    model_config = {"frozen": True}

    @field_validator("fill_strategy")
    @classmethod
    def validate_fill_strategy(cls, v: str) -> str:
        allowed_strategies = ["mean", "median", "mode"]
        if v not in allowed_strategies:
            raise ValueError(
                f"Fill strategy must be one of: {', '.join(allowed_strategies)}"
            )
        return v


def calculate_fill_value_for_column(series: pl.Series, strategy: str) -> Any:
    non_null_values = series.drop_nulls()

    if len(non_null_values) == 0:
        return (
            0
            if pl.Series([1]).cast(series.dtype).dtype
            in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
            else "UNKNOWN"
        )

    match strategy:
        case "mean":
            return non_null_values.mean()
        case "median":
            return non_null_values.median()
        case "mode":
            value_counts = non_null_values.value_counts()
            value_counts_sorted = value_counts.sort(
                value_counts.columns[1], descending=True
            )
            return value_counts_sorted[0, 0]
        case _:
            return 0


def impute_single_column(
    df: pl.DataFrame, column_name: str, fill_value: Any
) -> pl.DataFrame:
    return df.with_columns(pl.col(column_name).fill_null(fill_value))


def handle_missing_values(
    df: pl.DataFrame, config: PreprocessingConfig
) -> pl.DataFrame:
    imputed_df = df

    for col in config.numeric_features:
        fill_value = calculate_fill_value_for_column(df[col], config.fill_strategy)
        imputed_df = impute_single_column(imputed_df, col, fill_value)

        logger.info(
            "numeric_column_imputed",
            column=col,
            strategy=config.fill_strategy,
            fill_value=fill_value,
        )

    for col in config.categorical_features:
        fill_value = calculate_fill_value_for_column(df[col], "mode")
        imputed_df = impute_single_column(imputed_df, col, fill_value)

        logger.info("categorical_column_imputed", column=col, fill_value=fill_value)

    return imputed_df


def create_category_mapping(series: pl.Series) -> dict:
    unique_values = series.unique().drop_nulls()
    return {value: idx for idx, value in enumerate(unique_values)}


def encode_single_column(
    df: pl.DataFrame, column_name: str, mapping: dict
) -> pl.DataFrame:
    encoded_column = (
        pl.col(column_name)
        .map_elements(lambda val: mapping.get(val, -1))
        .cast(pl.Int32)
        .alias(f"{column_name}_encoded")
    )
    return df.with_columns(encoded_column)


def encode_categorical_features(
    df: pl.DataFrame, config: PreprocessingConfig
) -> pl.DataFrame:
    encoded_df = df

    for col in config.categorical_features:
        mapping = create_category_mapping(df[col])
        encoded_df = encode_single_column(encoded_df, col, mapping)

        logger.info(
            "categorical_column_encoded",
            column=col,
            unique_values=len(mapping),
            mapping=mapping,
        )

    return encoded_df


def preprocess_dataset(df: pl.DataFrame, config: PreprocessingConfig) -> pl.DataFrame:
    logger.info(
        "preprocessing_dataset",
        input_shape=df.shape,
        categorical_features=config.categorical_features,
        numeric_features=config.numeric_features,
        fill_strategy=config.fill_strategy,
    )

    result = df.pipe(handle_missing_values, config=config).pipe(
        encode_categorical_features, config=config
    )

    logger.info(
        "preprocessing_complete", input_shape=df.shape, output_shape=result.shape
    )

    return result
