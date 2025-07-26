from typing import Sequence

import polars as pl
import structlog
from pydantic import BaseModel


logger = structlog.get_logger()


class FeatureConfig(BaseModel):
    features_to_use: tuple[str, ...]
    normalize: bool = True

    model_config = {"frozen": True}


def normalize_single_column(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    if column_name not in df.columns:
        logger.warning("column_not_found_for_normalization", column=column_name)
        return df

    col_mean = df[column_name].mean()
    col_std = df[column_name].std()

    if col_std == 0:
        logger.warning("zero_std_dev_for_column", column=column_name)
        normalized_col = pl.lit(0).alias(f"{column_name}_norm")
    else:
        normalized_col = ((pl.col(column_name) - col_mean) / col_std).alias(
            f"{column_name}_norm"
        )

    logger.info(
        "normalized_feature", column=column_name, mean=col_mean, std_dev=col_std
    )

    return df.with_columns(normalized_col)


def normalize_features(
    df: pl.DataFrame, numeric_columns: Sequence[str]
) -> pl.DataFrame:
    normalized_df = df

    for column in numeric_columns:
        normalized_df = normalize_single_column(normalized_df, column)

    return normalized_df


def is_numeric_column(series: pl.Series) -> bool:
    return series.dtype in [
        pl.Int64,
        pl.Int32,
        pl.Int16,
        pl.Int8,
        pl.UInt64,
        pl.UInt32,
        pl.UInt16,
        pl.UInt8,
        pl.Float64,
        pl.Float32,
    ]


def filter_valid_columns(
    df: pl.DataFrame, columns: Sequence[str]
) -> tuple[list[str], list[str]]:
    valid_columns = []
    numeric_columns = []

    for col in columns:
        if col not in df.columns:
            logger.warning("column_not_found", column=col)
            continue

        valid_columns.append(col)

        if is_numeric_column(df[col]):
            numeric_columns.append(col)

    return valid_columns, numeric_columns


def extract_features(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    logger.info(
        "extracting_features",
        feature_count=len(config.features_to_use),
        normalize=config.normalize,
    )

    valid_columns, numeric_columns = filter_valid_columns(df, config.features_to_use)

    result = df.select(valid_columns).pipe(
        lambda df: normalize_features(df, numeric_columns)
        if config.normalize and numeric_columns
        else df
    )

    logger.info(
        "feature_extraction_complete",
        input_columns=len(df.columns),
        output_columns=len(result.columns),
    )

    return result
