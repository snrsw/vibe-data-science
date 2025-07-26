from pathlib import Path

import polars as pl
import structlog
from pydantic import BaseModel, Field


logger = structlog.get_logger()


class DatasetConfig(BaseModel):
    filepath: str
    target_column: str
    null_values: tuple[str, ...] = Field(default=("NA", "."))
    delimiter: str = Field(default=",")

    model_config = {"frozen": True}


def load_dataset(config: DatasetConfig) -> pl.DataFrame:
    filepath = Path(config.filepath)

    if not filepath.exists():
        logger.error("dataset_file_not_found", filepath=str(filepath))
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        dataset = pl.read_csv(
            source=filepath,
            null_values=config.null_values,
            separator=config.delimiter,
        )

        logger.info(
            "dataset_loaded_successfully",
            filepath=str(filepath),
            rows=dataset.shape[0],
            columns=dataset.shape[1],
        )

        return dataset
    except Exception as e:
        logger.error(
            "dataset_loading_failed",
            filepath=str(filepath),
            error=str(e),
        )
        raise
