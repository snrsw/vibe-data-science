#!/usr/bin/env python3
import sys
import structlog
from pathlib import Path

from vibe_data_science.ml.pipeline import (
    initialize_configs,
    get_typed_config,
    run_pipeline,
    PipelineConfig,
)


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()


def load_config_from_yaml(filepath: str) -> PipelineConfig:
    path = Path(filepath)
    if not path.exists():
        logger.error("config_file_not_found", filepath=str(path))
        raise FileNotFoundError(f"Config file not found: {filepath}")

    try:
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = PipelineConfig.model_validate(config_dict)
        logger.info("config_loaded_from_yaml", filepath=str(path))
        return config
    except Exception as e:
        logger.error("config_load_failed", filepath=str(path), error=str(e))
        raise ValueError(f"Failed to load config from {filepath}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vibe_data_science.ml.run_config <config_name_or_path>")
        sys.exit(1)

    config_name_or_path = sys.argv[1]
    initialize_configs()

    try:
        if config_name_or_path.endswith((".yaml", ".yml")):
            config = load_config_from_yaml(config_name_or_path)
        else:
            pipeline_config = get_typed_config(
                "pipeline", config_name_or_path, PipelineConfig
            )
            if not pipeline_config:
                logger.error("config_not_found", name=config_name_or_path)
                sys.exit(1)
            config = pipeline_config
    except Exception as e:
        logger.error("config_load_error", error=str(e))
        sys.exit(1)

    logger.info("running_pipeline", config_name=config_name_or_path)
    model, metrics = run_pipeline(config)

    logger.info(
        "pipeline_complete",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
    )


if __name__ == "__main__":
    main()
