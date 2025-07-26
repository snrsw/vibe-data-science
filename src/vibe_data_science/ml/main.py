#!/usr/bin/env python3
import sys
import structlog
from pathlib import Path
from typing import Optional
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from vibe_data_science.ml.pipeline import (
    initialize_configs,
    get_typed_config,
    list_configs,
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


class Command(str, Enum):
    RUN = "run"
    LIST = "list"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIBE_")

    command: Command = Field(
        default=Command.RUN,
        description="Command to execute (run or list)",
    )
    config_name_or_path: str = Field(
        default="",
        description="Configuration name or YAML file path",
    )
    config_type: Optional[str] = Field(
        default=None,
        description="Configuration type to list",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save model artifacts",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )


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


def run_command(settings: Settings) -> None:
    if not settings.config_name_or_path:
        print("Error: config_name_or_path is required for the run command")
        print(
            "Usage: python -m vibe_data_science.ml.main run --config-name-or-path=<config_name_or_path>"
        )
        sys.exit(1)

    initialize_configs()

    try:
        if settings.config_name_or_path.endswith((".yaml", ".yml")):
            config = load_config_from_yaml(settings.config_name_or_path)
        else:
            pipeline_config = get_typed_config(
                "pipeline", settings.config_name_or_path, PipelineConfig
            )
            if not pipeline_config:
                logger.error("config_not_found", name=settings.config_name_or_path)
                sys.exit(1)
            config = pipeline_config
    except Exception as e:
        logger.error("config_load_error", error=str(e))
        sys.exit(1)

    logger.info("running_pipeline", config_name=settings.config_name_or_path)
    model, metrics = run_pipeline(config)

    logger.info(
        "pipeline_complete",
        accuracy=metrics.metrics.get("accuracy"),
        precision=metrics.metrics.get("precision"),
        recall=metrics.metrics.get("recall"),
        f1=metrics.metrics.get("f1"),
    )


def list_command(settings: Settings) -> None:
    initialize_configs()
    configs = list_configs(settings.config_type)

    if settings.config_type:
        print(f"Available {settings.config_type} configurations:")
        for name, config_class in configs.items():
            print(f"  - {name} ({config_class})")
    else:
        for config_type, configs_dict in configs.items():
            print(f"\n{config_type} configurations:")
            for name, config_class in configs_dict.items():
                print(f"  - {name} ({config_class})")


def main() -> None:
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--help"]

    if args[0] == "--help" or args[0] == "-h":
        print("Usage: python -m vibe_data_science.ml.main [command] [options]")
        print("\nCommands:")
        print(
            "  run   Run a machine learning pipeline with the specified configuration"
        )
        print("  list  List available configurations")
        print("\nOptions:")
        print(
            "  --config-name-or-path  Configuration name or YAML file path (for run command)"
        )
        print("  --config-type          Configuration type to list (for list command)")
        print("  --output-dir           Directory to save model artifacts")
        print("  --verbose              Enable verbose logging")
        sys.exit(0)

    command = args[0] if args[0] in [c.value for c in Command] else Command.RUN.value

    # Parse arguments is handled in the settings_dict creation below

    # Convert command-line arguments to dictionary format for pydantic settings
    settings_dict = {"command": command}

    # Extract other command-line arguments
    for arg in args[1:]:
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg.lstrip("-").split("=", 1)
                settings_dict[key] = value
            else:
                key = arg.lstrip("-")
                settings_dict[key] = True
        # Handle positional arguments
        elif (
            command == Command.RUN.value and "config_name_or_path" not in settings_dict
        ):
            settings_dict["config_name_or_path"] = arg
        elif command == Command.LIST.value and "config_type" not in settings_dict:
            settings_dict["config_type"] = arg

    settings = Settings.model_validate(settings_dict)

    if command == Command.RUN.value:
        run_command(settings)
    elif command == Command.LIST.value:
        list_command(settings)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
