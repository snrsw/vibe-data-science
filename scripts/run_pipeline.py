#!/usr/bin/env python3
"""
Script to run a pipeline with a specific configuration.
Use as follows:

python scripts/run_pipeline.py configs/pipelines/optimized_random_forest.yaml
"""

import sys
from vibe_data_science.main import run_command
from vibe_data_science.main import Settings

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pipeline.py <config_path_or_name>")
        sys.exit(1)

    config_name_or_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    settings = Settings(
        command="run", config_name_or_path=config_name_or_path, output_dir=output_dir
    )

    run_command(settings)

