from pathlib import Path

import pytest
import polars as pl

from vibe_data_science.ml.data.loader import load_dataset, DatasetConfig


class TestDataLoader:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        file_path = tmp_path / "sample.csv"
        sample_data = """species,island,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g,sex
Adelie,Torgersen,39.1,18.7,181,3750,MALE
Adelie,Torgersen,39.5,17.4,186,3800,FEMALE
Chinstrap,Dream,46.5,17.9,192,3500,FEMALE
Gentoo,Biscoe,46.1,13.2,211,4500,MALE
NA,NA,NA,NA,NA,NA,NA
"""
        file_path.write_text(sample_data)
        return file_path

    def test_dataset_config_validation(self) -> None:
        valid_config = DatasetConfig(
            filepath="data/penguins_size.csv", target_column="species"
        )
        assert valid_config.filepath == "data/penguins_size.csv"
        assert valid_config.target_column == "species"

        config_with_empty_target = DatasetConfig(
            filepath="data/penguins_size.csv", target_column=""
        )
        assert config_with_empty_target.filepath == "data/penguins_size.csv"
        assert config_with_empty_target.target_column == ""

    def test_load_dataset(self, sample_csv: Path) -> None:
        config = DatasetConfig(filepath=str(sample_csv), target_column="species")
        dataset = load_dataset(config)

        assert isinstance(dataset, pl.DataFrame)
        assert dataset.shape == (5, 7)
        assert dataset.columns == [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]

    def test_nonexistent_file(self) -> None:
        config = DatasetConfig(filepath="nonexistent_file.csv", target_column="species")
        with pytest.raises(FileNotFoundError):
            load_dataset(config)
