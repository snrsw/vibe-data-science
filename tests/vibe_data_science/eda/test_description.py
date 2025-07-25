from pathlib import Path

import pytest


from vibe_data_science.eda.description import (
    analyze_dataset,
    generate_markdown_report,
    ColumnStats,
    DatasetDescription,
)


class TestColumnStats:
    def test_column_stats_model(self):
        stats = ColumnStats(
            name="test_column",
            dtype="str",
            count=10,
            null_count=2,
        )

        assert stats.name == "test_column"
        assert stats.dtype == "str"
        assert stats.count == 10
        assert stats.null_count == 2
        assert stats.unique_count is None


class TestDatasetDescription:
    def test_dataset_description_model(self):
        column = ColumnStats(
            name="test_column",
            dtype="str",
            count=10,
            null_count=2,
        )

        description = DatasetDescription(
            name="test_dataset",
            path=Path("data/test.csv"),
            row_count=10,
            column_count=1,
            columns=[column],
        )

        assert description.name == "test_dataset"
        assert description.path == Path("data/test.csv")
        assert description.row_count == 10
        assert description.column_count == 1
        assert len(description.columns) == 1
        assert description.columns[0].name == "test_column"


class TestAnalyzeDataset:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        file_path = tmp_path / "sample.csv"
        sample_data = """species,island,culmen_length_mm,culmen_depth_mm
Adelie,Torgersen,39.1,18.7
Adelie,Torgersen,39.5,17.4
Chinstrap,Dream,46.5,17.9
Gentoo,Biscoe,46.1,13.2
NA,NA,NA,NA
"""
        file_path.write_text(sample_data)
        return file_path

    def test_analyze_dataset(self, sample_csv):
        description = analyze_dataset(sample_csv)

        assert description.name == "sample"
        assert description.path == sample_csv
        assert description.row_count == 5
        assert description.column_count == 4

        column_names = [col.name for col in description.columns]
        expected_names = ["species", "island", "culmen_length_mm", "culmen_depth_mm"]
        assert column_names == expected_names

        species_col = next(col for col in description.columns if col.name == "species")
        assert species_col.count == 4
        assert species_col.null_count == 1
        assert species_col.unique_count == 4

        length_col = next(
            col for col in description.columns if col.name == "culmen_length_mm"
        )
        assert length_col.count == 4
        assert length_col.null_count == 1
        assert length_col.min_value == pytest.approx(39.1)
        assert length_col.max_value == pytest.approx(46.5)
        assert length_col.mean == pytest.approx(42.8, abs=0.1)


class TestGenerateMarkdownReport:
    def test_generate_markdown_report(self):
        column = ColumnStats(
            name="species",
            dtype="str",
            count=4,
            null_count=1,
            unique_count=3,
            most_frequent_values={"Adelie": 2, "Chinstrap": 1, "Gentoo": 1},
        )

        description = DatasetDescription(
            name="penguins",
            path=Path("data/penguins.csv"),
            row_count=5,
            column_count=1,
            columns=[column],
        )

        report = generate_markdown_report(description)

        assert "# Dataset: penguins" in report
        assert "## Columns" in report
        assert "### species" in report
        assert "- **Type**: str" in report
        assert "- **Count**: 4" in report
        assert "- **Null Count**: 1" in report
        assert "- **Most Frequent Values**:" in report
