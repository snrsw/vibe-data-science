from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel


class ColumnStats(BaseModel):
    name: str
    dtype: str
    count: int
    null_count: int
    unique_count: int | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    mean: float | None = None
    median: float | None = None
    std_dev: float | None = None
    mode: Any | None = None
    most_frequent_values: dict[Any, int] | None = None


class DatasetDescription(BaseModel):
    name: str
    path: Path
    row_count: int
    column_count: int
    columns: list[ColumnStats]


def analyze_dataset(filepath: str | Path) -> DatasetDescription:
    path = Path(filepath)
    df = pl.read_csv(path, null_values=["NA", "."])

    row_count = len(df)
    column_count = len(df.columns)

    columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype = str(col.dtype)
        count = row_count - col.null_count()
        null_count = col.null_count()

        col_stats = ColumnStats(
            name=col_name,
            dtype=dtype,
            count=count,
            null_count=null_count,
        )

        if isinstance(
            col.dtype,
            (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ),
        ):
            non_null_values = col.drop_nulls()
            if len(non_null_values) > 0:
                try:
                    col_stats.min_value = float(non_null_values.min())
                    col_stats.max_value = float(non_null_values.max())
                    col_stats.mean = float(non_null_values.mean())
                    col_stats.median = float(non_null_values.median())
                    col_stats.std_dev = float(non_null_values.std())
                except (TypeError, ValueError):
                    pass

        value_frequency = col.value_counts()
        col_stats.unique_count = len(value_frequency)

        if len(value_frequency) > 0:
            valueColumnName = value_frequency.columns[0]

            top_values = {}
            for row in value_frequency.iter_rows():
                val, count = row
                if val is not None:
                    top_values[val] = count
                    if len(top_values) >= 5:
                        break

            col_stats.most_frequent_values = top_values

            for val in value_frequency[valueColumnName]:
                if val is not None:
                    col_stats.mode = val
                    break

        columns.append(col_stats)

    return DatasetDescription(
        name=path.stem,
        path=path,
        row_count=row_count,
        column_count=column_count,
        columns=columns,
    )


def generate_markdown_report(description: DatasetDescription) -> str:
    report_lines = [
        f"# Dataset: {description.name}",
        "",
        f"- **File path**: {description.path}",
        f"- **Row count**: {description.row_count}",
        f"- **Column count**: {description.column_count}",
        "",
        "## Columns",
        "",
        "| Column | Type | Count | Null Count | Unique Values |",
        "| ------ | ---- | ----- | ---------- | ------------- |",
    ]

    for col in description.columns:
        report_lines.append(
            f"| {col.name} | {col.dtype} | {col.count} | {col.null_count} | {col.unique_count or 'N/A'} |"
        )

    report_lines.append("")

    for col in description.columns:
        report_lines.extend(
            [
                f"### {col.name}",
                "",
                f"- **Type**: {col.dtype}",
                f"- **Count**: {col.count}",
                f"- **Null Count**: {col.null_count}",
                f"- **Unique Values**: {col.unique_count}",
            ]
        )

        if col.min_value is not None:
            report_lines.append(f"- **Min**: {col.min_value}")

        if col.max_value is not None:
            report_lines.append(f"- **Max**: {col.max_value}")

        if col.mean is not None:
            report_lines.append(f"- **Mean**: {col.mean:.4f}")

        if col.median is not None:
            report_lines.append(f"- **Median**: {col.median}")

        if col.std_dev is not None:
            report_lines.append(f"- **Standard Deviation**: {col.std_dev:.4f}")

        if col.mode is not None:
            report_lines.append(f"- **Mode**: {col.mode}")

        if col.most_frequent_values:
            report_lines.append("- **Most Frequent Values**:")
            for val, count in col.most_frequent_values.items():
                percent = (count / col.count) * 100 if col.count > 0 else 0
                report_lines.append(f"  - {val}: {count} ({percent:.2f}%)")

        report_lines.append("")

    return "\n".join(report_lines)


def main() -> None:
    input_path = Path("data/penguins_size.csv")
    output_path = Path("docs/datasets.md")

    description = analyze_dataset(input_path)
    report = generate_markdown_report(description)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(f"Dataset analysis complete. Report saved to {output_path}")


if __name__ == "__main__":
    main()
