"""Explore, validate, clean, and summarize a CSV dataset with pandas.

Usage:
    python pandas_analysis.py --csv test_data.csv --output outputs/pandas
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV and reject empty files or files without records."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError("CSV contains no data rows")
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after whitespace normalization")
    return frame


def infer_date_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse columns whose names suggest dates, leaving others unchanged."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        if any(token in column.lower() for token in ("date", "timestamp", "_at")):
            parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
            non_null = result[column].notna().sum()
            if non_null and parsed.notna().sum() / non_null >= 0.8:
                result[column] = parsed
    return result


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trim text, normalize blank strings, and remove exact duplicate rows."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
    before = len(result)
    result = result.drop_duplicates().reset_index(drop=True)
    result = infer_date_columns(result)
    return result, {"original_rows": before, "cleaned_rows": len(result),
                    "duplicates_removed": before - len(result)}


def column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return type, completeness, cardinality, and sample values per column."""
    records = []
    for name in frame.columns:
        series = frame[name]
        examples = series.dropna().astype(str).unique()[:3].tolist()
        records.append({"column": name, "dtype": str(series.dtype),
                        "non_null": int(series.notna().sum()),
                        "missing": int(series.isna().sum()),
                        "missing_pct": round(float(series.isna().mean() * 100), 2),
                        "unique_values": int(series.nunique(dropna=True)),
                        "examples": " | ".join(examples)})
    return pd.DataFrame(records)


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numbers = frame.select_dtypes(include="number")
    if numbers.empty:
        return pd.DataFrame()
    summary = numbers.describe().T
    summary["median"] = numbers.median()
    summary["missing"] = numbers.isna().sum()
    summary["skew"] = numbers.skew()
    return summary.round(4)


def categorical_summary(frame: pd.DataFrame, max_categories: int = 15) -> pd.DataFrame:
    """Summarize the most frequent values in nonnumeric columns."""
    rows = []
    for column in frame.select_dtypes(exclude="number").columns:
        counts = frame[column].astype("string").value_counts(dropna=False).head(max_categories)
        for value, count in counts.items():
            rows.append({"column": column, "value": str(value), "count": int(count),
                         "percent": round(100 * int(count) / len(frame), 2)})
    return pd.DataFrame(rows, columns=["column", "value", "count", "percent"])


def grouped_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric values by the first low-cardinality categorical column."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categories = [c for c in frame.select_dtypes(include=["object", "string", "category"]).columns
                  if 1 < frame[c].nunique(dropna=True) <= 30]
    if not numeric or not categories:
        return pd.DataFrame()
    category = categories[0]
    return (frame.groupby(category, dropna=False)[numeric]
            .agg(["count", "mean", "median", "min", "max"])
            .round(3))


def write_report(csv_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_dataset(csv_path)
    cleaned, audit = clean_dataset(original)
    LOGGER.info("Loaded %s rows and %s columns", len(original), len(original.columns))
    cleaned.to_csv(output_dir / "cleaned_data.csv", index=False)
    column_profile(cleaned).to_csv(output_dir / "column_profile.csv", index=False)
    numeric_summary(cleaned).to_csv(output_dir / "numeric_summary.csv")
    categorical_summary(cleaned).to_csv(output_dir / "categorical_summary.csv", index=False)
    grouped_summary(cleaned).to_csv(output_dir / "grouped_summary.csv")
    correlations = cleaned.select_dtypes(include="number").corr()
    correlations.to_csv(output_dir / "correlations.csv")
    audit.update({"columns": len(cleaned.columns),
                  "total_missing_cells": int(cleaned.isna().sum().sum()),
                  "numeric_columns": cleaned.select_dtypes(include="number").columns.tolist()})
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    LOGGER.info("Saved analysis to %s", output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pandas"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        write_report(args.csv, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
