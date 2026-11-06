"""Audit CSV completeness, duplicates, type consistency, and numeric outliers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return per-column metrics, IQR outliers, and a dataset-level report."""
    columns = []
    outliers = []
    for name in df.columns:
        series = df[name]
        columns.append({"column": name, "dtype": str(series.dtype),
                        "missing_count": int(series.isna().sum()),
                        "missing_percent": round(series.isna().mean() * 100, 2),
                        "unique_count": int(series.nunique(dropna=True)),
                        "constant": bool(series.nunique(dropna=True) <= 1)})
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for index, value in values[(values < lower) | (values > upper)].items():
                outliers.append({"row_index": int(index), "column": name,
                                 "value": float(value), "lower_bound": float(lower),
                                 "upper_bound": float(upper)})
    report = {"rows": int(len(df)), "columns": int(len(df.columns)),
              "duplicate_rows": int(df.duplicated().sum()),
              "missing_cells": int(df.isna().sum().sum()),
              "outlier_cells": len(outliers)}
    return pd.DataFrame(columns), pd.DataFrame(outliers, columns=["row_index", "column", "value", "lower_bound", "upper_bound"]), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/quality"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("CSV must contain at least one record")
    args.output.mkdir(parents=True, exist_ok=True)
    profile, outliers, report = audit(df)
    profile.to_csv(args.output / "column_quality.csv", index=False)
    outliers.to_csv(args.output / "outliers.csv", index=False)
    (args.output / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
