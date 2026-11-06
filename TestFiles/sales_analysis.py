"""Summarize product, regional, and monthly sales from a transaction CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"date", "region", "product", "units", "unit_price"}


def analyze(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Calculate sales KPIs and breakdowns, verifying existing revenue values."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = df.copy().drop_duplicates()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ("units", "unit_price"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "region", "product", "units", "unit_price"])
    data = data.loc[(data["units"] >= 0) & (data["unit_price"] >= 0)].copy()
    if data.empty:
        raise ValueError("No valid sales transactions remain")
    data["calculated_revenue"] = data["units"] * data["unit_price"]
    data["month"] = data["date"].dt.to_period("M").astype(str)
    groupings = {"by_product": ["product"], "by_region": ["region"],
                 "by_month": ["month"], "by_region_product": ["region", "product"]}
    results = {}
    for label, cols in groupings.items():
        results[label] = (data.groupby(cols, dropna=False)
                          .agg(transactions=("units", "size"), units_sold=("units", "sum"),
                               revenue=("calculated_revenue", "sum"))
                          .reset_index().sort_values("revenue", ascending=False))
    summary = {"transactions": int(len(data)), "total_units": float(data["units"].sum()),
               "total_revenue": round(float(data["calculated_revenue"].sum()), 2),
               "average_transaction_value": round(float(data["calculated_revenue"].mean()), 2),
               "unique_products": int(data["product"].nunique()),
               "unique_regions": int(data["region"].nunique())}
    if "revenue" in data.columns:
        recorded = pd.to_numeric(data["revenue"], errors="coerce")
        summary["revenue_mismatch_rows"] = int((recorded.notna() & ~np.isclose(recorded.fillna(0), data["calculated_revenue"], atol=0.01)).sum())
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sales"))
    args = parser.parse_args()
    tables, summary = analyze(pd.read_csv(args.csv))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "sales_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
