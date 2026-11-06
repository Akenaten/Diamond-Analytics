"""Segment transaction records by revenue, volume, and satisfaction ratings.

Note: without a customer identifier, these are transaction-level segments,
not customer-level behavioral profiles.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"region", "product", "units", "revenue", "customer_rating"}


def segment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = df.drop_duplicates().copy()
    for col in ("units", "revenue", "customer_rating"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["units", "revenue"])
    data = data.loc[(data["units"] >= 0) & (data["revenue"] >= 0)].copy()
    if data.empty:
        raise ValueError("No valid transaction records")
    revenue_tertiles = data["revenue"].quantile([1/3, 2/3]).to_numpy()
    volume_tertiles = data["units"].quantile([1/3, 2/3]).to_numpy()
    def tier(value: float, thresholds: np.ndarray) -> str:
        return "low" if value <= thresholds[0] else "medium" if value <= thresholds[1] else "high"
    data["spend_tier"] = data["revenue"].map(lambda v: tier(v, revenue_tertiles))
    data["volume_tier"] = data["units"].map(lambda v: tier(v, volume_tertiles))
    data["rating_band"] = pd.cut(data["customer_rating"], bins=[-np.inf, 2.99, 3.99, np.inf],
                                  labels=["low", "medium", "high"])
    data["rating_band"] = data["rating_band"].astype("string").fillna("unrated")
    summary = (data.groupby(["spend_tier", "volume_tier", "rating_band"], dropna=False)
               .agg(records=("revenue", "size"), total_revenue=("revenue", "sum"),
                    mean_rating=("customer_rating", "mean"))
               .reset_index().sort_values("records", ascending=False))
    metadata = {"segmentation_level": "transaction", "records": int(len(data)),
                "spend_thresholds": revenue_tertiles.tolist(),
                "volume_thresholds": volume_tertiles.tolist(),
                "note": "Customer-level segmentation requires a customer_id column."}
    return data, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/segments"))
    args = parser.parse_args()
    data, summary, metadata = segment(pd.read_csv(args.csv))
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "segmented_transactions.csv", index=False)
    summary.to_csv(args.output / "segment_summary.csv", index=False)
    (args.output / "segment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
