"""Build daily sales trends, rolling statistics, and a baseline forecast."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_series(df: pd.DataFrame, window: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily revenue; forecast using the latest rolling mean."""
    if not {"date", "revenue"}.issubset(df.columns):
        raise ValueError("CSV requires date and revenue columns")
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive")
    data = df.drop_duplicates().copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data = data.dropna(subset=["date", "revenue"])
    if data.empty:
        raise ValueError("No valid dated revenue records")
    daily = (data.assign(date=data["date"].dt.normalize())
             .groupby("date")["revenue"].sum().sort_index()
             .asfreq("D", fill_value=0).to_frame())
    daily["rolling_mean"] = daily["revenue"].rolling(window, min_periods=1).mean()
    daily["rolling_std"] = daily["revenue"].rolling(window, min_periods=2).std()
    daily["day_over_day_pct"] = daily["revenue"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    daily["cumulative_revenue"] = daily["revenue"].cumsum()
    forecast_value = float(daily["revenue"].tail(window).mean())
    future = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame({"date": future, "baseline_forecast": forecast_value})
    return daily.reset_index(), forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trends"))
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()
    daily, forecast = build_series(pd.read_csv(args.csv), args.window, args.horizon)
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "daily_trends.csv", index=False)
    forecast.to_csv(args.output / "baseline_forecast.csv", index=False)
    print(f"Analyzed {len(daily)} days; forecast {len(forecast)} days using a {args.window}-day trailing average.")
    print("Note: baseline forecast is illustrative, not a validated predictive model.")


if __name__ == "__main__":
    main()
