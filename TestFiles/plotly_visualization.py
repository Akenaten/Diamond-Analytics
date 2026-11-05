"""Generate interactive Plotly visualizations for a CSV dataset.

Usage:
    python plotly_visualization.py --csv test_data.csv --output outputs/charts
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    data = pd.read_csv(csv_path, low_memory=False)
    if data.empty:
        raise ValueError("CSV contains no data rows")
    data.columns = data.columns.astype(str).str.strip()
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names after whitespace normalization")
    return data


def save_chart(fig: go.Figure, filename: str, output: Path) -> None:
    fig.update_layout(template="plotly_white", title_x=0.5,
                      margin=dict(l=60, r=35, t=80, b=60))
    fig.write_html(output / filename, include_plotlyjs=True, full_html=True)
    LOGGER.info("Created %s", filename)


def histogram_charts(data: pd.DataFrame, numeric: list[str], output: Path) -> None:
    for index, column in enumerate(numeric[:4], start=1):
        fig = px.histogram(data, x=column, nbins=25, marginal="box",
                           title=f"Distribution of {column}")
        save_chart(fig, f"{index:02d}_histogram.html", output)


def scatter_chart(data: pd.DataFrame, numeric: list[str], categories: list[str],
                  output: Path) -> None:
    if len(numeric) < 2:
        return
    color = categories[0] if categories else None
    fig = px.scatter(data, x=numeric[0], y=numeric[1], color=color,
                     hover_data=data.columns[:5].tolist(),
                     title=f"{numeric[1]} vs {numeric[0]}", opacity=0.8)
    save_chart(fig, "10_scatter.html", output)


def box_chart(data: pd.DataFrame, numeric: list[str], categories: list[str],
              output: Path) -> None:
    if not numeric:
        return
    x = categories[0] if categories else None
    fig = px.box(data, x=x, y=numeric[0], points="outliers",
                 title=f"{numeric[0]} spread" + (f" by {x}" if x else ""))
    save_chart(fig, "11_boxplot.html", output)


def category_chart(data: pd.DataFrame, categories: list[str], output: Path) -> None:
    if not categories:
        return
    column = categories[0]
    counts = (data[column].fillna("Missing").astype(str)
              .value_counts().head(15).rename_axis(column).reset_index(name="count"))
    fig = px.bar(counts, x=column, y="count", text="count",
                 title=f"Most frequent {column} values")
    fig.update_xaxes(tickangle=-30)
    save_chart(fig, "12_categories.html", output)


def correlation_chart(data: pd.DataFrame, numeric: list[str], output: Path) -> None:
    if len(numeric) < 2:
        return
    matrix = data[numeric].corr().round(2)
    fig = px.imshow(matrix, text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title="Numeric correlation matrix")
    save_chart(fig, "13_correlations.html", output)


def time_series_chart(data: pd.DataFrame, numeric: list[str], output: Path) -> None:
    if not numeric:
        return
    candidates = [c for c in data.columns if any(s in c.lower() for s in ("date", "timestamp"))]
    for column in candidates:
        dates = pd.to_datetime(data[column], errors="coerce", utc=True)
        if dates.notna().sum() < 2:
            continue
        temp = pd.DataFrame({"date": dates, "value": pd.to_numeric(data[numeric[0]], errors="coerce")})
        temp = temp.dropna().sort_values("date")
        if temp.empty:
            continue
        daily = temp.set_index("date")["value"].resample("D").mean().reset_index()
        fig = px.line(daily, x="date", y="value", markers=True,
                      title=f"Daily mean {numeric[0]}")
        save_chart(fig, "14_time_series.html", output)
        break


def generate_charts(csv_path: Path, output: Path) -> None:
    data = load_data(csv_path)
    output.mkdir(parents=True, exist_ok=True)
    numeric = data.select_dtypes(include="number").columns.tolist()
    categories = [col for col in data.select_dtypes(include=["object", "string", "category"]).columns
                  if 2 <= data[col].nunique(dropna=True) <= 20]
    histogram_charts(data, numeric, output)
    scatter_chart(data, numeric, categories, output)
    box_chart(data, numeric, categories, output)
    category_chart(data, categories, output)
    correlation_chart(data, numeric, output)
    time_series_chart(data, numeric, output)
    if not list(output.glob("*.html")):
        LOGGER.warning("No compatible numeric or categorical columns were found")
    else:
        LOGGER.info("Interactive HTML charts saved to %s", output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/charts"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        generate_charts(args.csv, args.output)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
