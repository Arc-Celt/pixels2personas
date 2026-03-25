"""
Anime count per time period.

Input:
- JSONL with at least `anime_json` and `air_start_date` (YYYY-MM-DD or YYYY-MM).

Output:
- anime_per_period.pdf

Example:

python scripts/plotting/anime_per_period.py \
  --anime-dates-jsonl /path/to/anime_normalized_dates.jsonl \
  --output-dir /path/to/out
"""

from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _parse_date_prefix(d: str) -> str | None:
    if not isinstance(d, str) or not d:
        return None
    # allow YYYY, YYYY-MM, YYYY-MM-DD
    if len(d) >= 10:
        return d[:10]
    if len(d) == 7:
        return d + "-01"
    if len(d) == 4:
        return d + "-01-01"
    return None


def load_anime_dates(anime_dates_jsonl: Path, cutoff: str) -> pd.DataFrame:
    rows = []
    with anime_dates_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            d = _parse_date_prefix(obj.get("air_start_date"))
            rows.append({"air_start_date": d})
    df = pd.DataFrame(rows)
    df["air_start_date"] = pd.to_datetime(df["air_start_date"], errors="coerce")
    df = df[df["air_start_date"].notna()].copy()
    df = df[df["air_start_date"] <= pd.Timestamp(cutoff)].copy()
    df["year"] = df["air_start_date"].dt.year
    return df


def periodize_years(year_series: pd.Series, max_year: int) -> pd.Series:
    years = pd.to_numeric(year_series, errors="coerce")
    bins = [1916, 1960] + list(range(1965, max_year + 2, 5))
    labels = ["1917-1960"]
    for start in range(1961, max_year + 1, 5):
        end = min(start + 4, max_year)
        labels.append(f"{start}-{end}")
    period = pd.cut(years, bins=bins, labels=labels, right=True)
    period = period.astype(str)
    period[years.isna()] = "Unknown"
    return period


def plot_anime_per_period(df: pd.DataFrame, out_file: Path, max_year: int) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")

    period_series = periodize_years(df["year"], max_year=max_year)
    period_labels = ["1917-1960"] + [
        f"{y}-{min(y + 4, max_year)}" for y in range(1961, max_year + 1, 5)
    ]
    period_counts = period_series.value_counts().reindex(period_labels, fill_value=0)

    plt.figure(figsize=(14, 6))
    periods = period_counts.index.tolist()
    counts = period_counts.values
    ax = plt.gca()
    x = np.arange(len(periods))
    ax.plot(x, counts, marker="o", linewidth=3)
    ax.set_xlabel("Period", fontsize=28, fontweight="bold")
    ax.set_ylabel("Number of Anime", fontsize=28, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=26, rotation=45, ha="center")
    plt.yticks(fontsize=24)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot number of anime per period.")
    p.add_argument("--anime-dates-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-name", type=str, default="anime_per_period.pdf")
    p.add_argument("--cutoff-date", type=str, default="2025-06-30")
    p.add_argument("--max-year", type=int, default=2025)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df = load_anime_dates(args.anime_dates_jsonl, cutoff=args.cutoff_date)
    out_file = args.output_dir / args.output_name
    plot_anime_per_period(df, out_file=out_file, max_year=int(args.max_year))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
