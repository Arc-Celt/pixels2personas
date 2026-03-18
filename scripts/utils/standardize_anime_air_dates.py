"""
Enrich anime-level metadata with normalized air dates, year, and period bins.

This script:
- loads anime-level aggregation CSV (one row per anime)
- merges in normalized air date information from a JSONL file
- normalizes dates to full YYYY-MM-DD strings, extracts year
- assigns each anime to a time period bin (e.g., 1981–1985, …)
- drops entries with year > a given cutoff (default 2025)
- writes an updated CSV with added `time`, `year`, and `period` columns.

Example:

python scripts/utils/standardize_anime_air_dates.py \
  --anime-info-csv /path/to/anime_info_agg.csv \
  --anime-dates-jsonl /path/to/anime_normalized_dates_with_source.jsonl \
  --output-csv /path/to/anime_info_agg_with_time.csv
"""

import argparse
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


def normalize_date(date_str: str) -> str:
    """Fill missing month/day with '01' to obtain YYYY-MM-DD."""
    if not isinstance(date_str, str) or not date_str or pd.isna(date_str):
        return ""
    parts = date_str.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    if len(parts) == 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return date_str


def get_period(year: float, max_year: int = 2025) -> str:
    """Assign a 5-year period label given a year."""
    if pd.isna(year):
        return "Unknown"
    year_int = int(year)
    if year_int <= 1980:
        return "1917-1980"
    for start in range(1981, max_year + 1, 5):
        end = start + 4
        if start <= year_int <= end:
            return f"{start}-{end}"
    if year_int > max_year:
        return f"{max_year-4}-{max_year}"
    return "Unknown"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize anime air dates and derive year/period features."
    )
    parser.add_argument(
        "--anime-info-csv",
        type=Path,
        required=True,
        help="CSV with aggregated anime information (must contain 'anime_json' and 'anime_title').",
    )
    parser.add_argument(
        "--anime-dates-jsonl",
        type=Path,
        required=True,
        help="JSONL file with normalized air_start_date and japanese_title per anime_json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for anime info with time annotations.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2025,
        help="Maximum year to retain; entries beyond this are dropped (default: 2025).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("Loading anime info CSV…")
    anime_df = pd.read_csv(args.anime_info_csv)

    print("Loading anime dates JSONL…")
    dates_list = []
    with args.anime_dates_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dates_list.append(json.loads(line))
    dates_df = pd.DataFrame(dates_list)

    print("Merging anime info and date info…")
    merged = anime_df.merge(
        dates_df[["anime_json", "japanese_title", "air_start_date"]],
        on="anime_json",
        how="left",
    )

    merged["anime_title"] = merged["anime_title"].combine_first(
        merged["japanese_title"]
    )

    print("Normalizing air_start_date…")
    merged["time"] = merged["air_start_date"].apply(normalize_date)

    print("Extracting year…")
    merged["year"] = merged["time"].apply(
        lambda x: int(x.split("-")[0]) if x else np.nan
    )

    print("Assigning period…")
    merged["period"] = merged["year"].apply(
        lambda y: get_period(y, max_year=args.max_year)
    )

    before = len(merged)
    merged = merged[(merged["year"].isna()) | (merged["year"] <= args.max_year)]
    after = len(merged)
    print(f"Dropped {before - after} entries with year > {args.max_year}")

    merged = merged.drop(columns=["japanese_title", "air_start_date"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Saved updated anime info to {args.output_csv}")


if __name__ == "__main__":
    main()
