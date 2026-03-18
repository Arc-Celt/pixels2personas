"""
Average characters per anime by role over time.

Input:
- Anime dates JSONL with `anime_json` and `air_start_date`
- Character info JSONL with `character_json` and `animeography` entries containing `anime_json` and `role`

Output:
- time_trend_avg_characters_per_anime_by_role.pdf

Example:

python scripts/plotting/avg_characters_per_anime_by_role.py \
  --anime-dates-jsonl /path/to/anime_normalized_dates.jsonl \
  --character-info-jsonl /path/to/character_info_agg.jsonl \
  --output-dir /path/to/out
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, Set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _parse_date_prefix(d: str) -> str | None:
    if not isinstance(d, str) or not d:
        return None
    if len(d) >= 10:
        return d[:10]
    if len(d) == 7:
        return d + "-01"
    if len(d) == 4:
        return d + "-01-01"
    return None


def load_valid_anime(
    anime_dates_jsonl: Path, cutoff: str
) -> tuple[Set[str], dict[str, int]]:
    valid: Set[str] = set()
    year_by_anime: dict[str, int] = {}

    with anime_dates_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            anime_json = obj.get("anime_json")
            d = _parse_date_prefix(obj.get("air_start_date"))
            if not anime_json or not d:
                continue
            ts = pd.to_datetime(d, errors="coerce")
            if pd.isna(ts) or ts > pd.Timestamp(cutoff):
                continue
            valid.add(anime_json)
            year_by_anime[anime_json] = int(ts.year)

    return valid, year_by_anime


def load_character_animeography(
    character_info_jsonl: Path, valid_anime: Set[str]
) -> pd.DataFrame:
    rows = []
    with character_info_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cj = obj.get("character_json")
            ao = obj.get("animeography", [])
            if not cj or not isinstance(ao, list):
                continue
            for entry in ao:
                if not isinstance(entry, dict):
                    continue
                anime_json = entry.get("anime_json")
                role = entry.get("role")
                if anime_json in valid_anime:
                    rows.append(
                        {"anime_json": anime_json, "role": role, "character_json": cj}
                    )
    return pd.DataFrame(rows)


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


def plot_time_trend(df: pd.DataFrame, out_file: Path, cutoff_year: int) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")

    counts = (
        df.groupby(["anime_json", "role"])["character_json"]
        .nunique()
        .reset_index(name="n_characters")
    )
    pivot = counts.pivot(
        index="anime_json", columns="role", values="n_characters"
    ).fillna(0)
    for col in ["Main", "Supporting"]:
        if col not in pivot.columns:
            pivot[col] = 0

    if "year" not in df.columns:
        raise ValueError(
            "year column missing; did you pass anime dates with air_start_date?"
        )
    year_map = (
        df.drop_duplicates("anime_json").set_index("anime_json")["year"].to_dict()
    )
    pivot = pivot.reset_index()
    pivot["year"] = pivot["anime_json"].map(year_map)
    pivot = pivot[pivot["year"].notna()].copy()
    pivot["year"] = pivot["year"].astype(int)
    pivot = pivot[pivot["year"] <= cutoff_year].copy()

    pivot["period"] = periodize_years(pivot["year"], max_year=cutoff_year)
    period_labels = ["1917-1960"] + [
        f"{y}-{min(y + 4, cutoff_year)}" for y in range(1961, cutoff_year + 1, 5)
    ]
    trend = (
        pivot.groupby("period")[["Main", "Supporting"]]
        .mean()
        .reindex(period_labels)
        .reset_index()
    )

    plt.figure(figsize=(14, 6))
    x = np.arange(len(trend["period"]))
    for col in ["Main", "Supporting"]:
        plt.plot(x, trend[col], label=col, marker="o", linewidth=3)
    plt.xlabel("Period", fontsize=24, fontweight="bold")
    plt.ylabel("Mean Character Count", fontsize=24, fontweight="bold")
    plt.xticks(x, trend["period"], fontsize=22, rotation=45, ha="center")
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot avg characters per anime by role over time."
    )
    p.add_argument("--anime-dates-jsonl", type=Path, required=True)
    p.add_argument("--character-info-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--output-name",
        type=str,
        default="time_trend_avg_characters_per_anime_by_role.pdf",
    )
    p.add_argument("--cutoff-date", type=str, default="2025-06-30")
    p.add_argument("--cutoff-year", type=int, default=2025)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    valid_anime, year_by_anime = load_valid_anime(
        args.anime_dates_jsonl, cutoff=args.cutoff_date
    )
    char_df = load_character_animeography(
        args.character_info_jsonl, valid_anime=valid_anime
    )
    if char_df.empty:
        raise SystemExit("No character-animeography rows found for the given cutoff.")
    char_df["year"] = char_df["anime_json"].map(year_by_anime)
    out_file = args.output_dir / args.output_name
    plot_time_trend(char_df, out_file=out_file, cutoff_year=int(args.cutoff_year))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
