"""
Anime rating and source proportions over time.

Input:
- JSONL with at least `air_start_date`, `rating`, `source`

Output:
- pg_rating_proportion_by_year.pdf
- source_proportion_by_year.pdf

Example:

python scripts/plotting/anime_rating_source_trends.py \
  --anime-jsonl /path/to/anime_normalized_dates_with_source.jsonl \
  --output-dir /path/to/out
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RATING_REMAP = {
    "G - All Ages": "G (All Ages)",
    "PG - Children": "PG (Children)",
    "PG-13 - Teens 13 or older": "PG-13 (Teens 13 or older)",
    "R - 17+ (violence & profanity)": "R (17+ Violence & Profanity)",
    "Rx - Hentai": "R+/Rx (Mild Nudity and More)",
    "R+ - Mild Nudity": "R+/Rx (Mild Nudity and More)",
    "None": "Unknown",
}

RATING_ORDER: List[str] = [
    "G (All Ages)",
    "PG (Children)",
    "PG-13 (Teens 13 or older)",
    "R (17+ Violence & Profanity)",
    "R+/Rx (Mild Nudity and More)",
    "Unknown",
]

SOURCE_TOP9: List[str] = [
    "Original",
    "Manga",
    "Visual novel",
    "Game",
    "Novel",
    "Light novel",
    "Web novel",
    "Web manga",
    "Book",
]

RATING_HATCHES: List[str] = ["", "//", "\\\\", "o", "xx", "*"]
SOURCE_HATCHES: List[str] = ["", "//", "\\\\", ".", "xx", "o", "++", "xx", "*", "o-"]


def group_source(src: str | None) -> str:
    if not isinstance(src, str) or src == "":
        return "Other"
    return src if src in SOURCE_TOP9 else "Other"


def load_anime(anime_jsonl: Path, cutoff_date: str) -> pd.DataFrame:
    df = pd.read_json(anime_jsonl, lines=True)
    df["air_start_date"] = pd.to_datetime(
        pd.Series(df["air_start_date"]), errors="coerce"
    )
    df = df[df["air_start_date"].notna()].copy()
    df = df[df["air_start_date"] <= pd.Timestamp(cutoff_date)].copy()
    df["year"] = pd.Series(df["air_start_date"]).dt.year
    df["year_grouped"] = df["year"].apply(
        lambda y: (
            "<1960"
            if pd.notna(y) and y < 1960
            else str(int(y)) if pd.notna(y) else None
        )
    )
    df = df[df["year_grouped"].notna()].copy()
    return df


def plot_rating_proportion(df: pd.DataFrame, out_file: Path) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")

    df = df.copy()
    df["rating"] = pd.Series(df.get("rating")).replace(RATING_REMAP).fillna("Unknown")

    rating_counts = (
        df.groupby(["year_grouped", "rating"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    total_per_year = rating_counts.groupby("year_grouped")["count"].transform("sum")
    rating_counts["proportion"] = rating_counts["count"] / total_per_year

    pivot = rating_counts.pivot(
        index="year_grouped", columns="rating", values="proportion"
    ).fillna(0)
    year_labels = sorted([y for y in pivot.index if y != "<1960"], key=int)
    year_labels = ["<1960"] + year_labels
    pivot = pivot.loc[year_labels]

    palette = sns.color_palette("colorblind", len(RATING_ORDER))
    color_dict = dict(zip(RATING_ORDER, palette))
    hatch_dict = dict(zip(RATING_ORDER, RATING_HATCHES))

    years_int = [int(y) for y in year_labels if y != "<1960"]
    ten_years = list(range(1970, max(years_int) + 1, 10)) if years_int else []
    xtick_labels = ["<1960"] + [str(y) for y in ten_years if str(y) in year_labels]
    xtick_positions = [year_labels.index("<1960")] + [
        year_labels.index(str(y)) for y in ten_years if str(y) in year_labels
    ]

    plt.figure(figsize=(16, 9))
    bottom = None
    for rating in RATING_ORDER:
        if rating not in pivot.columns:
            continue
        plt.bar(
            pivot.index,
            pivot[rating],
            bottom=bottom,
            label=rating,
            color=color_dict[rating],
            hatch=hatch_dict[rating],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = pivot[rating].copy() if bottom is None else (bottom + pivot[rating])

    plt.xlabel("Year", fontsize=34, fontweight="bold")
    plt.ylabel("Proportion", fontsize=34, fontweight="bold")
    plt.legend(
        title="Rating",
        title_fontsize=32,
        fontsize=30,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        handlelength=0.9,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, fontsize=32, ha="center")
    plt.yticks(fontsize=32)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_source_proportion(df: pd.DataFrame, out_file: Path) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")

    df = df.copy()
    df["source_grouped"] = pd.Series(df.get("source")).apply(group_source)

    source_counts = (
        df.groupby(["year_grouped", "source_grouped"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    total_per_year = source_counts.groupby("year_grouped")["count"].transform("sum")
    source_counts["proportion"] = source_counts["count"] / total_per_year

    pivot = source_counts.pivot(
        index="year_grouped", columns="source_grouped", values="proportion"
    ).fillna(0)
    year_labels = sorted([y for y in pivot.index if y != "<1960"], key=int)
    year_labels = ["<1960"] + year_labels
    pivot = pivot.loc[year_labels]

    palette = sns.color_palette("colorblind", len(SOURCE_TOP9)) + [(0.75, 0.75, 0.75)]
    source_order = SOURCE_TOP9 + ["Other"]
    color_dict = dict(zip(source_order, palette))
    hatch_dict = dict(zip(source_order, SOURCE_HATCHES))

    years_int = [int(y) for y in year_labels if y != "<1960"]
    ten_years = list(range(1970, max(years_int) + 1, 10)) if years_int else []
    xtick_labels = ["<1960"] + [str(y) for y in ten_years if str(y) in year_labels]
    xtick_positions = [year_labels.index("<1960")] + [
        year_labels.index(str(y)) for y in ten_years if str(y) in year_labels
    ]

    plt.figure(figsize=(16, 11))
    bottom = None
    for src in source_order:
        if src not in pivot.columns:
            continue
        plt.bar(
            pivot.index,
            pivot[src],
            bottom=bottom,
            label=src,
            color=color_dict[src],
            hatch=hatch_dict[src],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = pivot[src].copy() if bottom is None else (bottom + pivot[src])

    plt.xlabel("Year", fontsize=34, fontweight="bold")
    plt.ylabel("Proportion", fontsize=34, fontweight="bold")
    plt.legend(
        title="Source",
        title_fontsize=32,
        fontsize=30,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        handlelength=0.9,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, fontsize=32, ha="center")
    plt.yticks(fontsize=32)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot rating/source proportions by year.")
    p.add_argument("--anime-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cutoff-date", type=str, default="2025-06-30")
    p.add_argument(
        "--rating-output-name", type=str, default="pg_rating_proportion_by_year.pdf"
    )
    p.add_argument(
        "--source-output-name", type=str, default="source_proportion_by_year.pdf"
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df = load_anime(args.anime_jsonl, cutoff_date=args.cutoff_date)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rating_out = out_dir / args.rating_output_name
    source_out = out_dir / args.source_output_name
    plot_rating_proportion(df, out_file=rating_out)
    plot_source_proportion(df, out_file=source_out)
    print(f"Saved: {rating_out}")
    print(f"Saved: {source_out}")


if __name__ == "__main__":
    main()
