"""
Character popularity plots.

Input:
- CSV with at least `favorites` and `community` columns

Output:
- log_scaled_popularity_violin_distribution.pdf

Example:

python scripts/plotting/character_popularity_plots.py \
  --input-csv /path/to/character_table.csv \
  --output-dir /path/to/output_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")


COMMUNITY_LABELS = {
    0: "Energetic Extroverts",
    1: "Steadfast Leaders",
    2: "Cold-hearted Antagonists",
    3: "Passionate Strivers",
    4: "Kind Caregivers",
    5: "Justice Keepers",
    6: "Reserved Introverts",
    7: "Arrogant Tsunderes",
}


def plot_log_scaled_popularity_violin(df: pd.DataFrame, out_file: Path) -> None:
    needed = {"favorites", "community"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["community"] = pd.to_numeric(df["community"], errors="coerce")
    df = df.dropna(subset=["community", "favorites"]).copy()
    df["community"] = df["community"].astype(int)
    df = df[df["community"].isin(set(COMMUNITY_LABELS.keys()))].copy()
    if df.empty:
        raise ValueError("No rows left after filtering to communities 0..7.")

    fav = pd.to_numeric(df["favorites"], errors="coerce").fillna(0.0)
    fav = fav.clip(lower=0.0)
    df["log_favorites"] = np.log1p(fav)

    order = list(range(8))
    df["archetype"] = df["community"].map(COMMUNITY_LABELS)
    archetype_order = [COMMUNITY_LABELS[i] for i in order]

    axis_label_size = 24
    tick_label_size = 20
    title_size = 30
    fontweight = "bold"

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    sns.violinplot(
        data=df,
        x="archetype",
        y="log_favorites",
        order=archetype_order,
        palette=sns.color_palette("colorblind", 8),
        cut=0,
        inner="quartile",
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title(
        "Log-scaled popularity distribution by archetype",
        fontsize=title_size,
        fontweight=fontweight,
    )
    ax.set_xlabel("Archetype", fontsize=axis_label_size, fontweight=fontweight)
    ax.set_ylabel("log(1 + favorites)", fontsize=axis_label_size, fontweight=fontweight)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper popularity plots.")
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with favorites and community.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--output-name",
        type=str,
        default="log_scaled_popularity_violin_distribution.pdf",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df = pd.read_csv(args.input_csv)
    out_file = args.output_dir / args.output_name
    plot_log_scaled_popularity_violin(df, out_file=out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
