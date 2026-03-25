"""
Character popularity plots.

Input:
- Community CSV with at least `character_json` and `community` columns
- Character info JSONL with at least `character_json` and `favorites` columns

Output:
- log_scaled_popularity_violin_distribution.pdf

Example:

python scripts/plotting/character_popularity_plots.py \
  --communities-csv /path/to/personality_communities_umap.csv \
  --character-info-jsonl /path/to/character_info_agg_with_favorites_gender_updated.jsonl \
  --output-dir /path/to/output_dirs
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    needed = {"character_json", "favorites", "community"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # Remove duplicates, keep highest-favorite record per character.
    df = df.sort_values("favorites", ascending=False).drop_duplicates(
        subset=["character_json"], keep="first"
    )

    # Keep only known communities.
    valid = set(COMMUNITY_LABELS.keys())
    df = df[df["community"].isin(valid)].copy()

    # Keep valid non-negative favorites only.
    df = df[(df["favorites"].notna()) & (df["favorites"] >= 0)].copy()
    df["log_favorites"] = np.log1p(df["favorites"])

    # Two-row x-axis labels.
    two_row_labels = []
    for i in range(8):
        name = COMMUNITY_LABELS[i]
        words = name.split()
        if len(words) >= 2:
            mid = len(words) // 2
            first_row = " ".join(words[:mid])
            second_row = " ".join(words[mid:])
            two_row_labels.append(f"{first_row}\n{second_row}")
        else:
            two_row_labels.append(name)

    axis_label_size = 27
    tick_label_size = 26
    legend_label_size = 26
    fontweight = "bold"

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    positions = list(range(8))
    data_for_plot = []

    for comm in range(8):
        comm_data = df[df["community"] == comm]["log_favorites"]
        if len(comm_data) > 0:
            data_for_plot.append(comm_data.values)
        else:
            data_for_plot.append(np.array([]))

    parts = ax.violinplot(
        data_for_plot,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    violin_colors = sns.color_palette("tab10", 8)
    for i, body in enumerate(parts["bodies"]):
        if len(data_for_plot[i]) > 0:
            body.set_facecolor(violin_colors[i])
            body.set_alpha(0.7)

    parts["cmeans"].set_color("red")
    parts["cmeans"].set_linewidth(2)
    parts["cmeans"].set_label("Mean")

    parts["cmedians"].set_color("blue")
    parts["cmedians"].set_linewidth(2)
    parts["cmedians"].set_linestyle("--")
    parts["cmedians"].set_label("Median")

    for partname in ("cbars", "cmins", "cmaxes"):
        if partname in parts:
            parts[partname].set_color("black")
            parts[partname].set_linewidth(1)

    ax.set_xticks(positions)
    ax.set_xticklabels(two_row_labels, ha="center", rotation=45)
    ax.set_xlabel("Archetype", fontsize=axis_label_size, fontweight=fontweight)
    ax.set_ylabel("Log(Favorites + 1)", fontsize=axis_label_size, fontweight=fontweight)

    y_ticks = ax.get_yticks()
    y_tick_labels = [f"{int(np.expm1(y)):.0f}" if y > 0 else "0" for y in y_ticks]
    ax.set_yticklabels(y_tick_labels)

    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    legend_handles = [
        Line2D([0], [0], color="red", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="blue", linewidth=2, linestyle="--"),
    ]
    ax.legend(
        legend_handles,
        ["Mean", "Median"],
        loc="upper left",
        bbox_to_anchor=(0.925, 0.93),
        bbox_transform=fig.transFigure,
        fontsize=legend_label_size,
        frameon=True,
        handlelength=0.8,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )

    fig.subplots_adjust(left=0.07, right=0.935, bottom=0.18, top=0.95)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper popularity plots.")
    p.add_argument(
        "--communities-csv",
        type=Path,
        required=True,
        help="CSV with character_json and community columns.",
    )
    p.add_argument(
        "--character-info-jsonl",
        type=Path,
        required=True,
        help="JSONL with character_json and favorites columns.",
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
    df = pd.read_csv(args.communities_csv)
    info_df = pd.read_json(args.character_info_jsonl, lines=True)

    required_comm_cols = {"character_json", "community"}
    missing_comm_cols = required_comm_cols - set(df.columns)
    if missing_comm_cols:
        raise SystemExit(
            f"communities-csv is missing required columns: {sorted(missing_comm_cols)}"
        )

    required_info_cols = {"character_json", "favorites"}
    missing_info_cols = required_info_cols - set(info_df.columns)
    if missing_info_cols:
        raise SystemExit(
            "character-info-jsonl is missing required columns: "
            f"{sorted(missing_info_cols)}"
        )

    df = df.merge(
        info_df[["character_json", "favorites"]],
        on="character_json",
        how="left",
    )
    out_file = args.output_dir / args.output_name
    plot_log_scaled_popularity_violin(df, out_file=out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
