"""
Temporal trends of personality archetypes over time.

Input:
- CSV with at least: `period`, `community`, `role`

Output:
- personality_temporal_trends_grid.pdf

Example:

python scripts/plotting/personality_temporal_trends.py \
  --input-csv /path/to/character_level_table.csv \
  --output-dir /path/to/output_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def plot_temporal_trends_grid(df: pd.DataFrame, out_file: Path) -> None:
    sns.set_theme(style="white")

    df = df.copy()
    df = df[df["period"].astype(str) != "Unknown"].copy()
    df = df[df["community"].isin(set(COMMUNITY_LABELS.keys()))].copy()
    if df.empty:
        raise ValueError(
            "No rows available after filtering Unknown period / invalid communities."
        )

    counts = (
        df.groupby(["period", "community", "role"]).size().reset_index(name="count")
    )
    counts["total"] = counts.groupby(["period", "role"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / counts["total"] * 100.0

    unique_periods = counts["period"].unique().tolist()
    try:
        period_order = sorted(unique_periods, key=lambda p: int(str(p).split("-")[0]))
    except Exception:
        period_order = sorted(unique_periods)

    pivot = counts.pivot_table(
        index="period", columns=["community", "role"], values="proportion"
    ).fillna(0.0)
    pivot = pivot.reindex(index=period_order)

    axis_label_size = 22
    tick_label_size = 16
    legend_title_size = 16
    legend_label_size = 14
    fontweight = "bold"
    role_colors = sns.color_palette("colorblind", 2)

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True, sharey=True)
    for i, community in enumerate(range(8)):
        ax = axes[i // 4, i % 4]

        if (community, "Main") in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[(community, "Main")],
                label="Main",
                color=role_colors[0],
                linewidth=2.5,
                marker="o",
                markersize=6,
                linestyle="-",
            )
        if (community, "Supporting") in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[(community, "Supporting")],
                label="Supporting",
                color=role_colors[1],
                linewidth=2.5,
                marker="o",
                markersize=6,
                linestyle="--",
            )

        ax.set_title(
            COMMUNITY_LABELS[community],
            fontsize=legend_title_size,
            fontweight=fontweight,
        )
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("white")

        if i % 4 == 0:
            ax.set_ylabel("Percentage", fontsize=axis_label_size, fontweight=fontweight)
        else:
            ax.set_ylabel("")

        if i >= 4:
            ax.set_xlabel("Period", fontsize=axis_label_size, fontweight=fontweight)
        else:
            ax.set_xlabel("")

    handles, _ = axes[1, 3].get_legend_handles_labels()
    axes[1, 3].legend(
        handles,
        ["Main", "Supporting"],
        loc="upper right",
        fontsize=legend_label_size,
        title="Role",
        title_fontsize=legend_title_size,
        ncol=1,
        frameon=True,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(h_pad=0.7, w_pad=1.2, rect=[0, 0, 0.97, 1])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot temporal trends grid for personality archetypes."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with period/community/role columns.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--output-name", type=str, default="personality_temporal_trends_grid.pdf"
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df = pd.read_csv(args.input_csv)
    out_file = args.output_dir / args.output_name
    plot_temporal_trends_grid(df, out_file=out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
