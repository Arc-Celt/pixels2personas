"""
PCA means by period grid.

Input:
- PCA CSV with `character_json` and `pca_1`..`pca_7` (at minimum)
- Character info JSONL providing `period` and `animeography` (for role inference)

Output:
- pca_means_by_period_grid.pdf

Example:

python scripts/plotting/pca_means_by_period_grid.py \
  --pcs-csv /path/to/visual_top100_pcs.csv \
  --character-info-jsonl /path/to/character_info_agg_with_favorites_gender_updated.jsonl \
  --output-dir /path/to/out
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


PERIOD_ORDER: List[str] = [
    "1917-1980",
    "1981-1985",
    "1986-1990",
    "1991-1995",
    "1996-2000",
    "2001-2005",
    "2006-2010",
    "2011-2015",
    "2016-2020",
    "2021-2025",
]


def derive_character_roles(character_info_jsonl: Path) -> pd.DataFrame:
    df_info = pd.read_json(character_info_jsonl, lines=True)
    if "character_json" not in df_info.columns:
        raise ValueError("character info JSONL missing 'character_json'")

    def infer_role(animeography):
        if not isinstance(animeography, list) or len(animeography) == 0:
            return "Unknown"
        roles = []
        for entry in animeography:
            if isinstance(entry, dict):
                role = entry.get("role")
                if isinstance(role, str):
                    roles.append(role.strip())
        roles_lower = [r.lower() for r in roles]
        if any(r == "main" for r in roles_lower):
            return "Main"
        if len(roles_lower) > 0 and all(r == "supporting" for r in roles_lower):
            return "Supporting"
        return "Unknown"

    out = df_info[["character_json"]].copy()
    out["role"] = df_info.get("animeography", []).apply(infer_role)
    return out


def plot_pca_means_by_period_grid(df: pd.DataFrame, out_file: Path) -> None:
    sns.set_theme(style="whitegrid")

    needed = ["period", "role"] + [f"pca_{i}" for i in range(1, 8)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[df["period"].astype(str) != "Unknown"].copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)

    pcs_main = [1, 3, 5, 7]
    pcs_dash = [2, 4, 6]
    roles = ["Main", "Supporting"]

    axis_label_size = 26
    tick_label_size = 18
    legend_title_size = 22
    legend_label_size = 20
    fontweight = "bold"

    color_palette = sns.color_palette("tab10", 7)

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True, sharey=True)
    for row, pc in enumerate(pcs_main):
        for col, role in enumerate(roles):
            ax = axes[row, col]
            df_role = df[df["role"] == role].copy()
            period_stats = df_role.groupby("period")[
                [f"pca_{i}" for i in range(1, 8)]
            ].mean()

            for dpc in pcs_dash:
                means_dash = period_stats[f"pca_{dpc}"].reindex(PERIOD_ORDER)
                ax.plot(
                    PERIOD_ORDER,
                    means_dash.values,
                    label=f"PC{dpc}",
                    color=color_palette[dpc - 1],
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )

            means = period_stats[f"pca_{pc}"].reindex(PERIOD_ORDER)
            ax.plot(
                PERIOD_ORDER,
                means.values,
                marker="o",
                label=f"PC{pc}",
                color=color_palette[pc - 1],
                linewidth=3.5,
                linestyle="-",
            )

            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.7)
            ax.set_facecolor("white")

            if row == 0 and col == 1:
                legend_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[0],
                        linestyle="-",
                        linewidth=3.5,
                        label="PC1",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[1],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="PC2",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[2],
                        linestyle="-",
                        linewidth=3.5,
                        label="PC3",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[3],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="PC4",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[4],
                        linestyle="-",
                        linewidth=3.5,
                        label="PC5",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[5],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="PC6",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color_palette[6],
                        linestyle="-",
                        linewidth=3.5,
                        label="PC7",
                    ),
                ]
                legend_labels = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    fontsize=legend_label_size,
                    title="PC",
                    title_fontsize=legend_title_size,
                    ncol=1,
                    frameon=True,
                )
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_visible(False)

    for col, role in enumerate(roles):
        axes[0, col].set_title(
            role, fontsize=legend_title_size, fontweight=fontweight, pad=20
        )

    fig.supxlabel(
        "Period", fontsize=axis_label_size, fontweight=fontweight, x=0.48, y=0.02
    )
    fig.supylabel(
        "Mean PC Value", fontsize=axis_label_size, fontweight=fontweight, x=0.02, y=0.55
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(h_pad=0.01, w_pad=1.2, rect=[0, 0, 0.99, 1])
    fig.subplots_adjust(hspace=0.01)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot PCA means by period grid (paper figure)."
    )
    p.add_argument("--pcs-csv", type=Path, required=True)
    p.add_argument("--character-info-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-name", type=str, default="pca_means_by_period_grid.pdf")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    pcs_df = pd.read_csv(args.pcs_csv)
    if "character_json" not in pcs_df.columns:
        raise SystemExit("pcs-csv must contain 'character_json'.")

    info_df = pd.read_json(args.character_info_jsonl, lines=True)
    if "character_json" not in info_df.columns or "period" not in info_df.columns:
        raise SystemExit(
            "character-info-jsonl must contain 'character_json' and 'period'."
        )

    roles_df = derive_character_roles(args.character_info_jsonl)
    merged = pcs_df.merge(
        info_df[["character_json", "period"]], on="character_json", how="left"
    )
    merged = merged.merge(roles_df, on="character_json", how="left")

    out_file = args.output_dir / args.output_name
    plot_pca_means_by_period_grid(merged, out_file=out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
