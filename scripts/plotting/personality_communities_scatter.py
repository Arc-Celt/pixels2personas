"""
Scatter plot of personality communities (UMAP coordinates).

Input:
- CSV containing `umap_x`, `umap_y`, `community` and optionally `degree_centrality`

Output:
- personality_communities_plot.pdf

Example:

python scripts/plotting/personality_communities_scatter.py \
  --communities-csv /path/to/personality_communities_umap.csv \
  --output-dir /path/to/output_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


COMMUNITY_NAMES = {
    0: "Energetic Extroverts",
    1: "Steadfast Leaders",
    2: "Cold-hearted Antagonists",
    3: "Passionate Strivers",
    4: "Kind Caregivers",
    5: "Justice Keepers",
    6: "Reserved Introverts",
    7: "Arrogant Tsunderes",
}


def plot_community_scatter(
    df: pd.DataFrame,
    core_communities: List[int],
    out_file: Path,
    x_col: str = "umap_x",
    y_col: str = "umap_y",
    palette_name: str = "tab10",
    point_scale: float = 1.0,
) -> None:
    sns.set_theme(style="white")

    df = df.copy()
    if "degree_centrality" not in df.columns:
        df["degree_centrality"] = 1.0

    centrality = (
        pd.to_numeric(df["degree_centrality"], errors="coerce").fillna(0).clip(lower=0)
    )
    ranks = centrality.rank(pct=True).to_numpy().astype(np.float64)
    gamma = 2.5
    smooth = np.power(ranks, gamma)
    min_area, max_area = 0.5, 2.0
    sizes = min_area + (max_area - min_area) * smooth
    sizes = sizes * float(point_scale)
    df["plot_size"] = sizes

    df = df[df["community"].isin(core_communities)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering to core communities.")

    palette = sns.color_palette(palette_name, n_colors=len(core_communities))
    color_dict = {c: palette[i] for i, c in enumerate(core_communities)}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for c in core_communities:
        subset = df[df["community"] == c]
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=subset["plot_size"],
            alpha=0.7,
            color=color_dict[c],
            label=COMMUNITY_NAMES.get(c, f"Community {c}"),
            linewidths=0,
        )

    ax.set_xlabel("UMAP 1", fontsize=22, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=22, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_facecolor("white")

    try:
        x_vals = df[x_col].to_numpy()
        y_vals = df[y_col].to_numpy()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        x_range = max(1e-6, x_max - x_min)
        y_range = max(1e-6, y_max - y_min)
        left_pad = 0.03 * x_range
        right_pad = 0.03 * x_range
        bottom_pad = 0.03 * y_range
        top_pad = 0.03 * y_range
        ax.set_xlim(x_min - left_pad, x_max + right_pad)
        ax.set_ylim(y_min - bottom_pad, y_max + top_pad)
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=COMMUNITY_NAMES.get(c, f"Community {c}"),
            markerfacecolor=color_dict[c],
            markersize=12,
        )
        for c in core_communities
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=15,
        title="Archetype",
        title_fontsize=18,
        loc="upper left",
        frameon=True,
        ncol=1,
        bbox_to_anchor=(1.02, 1.0),
        bbox_transform=ax.transAxes,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot UMAP personality community scatter.")
    p.add_argument("--communities-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--output-name", type=str, default="personality_communities_plot.pdf"
    )
    p.add_argument("--point-scale", type=float, default=5.0)
    p.add_argument("--core-communities", type=int, nargs="+", default=list(range(8)))
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df = pd.read_csv(args.communities_csv)
    out_file = args.output_dir / args.output_name
    plot_community_scatter(
        df=df,
        core_communities=list(args.core_communities),
        out_file=out_file,
        point_scale=float(args.point_scale),
    )
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
