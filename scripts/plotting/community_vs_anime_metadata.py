"""
Community (archetype) distributions across anime metadata and gender.

Input:
- Communities CSV with `character_json` and `community`
- Character info JSONL (for animeography and gender)
- Anime info CSV with at least `anime_json`, plus `rating` and `source` columns

Output:
- gender_share_by_prototype_percentage.pdf
- rating_pairgrid_dotplot_by_prototype.pdf
- source_pairgrid_dotplot_by_prototype.pdf

Example:

python scripts/plotting/community_vs_anime_metadata.py \
  --communities-csv /path/to/personality_communities_umap.csv \
  --character-info-jsonl /path/to/character_info_agg.jsonl \
  --anime-info-csv /path/to/anime_info_agg.csv \
  --output-dir /path/to/output_dir
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")


COMMUNITY_LABELS: Dict[int, str] = {
    0: "Energetic Extroverts",
    1: "Steadfast Leaders",
    2: "Cold-hearted Antagonists",
    3: "Passionate Strivers",
    4: "Kind Caregivers",
    5: "Justice Keepers",
    6: "Reserved Introverts",
    7: "Arrogant Tsunderes",
}

LEGEND_LABELS: Dict[int, str] = {
    0: "Energetic\nExtroverts",
    1: "Steadfast\nLeaders",
    2: "Cold-hearted\nAntagonists",
    3: "Passionate\nStrivers",
    4: "Kind\nCaregivers",
    5: "Justice\nKeepers",
    6: "Reserved\nIntroverts",
    7: "Arrogant\nTsunderes",
}

SOURCE_TOP10 = [
    "Original",
    "Manga",
    "Visual novel",
    "Game",
    "Novel",
    "Light novel",
    "Web novel",
    "Web manga",
    "Book",
    "Music",
]


def normalize_rating(rating: Optional[str]) -> str:
    mapping = {
        "G - All Ages": "G (All Ages)",
        "PG - Children": "PG (Children)",
        "PG-13 - Teens 13 or older": "PG-13 (Teens 13 or older)",
        "R - 17+ (violence & profanity)": "R (17+ Violence & Profanity)",
        "Rx - Hentai": "R+/Rx (Mild Nudity and More)",
        "R+ - Mild Nudity": "R+/Rx (Mild Nudity and More)",
        None: "Unknown",
        "None": "Unknown",
    }
    if isinstance(rating, str):
        return mapping.get(rating, rating)
    return "Unknown"


def group_source(src: Optional[str]) -> str:
    if not isinstance(src, str) or src == "":
        return "Other"
    return src if src in SOURCE_TOP10 else "Other"


def derive_character_to_anime(character_info_jsonl: Path) -> pd.DataFrame:
    df_info = pd.read_json(character_info_jsonl, lines=True)
    if "character_json" not in df_info.columns:
        raise ValueError("character info JSONL missing 'character_json' column")

    rows: List[Dict] = []
    for _, row in df_info.iterrows():
        cj = row.get("character_json")
        ao = row.get("animeography")
        if isinstance(ao, list):
            for entry in ao:
                if not isinstance(entry, dict):
                    continue
                anime_json = (
                    entry.get("anime_json")
                    or entry.get("anime")
                    or entry.get("anime_file")
                )
                role = entry.get("role")
                rows.append(
                    {"character_json": cj, "anime_json": anime_json, "role": role}
                )

    map_df = pd.DataFrame(rows)
    if map_df.empty:
        return pd.DataFrame(
            columns=["character_json", "chosen_anime_json", "chosen_role"]
        )

    def choose_one(grp: pd.DataFrame) -> pd.Series:
        grp = grp.copy()
        grp["role_l"] = grp["role"].astype(str).str.lower()
        mains = grp[grp["role_l"] == "main"]
        chosen = mains.iloc[0] if len(mains) > 0 else grp.iloc[0]
        return pd.Series(
            {
                "chosen_anime_json": chosen.get("anime_json"),
                "chosen_role": chosen.get("role"),
            }
        )

    chosen_map = map_df.groupby("character_json", as_index=False).apply(choose_one)
    if "character_json" not in chosen_map.columns:
        chosen_map = chosen_map.reset_index().rename(
            columns={"level_0": "character_json"}
        )
    return chosen_map[["character_json", "chosen_anime_json", "chosen_role"]]


def stacked_percentage_bars_archetypes(
    pivot: pd.DataFrame, out_file: Path, x_label: str, y_label: str
) -> None:
    distinct_colors = sns.color_palette("colorblind", 8)

    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    bottom = pd.Series(0.0, index=pivot.index)
    for community in range(8):
        heights = (
            pivot[community]
            if community in pivot.columns
            else pd.Series(0.0, index=pivot.index)
        )
        ax.bar(
            pivot.index,
            heights,
            bottom=bottom,
            color=distinct_colors[community],
            label=LEGEND_LABELS[community],
            linewidth=0,
        )
        bottom = bottom + heights

    ax.set_xlabel(x_label, fontsize=36, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=36, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("white")
    ax.legend(
        title="Archetype",
        title_fontsize=34,
        fontsize=32,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        ncol=1,
        columnspacing=1.0,
        labelspacing=0.5,
    )
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def pairgrid_dotplot_by_prototype(
    df: pd.DataFrame,
    feature_col: str,
    out_file: Path,
    title: str,
) -> None:
    """
    Dot-plot style: for each archetype, show feature distribution as proportions.

    This mirrors the “pairgrid dotplot” style used in the revised plotting script.
    """
    # Crosstab: community x feature -> proportions
    tab = pd.crosstab(df["community"], df[feature_col], normalize="index") * 100.0
    tab = tab.reindex(index=sorted(COMMUNITY_LABELS.keys()))

    long = (
        tab.reset_index()
        .melt(id_vars=["community"], var_name=feature_col, value_name="percentage")
        .dropna()
    )
    long["archetype"] = long["community"].map(COMMUNITY_LABELS)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    sns.scatterplot(
        data=long,
        x=feature_col,
        y="archetype",
        size="percentage",
        hue="archetype",
        sizes=(10, 800),
        alpha=0.8,
        palette="colorblind",
        legend=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=38, fontweight="bold")
    ax.set_xlabel(feature_col.replace("_", " ").title(), fontsize=36, fontweight="bold")
    ax.set_ylabel("Archetype", fontsize=36, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=28)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_joined_table(
    communities_csv: Path,
    character_info_jsonl: Path,
    anime_info_csv: Path,
) -> pd.DataFrame:
    comm_df = pd.read_csv(communities_csv)
    if "character_json" not in comm_df.columns or "community" not in comm_df.columns:
        raise ValueError(
            "communities CSV must include 'character_json' and 'community'"
        )

    info_df = pd.read_json(character_info_jsonl, lines=True)
    if "character_json" not in info_df.columns:
        raise ValueError("character info JSONL missing 'character_json'")

    anime_df = pd.read_csv(anime_info_csv)
    if "anime_json" not in anime_df.columns:
        raise ValueError("anime info CSV missing 'anime_json'")

    char_to_anime = derive_character_to_anime(character_info_jsonl)
    merged = comm_df.merge(char_to_anime, on="character_json", how="left")

    merged = merged.merge(
        (
            info_df[["character_json", "gender"]].copy()
            if "gender" in info_df.columns
            else info_df[["character_json"]].copy()
        ),
        on="character_json",
        how="left",
    )

    merged = merged.merge(
        anime_df,
        left_on="chosen_anime_json",
        right_on="anime_json",
        how="left",
        suffixes=("", "_anime"),
    )

    if "rating" in merged.columns:
        merged["rating_norm"] = merged["rating"].apply(normalize_rating)
    if "source" in merged.columns:
        merged["source_grouped"] = merged["source"].apply(group_source)

    if "gender" in merged.columns:
        g = merged["gender"].astype(str).str.upper()
        merged["gender_norm"] = g.where(g.isin(["M", "F"]), "Unknown")

    merged["community"] = pd.to_numeric(merged["community"], errors="coerce")
    merged = merged.dropna(subset=["community"]).copy()
    merged["community"] = merged["community"].astype(int)
    merged = merged[merged["community"].isin(set(COMMUNITY_LABELS.keys()))].copy()
    return merged


def make_gender_share_plot(df: pd.DataFrame, out_file: Path) -> None:
    if "gender_norm" not in df.columns:
        raise ValueError(
            "gender column missing (expected 'gender' in character info JSONL)."
        )

    tab = pd.crosstab(df["community"], df["gender_norm"], normalize="index") * 100.0
    tab = tab.reindex(index=sorted(COMMUNITY_LABELS.keys()))
    tab = tab.reindex(columns=["M", "F", "Unknown"], fill_value=0.0)

    # Plot as stacked bars by archetype (x) with gender stacks
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    colors = sns.color_palette("colorblind", len(tab.columns))
    bottom = np.zeros(len(tab.index), dtype=float)
    x = np.arange(len(tab.index))
    labels = [LEGEND_LABELS[i] for i in tab.index]

    for i, col in enumerate(tab.columns):
        vals = tab[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=colors[i], label=str(col), linewidth=0)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Archetype", fontsize=36, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=36, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("white")
    ax.legend(
        title="Gender",
        title_fontsize=34,
        fontsize=32,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot archetype distributions across rating/source and gender."
    )
    p.add_argument("--communities-csv", type=Path, required=True)
    p.add_argument("--character-info-jsonl", type=Path, required=True)
    p.add_argument("--anime-info-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--make-gender", action="store_true", help="Generate gender plot.")
    p.add_argument(
        "--make-rating", action="store_true", help="Generate rating dotplot."
    )
    p.add_argument(
        "--make-source", action="store_true", help="Generate source dotplot."
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    df = build_joined_table(
        communities_csv=args.communities_csv,
        character_info_jsonl=args.character_info_jsonl,
        anime_info_csv=args.anime_info_csv,
    )

    # Default: generate all three if none specified
    if not (args.make_gender or args.make_rating or args.make_source):
        make_gender = make_rating = make_source = True
    else:
        make_gender, make_rating, make_source = (
            args.make_gender,
            args.make_rating,
            args.make_source,
        )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if make_gender:
        out_file = out_dir / "gender_share_by_prototype_percentage.pdf"
        make_gender_share_plot(df, out_file=out_file)
        print(f"Saved: {out_file}")

    if make_rating:
        if "rating_norm" not in df.columns:
            raise ValueError(
                "rating column missing in anime info CSV (expected 'rating')."
            )
        out_file = out_dir / "rating_pairgrid_dotplot_by_prototype.pdf"
        pairgrid_dotplot_by_prototype(
            df=df.dropna(subset=["rating_norm"]).rename(
                columns={"rating_norm": "rating"}
            ),
            feature_col="rating",
            out_file=out_file,
            title="Rating distribution by archetype",
        )
        print(f"Saved: {out_file}")

    if make_source:
        if "source_grouped" not in df.columns:
            raise ValueError(
                "source column missing in anime info CSV (expected 'source')."
            )
        out_file = out_dir / "source_pairgrid_dotplot_by_prototype.pdf"
        pairgrid_dotplot_by_prototype(
            df=df.dropna(subset=["source_grouped"]).rename(
                columns={"source_grouped": "source"}
            ),
            feature_col="source",
            out_file=out_file,
            title="Source distribution by archetype",
        )
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
