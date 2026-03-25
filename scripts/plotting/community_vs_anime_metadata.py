"""
Community distribution across anime metadata (rating/source) and character gender.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

AXIS_LABEL_SIZE = 36
TICK_LABEL_SIZE = 32
LEGEND_TITLE_SIZE = 34
LEGEND_LABEL_SIZE = 32
FONTWEIGHT = "bold"

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

SOURCE_TOP10: List[str] = [
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
    rows: List[Dict] = []
    for _, row in df_info.iterrows():
        cj = row.get("character_json")
        ao = row.get("animeography")
        if not isinstance(ao, list):
            continue
        for entry in ao:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "character_json": cj,
                    "anime_json": entry.get("anime_json")
                    or entry.get("anime")
                    or entry.get("anime_file"),
                    "role": entry.get("role"),
                }
            )
    map_df = pd.DataFrame(rows)
    if map_df.empty:
        return pd.DataFrame(columns=["character_json", "chosen_anime_json"])
    map_df["is_main"] = map_df["role"].astype(str).str.lower().eq("main")
    map_df = map_df.sort_values(["character_json", "is_main"], ascending=[True, False])
    chosen = map_df.drop_duplicates("character_json", keep="first")
    return chosen.rename(columns={"anime_json": "chosen_anime_json"})[
        ["character_json", "chosen_anime_json"]
    ]


def stacked_percentage_bars_generic(pivot: pd.DataFrame, out_file: Path) -> None:
    colors = sns.color_palette("colorblind", len(pivot.columns))
    fig, ax = plt.subplots(figsize=(17, 9), dpi=300)
    bottom = pd.Series(0.0, index=pivot.index)
    for i, col in enumerate(pivot.columns):
        vals = pivot[col]
        ax.bar(pivot.index, vals, bottom=bottom, color=colors[i], label=str(col), linewidth=0)
        bottom = bottom + vals
    ax.set_xlabel("Archetype", fontsize=AXIS_LABEL_SIZE, fontweight=FONTWEIGHT)
    ax.set_ylabel("Percentage", fontsize=AXIS_LABEL_SIZE, fontweight=FONTWEIGHT)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", rotation=45, labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, ls="--", color="black", lw=2, alpha=0.7)
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 1.01, 50, "50%", fontsize=LEGEND_TITLE_SIZE, ha="left", va="center")
    ax.legend(
        title="Gender",
        title_fontsize=LEGEND_TITLE_SIZE,
        fontsize=LEGEND_LABEL_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.99, 1.0),
        frameon=True,
        handlelength=0.9,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def pairgrid_dotplot(
    data: pd.DataFrame, group_col: str, group_order: List[str], out_file: Path, n_rows: int, n_cols: int, bottom_margin: float
) -> None:
    proto_order = [lbl.replace("\n", " ") for lbl in LEGEND_LABELS.values()][::-1]
    plot_data = data.copy()
    plot_data["Prototype"] = plot_data["community_label"].str.replace("\n", " ")
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6.5, n_rows * 6), sharey=True)
    axes = axes.flatten()
    for i, group in enumerate(group_order):
        ax = axes[i]
        facet = plot_data[plot_data[group_col] == group]
        y_data = []
        for p in proto_order:
            vals = facet.loc[facet["Prototype"] == p, "prop"]
            y_data.append(float(vals.iloc[0]) if not vals.empty else 0.0)
        ax.scatter(y_data, proto_order, color=sns.color_palette("colorblind", 1)[0], s=180, edgecolor="w", linewidth=1)
        ax.set_xlim(0, 100)
        if i % n_cols == 0:
            ax.set_ylabel("Archetype", fontsize=AXIS_LABEL_SIZE, fontweight=FONTWEIGHT)
        else:
            ax.set_ylabel("")
        ax.set_title(group.replace("\n", " "), fontsize=LEGEND_TITLE_SIZE, fontweight=FONTWEIGHT)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_yticks(proto_order)
    for j in range(len(group_order), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(bottom=bottom_margin)
    fig.supxlabel("Percentage", fontsize=AXIS_LABEL_SIZE, fontweight=FONTWEIGHT)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def build_inputs(communities_csv: Path, character_info_jsonl: Path, anime_meta_jsonl: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_char = pd.read_csv(communities_csv)
    df_char = df_char[df_char["community"].isin(set(range(8)))].copy()
    df_info = pd.read_json(character_info_jsonl, lines=True)
    if "gender" not in df_char.columns and "gender" in df_info.columns:
        df_char = df_char.merge(df_info[["character_json", "gender"]], on="character_json", how="left")
    df_char["gender_norm"] = df_char.get("gender", pd.Series(index=df_char.index)).astype(str).str.upper().map({"M": "Male", "F": "Female"}).fillna("Unknown")
    char2anime = derive_character_to_anime(character_info_jsonl)
    df_anime = pd.read_json(anime_meta_jsonl, lines=True)
    df_anime["rating_norm"] = df_anime["rating"].apply(normalize_rating)
    df_anime["source_grouped"] = df_anime["source"].apply(group_source)
    df_join = df_char.merge(char2anime, on="character_json", how="left")
    df_join = df_join.merge(df_anime[["anime_json", "rating_norm", "source_grouped"]], left_on="chosen_anime_json", right_on="anime_json", how="left")
    return df_char, df_join


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot community vs anime metadata.")
    p.add_argument("--communities-csv", type=Path, required=True)
    p.add_argument("--character-info-jsonl", type=Path, required=True)
    p.add_argument("--anime-meta-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--make-gender", action="store_true")
    p.add_argument("--make-rating", action="store_true")
    p.add_argument("--make-source", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    df_char, df_join = build_inputs(args.communities_csv, args.character_info_jsonl, args.anime_meta_jsonl)
    do_gender = args.make_gender
    do_rating = args.make_rating
    do_source = args.make_source
    if not (do_gender or do_rating or do_source):
        do_gender = do_rating = do_source = True

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if do_gender:
        df_g = df_char[df_char["gender_norm"].isin(["Male", "Female"])].copy()
        g = df_g.groupby(["community", "gender_norm"]).size().reset_index(name="count")
        g["total"] = g.groupby("community")["count"].transform("sum")
        g["prop"] = g["count"] / g["total"] * 100.0
        pivot = g.pivot(index="community", columns="gender_norm", values="prop").fillna(0.0)
        pivot = pivot.reindex(index=list(range(8)))
        pivot.index = [LEGEND_LABELS[i] for i in pivot.index]
        p = out / "gender_share_by_prototype_percentage.pdf"
        stacked_percentage_bars_generic(pivot, p)
        print(f"Saved: {p}")

    if do_rating:
        def compact_rating(name: str) -> str:
            if name.startswith("G "):
                return "G"
            if name.startswith("PG-13"):
                return "PG-13"
            if name.startswith("PG "):
                return "PG"
            if name.startswith("R+/Rx"):
                return "R+/Rx"
            if name.startswith("R "):
                return "R"
            return name
        order_raw = ["G (All Ages)", "PG (Children)", "PG-13 (Teens 13 or older)", "R (17+ Violence & Profanity)", "R+/Rx (Mild Nudity and More)"]
        order = [compact_rating(x) for x in order_raw]
        d = df_join[df_join["rating_norm"].notna() & (df_join["rating_norm"] != "Unknown")].copy()
        d["rating_compact"] = d["rating_norm"].map(compact_rating)
        c = d.groupby(["rating_compact", "community"]).size().reset_index(name="count")
        c["total"] = c.groupby("rating_compact")["count"].transform("sum")
        c["prop"] = c["count"] / c["total"] * 100.0
        c["community_label"] = c["community"].map(lambda x: LEGEND_LABELS[x].replace("\n", " "))
        c = c[c["rating_compact"].isin(order)]
        p = out / "rating_pairgrid_dotplot_by_prototype.pdf"
        pairgrid_dotplot(c, "rating_compact", order, p, n_rows=3, n_cols=2, bottom_margin=0.1)
        print(f"Saved: {p}")

    if do_source:
        def format_source(lbl: str) -> str:
            t = str(lbl).title()
            return t.replace(" ", "\n", 1) if " " in t else t
        order = [format_source(s) for s in SOURCE_TOP10 if s != "Music"] + ["Other"]
        d = df_join[df_join["source_grouped"].notna()].copy()
        d["source_compact"] = d["source_grouped"].map(format_source).replace({"Music": "Other"})
        c = d.groupby(["source_compact", "community"]).size().reset_index(name="count")
        c["total"] = c.groupby("source_compact")["count"].transform("sum")
        c["prop"] = c["count"] / c["total"] * 100.0
        c["community_label"] = c["community"].map(lambda x: LEGEND_LABELS[x].replace("\n", " "))
        c = c[c["source_compact"].isin(order)]
        p = out / "source_pairgrid_dotplot_by_prototype.pdf"
        pairgrid_dotplot(c, "source_compact", order, p, n_rows=5, n_cols=2, bottom_margin=0.05)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()

