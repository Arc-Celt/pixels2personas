"""
Build a mapping from anime JSON files to aggregated anime-level metadata and
their associated characters (with role, year, and period).
- loads per-character metadata (including animeography entries and year/period)
- loads per-anime JSON files with MAL-style metadata
- constructs a mapping `anime_json -> {anime info, characters[…]}`
- writes the result as a CSV with one row per anime and a JSON-encoded
  `characters` column.

Example:

python scripts/utils/anime_character_mapping.py \
  --char-meta-jsonl /path/to/character_info_agg.jsonl \
  --anime-json-dir /path/to/parsed_mal_char_directory \
  --output-csv /path/to/anime_info_agg.csv
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List
import pandas as pd
from tqdm import tqdm


def load_character_metadata(char_meta_path: Path) -> pd.DataFrame:
    """Load character metadata JSONL."""
    return pd.read_json(char_meta_path, lines=True)


def collect_anime_files(anime_json_dir: Path) -> List[str]:
    """Return list of anime JSON filenames in the given directory."""
    return sorted([f for f in os.listdir(anime_json_dir) if f.endswith(".json")])


def build_anime_info_cache(
    anime_json_dir: Path, anime_files: List[str]
) -> Dict[str, Dict]:
    """Load all anime JSON files into a cache keyed by filename."""
    anime_info_cache: Dict[str, Dict] = {}

    def load_file(anime_json_file: str) -> None:
        if anime_json_file in anime_info_cache:
            return
        anime_path = anime_json_dir / anime_json_file
        if not anime_path.exists():
            anime_info_cache[anime_json_file] = {
                "anime_json": anime_json_file,
                "anime_title": None,
                "score": None,
                "ranked": None,
                "popularity": None,
                "members": None,
                "anime_favorites": None,
            }
            return
        try:
            with anime_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            anime_info_cache[anime_json_file] = {
                "anime_json": anime_json_file,
                "anime_title": data.get("title"),
                "score": data.get("score"),
                "ranked": data.get("ranked"),
                "popularity": data.get("popularity"),
                "members": data.get("members"),
                "anime_favorites": data.get("favorites"),
            }
        except Exception as e:
            print(f"[WARN] Failed to load {anime_json_file}: {e}")
            anime_info_cache[anime_json_file] = {
                "anime_json": anime_json_file,
                "anime_title": None,
                "score": None,
                "ranked": None,
                "popularity": None,
                "members": None,
                "anime_favorites": None,
            }

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(
            tqdm(
                executor.map(load_file, anime_files),
                total=len(anime_files),
                desc="Loading anime info",
            )
        )

    return anime_info_cache


def build_anime_character_mapping(
    meta: pd.DataFrame,
    anime_info_cache: Dict[str, Dict],
    anime_files: List[str],
) -> pd.DataFrame:
    """Construct anime-level rows with aggregated character lists."""
    anime_map: Dict[str, Dict] = {}
    for anime_json in anime_files:
        anime_info = anime_info_cache.get(anime_json, {})
        anime_map[anime_json] = {
            "anime_json": anime_json,
            "anime_title": anime_info.get("anime_title"),
            "score": anime_info.get("score"),
            "ranked": anime_info.get("ranked"),
            "popularity": anime_info.get("popularity"),
            "members": anime_info.get("members"),
            "anime_favorites": anime_info.get("anime_favorites"),
            "characters": [],
        }

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Assigning characters"):
        try:
            char_json = row["character_json"]
            year = row.get("year")
            period = row.get("period")
            animeography = row.get("animeography", [])
            for entry in animeography:
                anime_json = entry.get("anime_json")
                role = entry.get("role", "Unknown")
                if not anime_json:
                    continue
                if anime_json not in anime_map:
                    anime_map[anime_json] = {
                        "anime_json": anime_json,
                        "anime_title": entry.get("title"),
                        "score": None,
                        "ranked": None,
                        "popularity": None,
                        "members": None,
                        "anime_favorites": None,
                        "characters": [],
                    }
                anime_map[anime_json]["characters"].append(
                    {
                        "character_json": char_json,
                        "role": role,
                        "year": year,
                        "period": period,
                    }
                )
        except Exception as e:
            print(
                "[WARN] Failed to process row for character_json="
                f"{row.get('character_json', 'UNKNOWN')}: {e}"
            )

    rows = []
    for anime in anime_map.values():
        row = anime.copy()
        row["characters"] = json.dumps(row["characters"], ensure_ascii=False)
        rows.append(row)

    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an anime-level aggregation mapping anime JSON files to their "
            "metadata and associated characters."
        )
    )
    parser.add_argument(
        "--char-meta-jsonl",
        type=Path,
        required=True,
        help="JSONL file with per-character metadata (including animeography).",
    )
    parser.add_argument(
        "--anime-json-dir",
        type=Path,
        required=True,
        help="Directory containing parsed anime JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for aggregated anime information.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    meta = load_character_metadata(args.char_meta_jsonl)
    anime_files = collect_anime_files(args.anime_json_dir)
    anime_info_cache = build_anime_info_cache(args.anime_json_dir, anime_files)

    df = build_anime_character_mapping(meta, anime_info_cache, anime_files)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved mapping to {args.output_csv} with {len(df)} animes.")


if __name__ == "__main__":
    main()
