"""
Create a year/period-labeled image dataset CSV from character + anime metadata.

Input:
- Character info JSONL with `character_json`, `character_name`, and `animeography`
- Anime metadata JSONL with `anime_json` and `air_start_date`
- One or more image roots (directories) to search for per-character images

Output:
- CSV with one row per (character, file_type) image that exists, including:
  `character_json`, `character_name`, `file_type`, `file_path`, `year`, `period`,
  `anime_title`, `anime_json`

Example:

python scripts/utils/create_year_labeled_dataset.py \
  --character-info-jsonl /path/to/character_info_agg.jsonl \
  --anime-jsonl /path/to/anime_normalized_dates_with_source.jsonl \
  --image-root contour=/path/to/mal_char_hr_contours \
  --image-root blob=/path/to/mal_char_blobs \
  --output-csv /path/to/character_year_labeled_dataset.csv
"""

from __future__ import annotations
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
from tqdm import tqdm


def parse_date_prefix(d: object) -> Optional[str]:
    if not isinstance(d, str) or not d or d == "unknown":
        return None
    # allow YYYY, YYYY-MM, YYYY-MM-DD
    if len(d) >= 10:
        return d[:10]
    if len(d) == 7:
        return d + "-01"
    if len(d) == 4:
        return d + "-01-01"
    return None


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except Exception:
        try:
            return int(date_str)
        except Exception:
            return None


def periodize_year(year: int, period_length: int) -> str:
    if year == 2025:
        return "2025-2025"
    start = (year // period_length) * period_length
    end = start + period_length - 1
    return f"{start}-{end}"


def char_json_to_image_filename(character_json: str, ext: str) -> str:
    base = character_json[:-5] if character_json.endswith(".json") else character_json
    return f"{base}.{ext}"


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_anime_lookup(anime_jsonl: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for obj in tqdm(load_jsonl(anime_jsonl), desc="Loading anime"):
        anime_json = obj.get("anime_json")
        if not isinstance(anime_json, str) or not anime_json:
            continue
        out[anime_json] = {
            "air_start_date": obj.get("air_start_date"),
            "title": obj.get("japanese_title") or obj.get("title") or "Unknown",
        }
    return out


def pick_first_anime(animeography: object) -> Optional[dict]:
    if not isinstance(animeography, list) or not animeography:
        return None
    first = animeography[0]
    return first if isinstance(first, dict) else None


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def batch_exists(paths: Sequence[Path], max_workers: int) -> Dict[Path, bool]:
    results: Dict[Path, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_path = {ex.submit(file_exists, p): p for p in paths}
        for fut in tqdm(
            as_completed(fut_to_path), total=len(fut_to_path), desc="Checking files"
        ):
            p = fut_to_path[fut]
            try:
                results[p] = bool(fut.result())
            except Exception:
                results[p] = False
    return results


def parse_image_root(spec: str) -> Tuple[str, Path]:
    """
    Parse `file_type=/path/to/images` spec.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected format file_type=/abs/or/rel/path")
    key, val = spec.split("=", 1)
    key = key.strip()
    val = val.strip()
    if not key:
        raise argparse.ArgumentTypeError("Empty file_type in --image-root")
    if not val:
        raise argparse.ArgumentTypeError("Empty path in --image-root")
    return key, Path(val)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create year-labeled dataset CSV.")
    p.add_argument("--character-info-jsonl", type=Path, required=True)
    p.add_argument("--anime-jsonl", type=Path, required=True)
    p.add_argument(
        "--image-root",
        type=parse_image_root,
        action="append",
        required=True,
        help="Repeatable. Format: file_type=/path/to/images",
    )
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--image-ext", type=str, default="jpg")
    p.add_argument("--period-length", type=int, default=3)
    p.add_argument("--cutoff-date", type=str, default="2025-06-30")
    p.add_argument("--max-workers", type=int, default=50)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    anime_lookup = build_anime_lookup(args.anime_jsonl)
    cutoff = datetime.strptime(args.cutoff_date, "%Y-%m-%d")

    image_roots: Dict[str, Path] = {}
    for ft, root in args.image_root:
        image_roots[ft] = root

    candidates: List[dict] = []
    all_paths: List[Path] = []

    for char in tqdm(load_jsonl(args.character_info_jsonl), desc="Scanning characters"):
        character_json = char.get("character_json")
        character_name = char.get("character_name") or char.get("name") or ""
        if not isinstance(character_json, str) or not character_json:
            continue

        first = pick_first_anime(char.get("animeography"))
        if not first:
            continue
        anime_json = first.get("anime_json")
        if not isinstance(anime_json, str) or anime_json not in anime_lookup:
            continue

        anime_meta = anime_lookup[anime_json]
        d0 = parse_date_prefix(anime_meta.get("air_start_date"))
        year = extract_year_from_date(d0)
        if year is None:
            continue

        try:
            if d0 is None:
                continue
            air_date = datetime.strptime(d0, "%Y-%m-%d")
            if air_date > cutoff:
                continue
        except Exception:
            continue

        period = periodize_year(int(year), period_length=int(args.period_length))
        img_name = char_json_to_image_filename(character_json, ext=str(args.image_ext))

        for file_type, root in image_roots.items():
            file_path = root / img_name
            candidates.append(
                {
                    "character_json": character_json,
                    "character_name": character_name,
                    "file_type": file_type,
                    "file_path": str(file_path),
                    "year": int(year),
                    "period": period,
                    "anime_title": anime_meta.get("title", "Unknown"),
                    "anime_json": anime_json,
                }
            )
            all_paths.append(file_path)

    if not candidates:
        raise SystemExit("No candidates found (check inputs and image roots).")

    exists_map = batch_exists(all_paths, max_workers=int(args.max_workers))
    kept = [row for row, p in zip(candidates, all_paths) if exists_map.get(p, False)]

    out_df = pd.DataFrame(kept)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv} (rows={len(out_df):,})")


if __name__ == "__main__":
    main()
