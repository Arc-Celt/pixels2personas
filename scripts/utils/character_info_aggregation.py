"""
Aggregate character information into a single JSONL file by merging:
- character biography data (name, animeography)
- personality keywords and gender extracted by an LLM.

Example usage:

python scripts/utils/character_info_aggregation.py \
  --char-bio-dir /path/to/data/anime/char_bio_json_directory \
  --keywords-file /path/to/data/anime/extracted/char_personality_qwen3_32b_fp8.jsonl \
  --output-file /path/to/output/character_info_agg.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable
from tqdm import tqdm


def load_keywords(keywords_file: Path) -> Dict[str, Dict]:
    """Load personality keywords and gender into a lookup map."""
    print(f"Loading personality keywords from {keywords_file}...")
    kw_map: Dict[str, Dict] = {}
    if not keywords_file.exists():
        print(f"[WARN] Keywords file not found: {keywords_file}")
        return {}

    with keywords_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                char_json = data.get("character_json")
                if not char_json:
                    continue

                kw_obj = data.get("personality_keywords") or {}
                if isinstance(kw_obj, dict):
                    english_kw = kw_obj.get("English") or kw_obj.get("english") or []
                else:
                    english_kw = kw_obj if isinstance(kw_obj, list) else []

                kw_map[char_json] = {
                    "keywords": english_kw,
                    "gender": data.get("gender", "Unknown"),
                }
            except json.JSONDecodeError:
                continue
    print(f"Loaded keywords for {len(kw_map)} characters.")
    return kw_map


def aggregate_character_info(
    char_bio_dir: Path,
    keywords_file: Path,
    output_file: Path,
) -> None:
    """Merge biography data with personality keywords and save to JSONL."""
    kw_map = load_keywords(keywords_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    char_files = sorted(char_bio_dir.glob("*.json"))
    print(f"Processing {len(char_files)} character files from {char_bio_dir}...")

    count = 0
    with output_file.open("w", encoding="utf-8") as out_f:
        for char_file in tqdm(char_files, desc="Aggregating characters"):
            try:
                with char_file.open("r", encoding="utf-8") as f:
                    char_data = json.load(f)

                char_json = char_file.name
                kw_data = kw_map.get(char_json, {})
                gender = kw_data.get("gender")
                if not gender or gender == "Unknown":
                    gender = char_data.get("gender", "Unknown")

                # Standardize gender format (M/F/Unknown)
                if gender in ("Male", "M"):
                    gender = "M"
                elif gender in ("Female", "F"):
                    gender = "F"
                else:
                    gender = "Unknown"

                entry = {
                    "character_json": char_json,
                    "character_name": char_data.get("name", ""),
                    "personality_keywords": kw_data.get("keywords", []),
                    "gender": gender,
                    "animeography": char_data.get("animeography", []),
                }

                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

            except Exception as e:
                print(f"[WARN] Failed to process {char_file.name}: {e}")

    print(f"Aggregated {count} characters into {output_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for character info aggregation."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate character biographies with personality keywords into a JSONL file."
        )
    )
    parser.add_argument(
        "--char-bio-dir",
        type=Path,
        required=True,
        help="Directory containing character biography JSON files.",
    )
    parser.add_argument(
        "--keywords-file",
        type=Path,
        required=True,
        help="JSONL file with LLM-extracted personality keywords and gender.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL file path for aggregated character information.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    aggregate_character_info(
        char_bio_dir=args.char_bio_dir,
        keywords_file=args.keywords_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
