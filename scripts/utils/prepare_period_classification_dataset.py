"""
Prepare a period classification dataset for temporal prediction of anime characters.

This script:
- loads per-character metadata with `period`, `year`, and `character_json`
- filters to characters with known periods
- finds corresponding avatar images in a source directory
- creates stratified train/val/test splits over periods
- organizes images into a directory tree by split and period
- writes `labels.csv` and `metadata.json` describing the dataset.

Example:

python scripts/utils/prepare_period_classification_dataset.py \
  --character-info-jsonl /path/to/character_info_agg.jsonl \
  --avatar-dir /path/to/mal_char_ind_hr_images_directory \
  --output-dir /path/to/classification_period_directory
"""

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PERIODS: List[Tuple[int, int, str]] = [
    (1917, 1980, "1917_1980"),
    (1981, 1985, "1981_1985"),
    (1986, 1990, "1986_1990"),
    (1991, 1995, "1991_1995"),
    (1996, 2000, "1996_2000"),
    (2001, 2005, "2001_2005"),
    (2006, 2010, "2006_2010"),
    (2011, 2015, "2011_2015"),
    (2016, 2020, "2016_2020"),
    (2021, 2025, "2021_2025"),
]


def load_and_filter_characters(character_info_jsonl: Path) -> pd.DataFrame:
    """Load character info JSONL and retain rows with known periods."""
    print("Loading character info…")
    data = []
    with character_info_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} characters")

    df = df.drop_duplicates(subset=["character_json"], keep="first")
    print(f"After deduplication: {len(df):,} characters")

    df["period"] = df["period"].astype(str).str.replace("-", "_")
    valid_df = df[df["period"].notna() & (df["period"].str.lower() != "unknown")].copy()
    print(f"Characters with valid periods: {len(valid_df):,}")

    period_counts = valid_df["period"].value_counts().sort_index()
    print("Period distribution (all characters):")
    for period, count in period_counts.items():
        start_year, end_year = period.split("_")
        print(f"  {start_year}-{end_year}: {count:,}")

    return valid_df


def find_character_images(df: pd.DataFrame, avatar_dir: Path) -> List[Dict]:
    """Find available images for characters based on `character_json` filenames."""
    print(f"Checking images for {len(df):,} characters…")

    if not avatar_dir.exists():
        raise FileNotFoundError(f"Avatar directory not found: {avatar_dir}")

    needed_chars = {
        str(row["character_json"]).replace(".json", "") for _, row in df.iterrows()
    }
    print(f"Need images for {len(needed_chars):,} unique characters")

    image_lookup: Dict[str, Path] = {}
    jpg_files = list(avatar_dir.glob("*.jpg"))
    print(f"Scanning {len(jpg_files):,} jpg files…")

    for file_path in jpg_files:
        base_name = file_path.stem
        if base_name in needed_chars:
            image_lookup[base_name] = file_path

    print(f"Found {len(image_lookup):,} matching images")

    available_chars: List[Dict] = []
    missing_count = 0
    for _, row in df.iterrows():
        char_json = row["character_json"]
        base_name = str(char_json).replace(".json", "")
        if base_name in image_lookup:
            image_path = image_lookup[base_name]
            char_dict = row.to_dict()
            char_dict["image_path"] = str(image_path)
            char_dict["image_filename"] = image_path.name
            available_chars.append(char_dict)
        else:
            missing_count += 1

    print(f"Characters with images: {len(available_chars):,}")
    print(f"Characters without images: {missing_count:,}")

    if available_chars:
        coverage_rate = len(available_chars) / len(df) * 100.0
        print(f"Image coverage rate: {coverage_rate:.1f}%")

    return available_chars


def copy_single_image(
    char_data: Dict,
    target_base_dir: Path,
    split_name: str,
    use_symlinks: bool,
    skip_existing: bool,
) -> str:
    source_path = Path(char_data["image_path"])
    period = char_data["period"]
    target_dir = target_base_dir / split_name / period
    target_path = target_dir / source_path.name

    try:
        if skip_existing and target_path.exists():
            return "skipped"

        target_dir.mkdir(parents=True, exist_ok=True)

        if use_symlinks:
            if target_path.exists():
                target_path.unlink()
            target_path.symlink_to(source_path.absolute())
        else:
            shutil.copy2(source_path, target_path)

        return "copied"
    except Exception as e:
        print(f"Error copying {source_path} -> {target_path}: {e}")
        return "error"


def copy_images_to_directory(
    character_list: List[Dict],
    target_base_dir: Path,
    split_name: str,
    use_symlinks: bool,
    skip_existing: bool,
    max_workers: int,
) -> Tuple[int, int, int]:
    total = len(character_list)
    print(f"Processing {total:,} images for split='{split_name}'…")

    period_groups: Dict[str, List[Dict]] = {}
    for char in character_list:
        period = char["period"]
        period_groups.setdefault(period, []).append(char)

    for period in period_groups.keys():
        (target_base_dir / split_name / period).mkdir(parents=True, exist_ok=True)

    max_workers = max(1, max_workers)
    chunk_size = max(200, total // max_workers) if total else 0

    results: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, total, chunk_size or total):
            chunk = character_list[i : i + (chunk_size or total)]
            args_list = [
                (char, target_base_dir, split_name, use_symlinks, skip_existing)
                for char in chunk
            ]
            futures.append(
                executor.submit(
                    lambda batch: [copy_single_image(*args) for args in batch],
                    args_list,
                )
            )

        for future in futures:
            results.extend(future.result())

    copied = results.count("copied")
    skipped = results.count("skipped")
    errors = results.count("error")
    print(f"{split_name}: {copied:,} copied, {skipped:,} skipped, {errors} errors")
    return copied, skipped, errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a period classification dataset (train/val/test splits)."
    )
    parser.add_argument(
        "--character-info-jsonl",
        type=Path,
        required=True,
        help="JSONL file with per-character metadata including 'period' and 'character_json'.",
    )
    parser.add_argument(
        "--avatar-dir",
        type=Path,
        required=True,
        help="Directory with high-resolution character avatar images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the period classification dataset.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.5,
        help="Fraction of data reserved for test split (default: 0.5).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of remaining train+val reserved for val (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stratified splits (default: 42).",
    )
    parser.add_argument(
        "--skip-existing-images",
        action="store_true",
        help="Skip copying images that already exist in the target location.",
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        help="Use symlinks instead of copying image files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of worker threads for copying (default: 8).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    df = load_and_filter_characters(args.character_info_jsonl)
    available_chars = find_character_images(df, args.avatar_dir)

    if not available_chars:
        print("No characters with images found; nothing to do.")
        return

    chars_df = pd.DataFrame(available_chars)
    period_counts = chars_df["period"].value_counts().sort_index()
    print("Final period distribution (with images):")
    for period, count in period_counts.items():
        start_year, end_year = period.split("_")
        print(f"  {start_year}-{end_year}: {count:,} characters")

    train_val_chars, test_chars = train_test_split(
        available_chars,
        test_size=args.test_split,
        random_state=args.random_state,
        stratify=[char["period"] for char in available_chars],
    )

    train_chars, val_chars = train_test_split(
        train_val_chars,
        test_size=args.val_split,
        random_state=args.random_state + 1,
        stratify=[char["period"] for char in train_val_chars],
    )

    print("Split sizes:")
    total = len(available_chars)
    print(f"  Train: {len(train_chars):,} ({len(train_chars) / total * 100:.1f}%)")
    print(f"  Val: {len(val_chars):,} ({len(val_chars) / total * 100:.1f}%)")
    print(f"  Test: {len(test_chars):,} ({len(test_chars) / total * 100:.1f}%)")

    output_base_dir = args.output_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)

    train_copied, train_skipped, train_errors = copy_images_to_directory(
        train_chars,
        output_base_dir,
        "train",
        use_symlinks=args.use_symlinks,
        skip_existing=args.skip_existing_images,
        max_workers=args.max_workers,
    )
    val_copied, val_skipped, val_errors = copy_images_to_directory(
        val_chars,
        output_base_dir,
        "val",
        use_symlinks=args.use_symlinks,
        skip_existing=args.skip_existing_images,
        max_workers=args.max_workers,
    )
    test_copied, test_skipped, test_errors = copy_images_to_directory(
        test_chars,
        output_base_dir,
        "test",
        use_symlinks=args.use_symlinks,
        skip_existing=args.skip_existing_images,
        max_workers=args.max_workers,
    )

    all_chars = train_chars + val_chars + test_chars
    labels_data = []
    for char in all_chars:
        split = (
            "train" if char in train_chars else "val" if char in val_chars else "test"
        )
        labels_data.append(
            {
                "character_json": char["character_json"],
                "character_name": char.get("character_name", ""),
                "image_filename": char["image_filename"],
                "period": char["period"],
                "year": char.get("year"),
                "split": split,
            }
        )

    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(output_base_dir / "labels.csv", index=False)

    train_df = pd.DataFrame(train_chars)
    val_df = pd.DataFrame(val_chars)
    test_df = pd.DataFrame(test_chars)

    train_counts = train_df["period"].value_counts().sort_index()
    val_counts = val_df["period"].value_counts().sort_index()
    test_counts = test_df["period"].value_counts().sort_index()

    metadata = {
        "task": "period_classification",
        "description": "10-class period classification based on anime release years",
        "num_classes": len(PERIODS),
        "periods": {label: f"{start}-{end}" for start, end, label in PERIODS},
        "dataset_size": {
            "total_samples": len(all_chars),
            "train_samples": len(train_chars),
            "val_samples": len(val_chars),
            "test_samples": len(test_chars),
        },
        "class_distribution": {
            "train": train_counts.to_dict(),
            "val": val_counts.to_dict(),
            "test": test_counts.to_dict(),
        },
        "images": {
            "train_copied": train_copied,
            "val_copied": val_copied,
            "test_copied": test_copied,
            "total_copied": train_copied + val_copied + test_copied,
            "copy_errors": train_errors + val_errors + test_errors,
        },
    }

    with (output_base_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("Period classification dataset created.")
    print(f"Output directory: {output_base_dir}")
    print(f"Total samples: {len(all_chars):,}")
    print(f"Classes: {len(PERIODS)} periods")


if __name__ == "__main__":
    main()
