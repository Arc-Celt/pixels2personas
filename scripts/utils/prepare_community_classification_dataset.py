"""
Prepare an 8-class community classification dataset from personality communities.

This script:
- loads per-character community assignments and names from a CSV
- filters to the 8 target communities
- finds corresponding avatar images in a source directory
- creates stratified train/val/test splits over communities
- organizes images into a directory tree by split and community
- writes `labels.csv` and `metadata.json` describing the dataset.

Example:

python scripts/utils/prepare_community_classification_dataset.py \
  --communities-csv /path/to/personality_communities_umap.csv \
  --avatar-dir /path/to/mal_char_ind_hr_images \
  --output-dir /path/to/classification_community
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


COMMUNITY_LABELS: Dict[int, str] = {
    0: "Energetic Extroverts",
    1: "Steadfast Leaders",
    2: "Cold-hearted Villains",
    3: "Passionate Strivers",
    4: "Kind Caregivers",
    5: "Justice Keepers",
    6: "Reserved Introverts",
    7: "Arrogant Tsunderes",
}


def load_and_process_data(communities_csv: Path) -> pd.DataFrame:
    """Load community CSV and restrict to desired communities."""
    print("Loading community CSV…")
    if not communities_csv.exists():
        raise FileNotFoundError(f"Communities CSV not found: {communities_csv}")

    df = pd.read_csv(communities_csv)
    print(f"Loaded {len(df):,} rows from community CSV")

    required_cols = ["character_json", "character_name", "community"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in communities CSV: {missing}")

    df_dedup = df.drop_duplicates(subset=["character_json"], keep="first")
    print(f"After deduplication: {len(df_dedup):,} unique characters")

    allowed = set(COMMUNITY_LABELS.keys())
    valid_df = df_dedup[df_dedup["community"].isin(allowed)].copy()
    print(f"Characters in target communities {sorted(allowed)}: {len(valid_df):,}")

    comm_counts = valid_df["community"].value_counts().sort_index()
    print("Community distribution (all with embeddings):")
    for cid, count in comm_counts.items():
        label = COMMUNITY_LABELS.get(int(cid), f"Community_{cid}")
        print(f"  {cid}: {label} -> {count:,} characters")

    return valid_df


def find_character_images(df: pd.DataFrame, avatar_dir: Path) -> List[Dict]:
    """Find available images for characters with optimized file lookup."""
    print(f"Checking images for {len(df):,} characters…")

    if not avatar_dir.exists():
        raise FileNotFoundError(f"Avatar directory not found: {avatar_dir}")

    needed_chars = {
        str(row["character_json"]).replace(".json", "") for _, row in df.iterrows()
    }
    print(f"Need to find {len(needed_chars):,} unique characters")

    image_lookup: Dict[str, Path] = {}
    found_count = 0

    jpg_files = list(avatar_dir.glob("*.jpg"))
    print(f"Scanning {len(jpg_files):,} jpg files…")

    for file_path in jpg_files:
        base_name = file_path.stem
        if base_name in needed_chars:
            image_lookup[base_name] = file_path
            found_count += 1
            if found_count >= len(needed_chars):
                break

    print(f"Found {found_count:,} matching images")

    available_chars: List[Dict] = []
    missing_count = 0
    for _, char_data in df.iterrows():
        char_json = char_data["character_json"]
        base_name = str(char_json).replace(".json", "")
        if base_name in image_lookup:
            image_path = image_lookup[base_name]
            char_dict = char_data.to_dict()
            char_dict["image_path"] = str(image_path)
            char_dict["image_filename"] = image_path.name
            available_chars.append(char_dict)
        else:
            missing_count += 1

    print(f"Characters with images: {len(available_chars):,}")
    print(f"Characters without images: {missing_count:,}")

    total_chars = len(df)
    coverage_rate = len(available_chars) / total_chars * 100 if total_chars else 0
    print(f"Image coverage rate: {coverage_rate:.1f}%")

    return available_chars


def copy_single_image(
    char_data: Dict,
    target_base_dir: Path,
    split_name: str,
    skip_existing: bool,
    use_symlinks: bool,
) -> str:
    source_path = Path(char_data["image_path"])
    cid = int(char_data["community"])
    label = COMMUNITY_LABELS.get(cid, f"Community_{cid}")
    folder_name = f"{cid}_{label.replace(' ', '_')}"
    target_dir = target_base_dir / split_name / folder_name
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
    skip_existing: bool,
    use_symlinks: bool,
    max_workers: int,
) -> Tuple[int, int, int]:
    total = len(character_list)
    print(f"Processing {total:,} images for split='{split_name}'…")

    if total == 0:
        print(f"No characters for split '{split_name}'")
        return 0, 0, 0

    max_workers = max(1, min(max_workers, 12))
    chunk_size = max(200, total // max_workers)

    results: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, total, chunk_size):
            chunk = character_list[i : i + chunk_size]
            args_list = [
                (char, target_base_dir, split_name, skip_existing, use_symlinks)
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
        description="Prepare an 8-class personality community classification dataset."
    )
    parser.add_argument(
        "--communities-csv",
        type=Path,
        required=True,
        help="CSV with personality community assignments (must include 'character_json', 'character_name', 'community').",
    )
    parser.add_argument(
        "--avatar-dir",
        type=Path,
        required=True,
        help="Directory with character avatar images (.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the community classification dataset.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test split (default: 0.2).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.125,
        help="Fraction of remaining train+val reserved for val (default: 0.125).",
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

    df = load_and_process_data(args.communities_csv)
    available_chars = find_character_images(df, args.avatar_dir)
    if not available_chars:
        print("No characters with images found; nothing to do.")
        return

    chars_df = pd.DataFrame(available_chars)
    comm_counts = chars_df["community"].value_counts().sort_index()
    print("Final community distribution (with images):")
    for cid, count in comm_counts.items():
        label = COMMUNITY_LABELS.get(int(cid), f"Community_{cid}")
        print(f"  {cid}: {label} -> {count:,} characters")

    print("Creating train/val/test splits…")
    train_val_chars, test_chars = train_test_split(
        available_chars,
        test_size=args.test_split,
        random_state=args.random_state,
        stratify=[char["community"] for char in available_chars],
    )

    train_chars, val_chars = train_test_split(
        train_val_chars,
        test_size=args.val_split,
        random_state=args.random_state + 1,
        stratify=[char["community"] for char in train_val_chars],
    )

    total = len(available_chars)
    print("Split sizes:")
    print(f"  Train: {len(train_chars):,} ({len(train_chars) / total * 100:.1f}%)")
    print(f"  Val:   {len(val_chars):,} ({len(val_chars) / total * 100:.1f}%)")
    print(f"  Test:  {len(test_chars):,} ({len(test_chars) / total * 100:.1f}%)")

    output_base_dir = args.output_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying images to {output_base_dir}…")
    train_copied, train_skipped, train_errors = copy_images_to_directory(
        train_chars,
        output_base_dir,
        "train",
        skip_existing=args.skip_existing_images,
        use_symlinks=args.use_symlinks,
        max_workers=args.max_workers,
    )
    val_copied, val_skipped, val_errors = copy_images_to_directory(
        val_chars,
        output_base_dir,
        "val",
        skip_existing=args.skip_existing_images,
        use_symlinks=args.use_symlinks,
        max_workers=args.max_workers,
    )
    test_copied, test_skipped, test_errors = copy_images_to_directory(
        test_chars,
        output_base_dir,
        "test",
        skip_existing=args.skip_existing_images,
        use_symlinks=args.use_symlinks,
        max_workers=args.max_workers,
    )

    all_chars = train_chars + val_chars + test_chars
    labels_data = []
    for char in all_chars:
        if char in train_chars:
            split = "train"
        elif char in val_chars:
            split = "val"
        else:
            split = "test"
        cid = int(char["community"])
        labels_data.append(
            {
                "character_json": char["character_json"],
                "character_name": char["character_name"],
                "image_filename": char["image_filename"],
                "community": cid,
                "community_label": COMMUNITY_LABELS.get(cid, f"Community_{cid}"),
                "split": split,
            }
        )

    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(output_base_dir / "labels.csv", index=False)

    train_df = pd.DataFrame(train_chars)
    val_df = pd.DataFrame(val_chars)
    test_df = pd.DataFrame(test_chars)

    train_counts = train_df["community"].value_counts().sort_index()
    val_counts = val_df["community"].value_counts().sort_index()
    test_counts = test_df["community"].value_counts().sort_index()

    metadata = {
        "task": "community_classification",
        "description": "8-class personality community classification (UMAP/Leiden clusters).",
        "num_classes": len(COMMUNITY_LABELS),
        "communities": {
            str(cid): COMMUNITY_LABELS[cid] for cid in sorted(COMMUNITY_LABELS.keys())
        },
        "dataset_size": {
            "total_samples": len(all_chars),
            "train_samples": len(train_chars),
            "val_samples": len(val_chars),
            "test_samples": len(test_chars),
        },
        "class_distribution": {
            "train": {str(int(k)): int(v) for k, v in train_counts.to_dict().items()},
            "val": {str(int(k)): int(v) for k, v in val_counts.to_dict().items()},
            "test": {str(int(k)): int(v) for k, v in test_counts.to_dict().items()},
        },
        "images": {
            "train_copied": train_copied,
            "val_copied": val_copied,
            "test_copied": test_copied,
            "total_copied": train_copied + val_copied + test_copied,
            "copy_errors": train_errors + val_errors + test_errors,
        },
    }

    with (output_base_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Community classification dataset created.")
    print(f"Output directory: {output_base_dir}")
    print(f"Total samples: {len(all_chars):,}")
    print(f"Classes: {len(COMMUNITY_LABELS)} communities")
    print(f"Images copied: {train_copied + val_copied + test_copied:,}")


if __name__ == "__main__":
    main()
