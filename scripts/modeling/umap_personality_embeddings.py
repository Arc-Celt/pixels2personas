"""
UMAP dimensionality reduction for personality embeddings.

Applies UMAP to reduce personality keyword embeddings to 2D and writes out a
CSV with per-character coordinates.

Example usage:

python scripts/modeling/umap_personality_embeddings.py \
  --embeddings-jsonl /path/to/char_personality_qwen3_32b_fp8_embeddings.jsonl \
  --output-csv /path/to/qwen3_32b_fp8_personality_umap_2d.csv \
  --n-components 2 \
  --random-state 42
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from umap import UMAP

warnings.filterwarnings("ignore")


def load_embeddings_from_jsonl(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load embeddings and metadata from a JSONL file."""
    print(f"Loading embeddings from: {file_path}")

    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    df = pd.DataFrame(data)

    if "embedding" not in df.columns:
        raise ValueError("No 'embedding' column found in JSONL file.")

    print(f"Loaded {len(df)} embeddings with {len(df.iloc[0]['embedding'])} dimensions")

    return df, "embedding"


def apply_umap(
    embeddings, n_components: int = 2, random_state: int = 42
) -> Tuple[UMAP, np.ndarray]:
    """Apply UMAP dimensionality reduction."""
    print(f"Applying UMAP reduction to {n_components} dimensions...")

    X = np.vstack(embeddings)
    print(f"Input matrix shape: {X.shape}")

    umap_reducer = UMAP(
        n_components=n_components,
        random_state=random_state,
    )

    X_umap = umap_reducer.fit_transform(X)
    print(f"UMAP output shape: {X_umap.shape}")

    return umap_reducer, X_umap


def save_umap_results(
    df: pd.DataFrame,
    umap_coords: np.ndarray,
    output_csv: Path,
) -> None:
    """Save UMAP 2D coordinates alongside character identifiers."""
    print("Saving personality UMAP results...")

    df = df.copy()
    df["umap_x"] = umap_coords[:, 0]
    df["umap_y"] = umap_coords[:, 1]

    result_cols = ["character_json", "character_name", "umap_x", "umap_y"]
    available_cols = [col for col in result_cols if col in df.columns]

    missing = set(result_cols) - set(available_cols)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[available_cols].to_csv(output_csv, index=False)

    print(f"Saved personality UMAP results to {output_csv}")
    print(f"Saved {len(df)} entries with columns: {available_cols}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for UMAP over personality embeddings."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply UMAP to personality keyword embeddings and save 2D coordinates."
        )
    )
    parser.add_argument(
        "--embeddings-jsonl",
        type=Path,
        required=True,
        help="JSONL file containing per-character personality embeddings.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for 2D UMAP coordinates.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of UMAP output dimensions (default: 2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for UMAP reproducibility (default: 42).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    embeddings_file: Path = args.embeddings_jsonl
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    print("=" * 50)
    print("UMAP Reduction for Personality Embeddings")
    print("=" * 50)

    df, embed_col = load_embeddings_from_jsonl(embeddings_file)
    _, X_umap = apply_umap(
        embeddings=df[embed_col].values,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    save_umap_results(df=df, umap_coords=X_umap, output_csv=args.output_csv)

    print(f"Personality UMAP complete. Shape: {X_umap.shape}")


if __name__ == "__main__":
    main()
