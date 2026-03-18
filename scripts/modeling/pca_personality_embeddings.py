"""
Principal Component Analysis for personality embeddings.

Extracts the top principal components from normalized personality keyword
embeddings and saves:
- explained variance statistics
- per-character top-N PC loadings
- example analyses for extreme characters on leading PCs.

Example usage:

python scripts/modeling/pca_personality_embeddings.py \
  --community-csv /path/to/personality_communities_umap.csv \
  --embeddings-jsonl /path/to/char_personality_qwen3_32b_fp8_embeddings.jsonl \
  --output-dir /path/to/output/pca_personality \
  --n-components 100 \
  --top-pcs-to-analyze 10
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_personality_embeddings(
    community_csv: Path,
    embeddings_jsonl: Path,
) -> pd.DataFrame:
    """Load personality embeddings and merge them with community detection CSV."""
    print(f"Loading personality data from: {community_csv}")

    df = pd.read_csv(community_csv)
    print(f"Loaded {len(df)} characters from community detection CSV")
    print(f"CSV columns: {df.columns.tolist()}")

    if not embeddings_jsonl.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_jsonl}")

    print(f"Loading embeddings from: {embeddings_jsonl}")
    embeddings_data = {}
    with embeddings_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            char_json = entry.get("character_json", "")
            if char_json and "embedding" in entry:
                embeddings_data[char_json] = entry["embedding"]

    print(f"Loaded {len(embeddings_data)} embeddings from JSONL")

    df["embedding"] = df["character_json"].map(embeddings_data)

    # Filter out rows without embeddings
    df = df[df["embedding"].notna()].copy()
    print(f"After merging: {len(df)} characters with embeddings")

    if len(df) == 0:
        raise ValueError("No characters with embeddings found after merging!")

    # Verify embedding dimensions
    first_embedding = df.iloc[0]["embedding"]
    if isinstance(first_embedding, list):
        embed_dim = len(first_embedding)
    else:
        raise ValueError(f"Embedding is not a list: {type(first_embedding)}")

    print(f"Embedding dimension: {embed_dim}")

    return df


def perform_pca(
    embeddings, n_components: int, random_state: int = 42
) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """Perform PCA on normalized embeddings."""
    print(f"Performing PCA with {n_components} components...")

    X = np.vstack(embeddings)
    print(f"Input matrix shape: {X.shape}")

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"PC1 variance: {explained_var[0] * 100:.2f}%")
    print(f"Top 10 PCs: {explained_var[:10].sum() * 100:.2f}%")
    print(f"Top 20 PCs: {explained_var[:20].sum() * 100:.2f}%")
    print(f"Top 25 PCs: {explained_var[:25].sum() * 100:.2f}%")
    print(f"Top 50 PCs: {explained_var[:50].sum() * 100:.2f}%")
    print(f"Top 75 PCs: {explained_var[:75].sum() * 100:.2f}%")
    print(f"Top {n_components} PCs: {explained_var.sum() * 100:.2f}%")

    return pca, X_pca, explained_var, cumulative_var


def analyze_pc_examples(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    explained_var: np.ndarray,
    output_dir: Path,
    top_pcs: int,
) -> None:
    """Save top/bottom examples for leading PCs."""
    print(f"Analyzing top/bottom examples for top {top_pcs} PCs...")

    analysis_dir = output_dir / "personality_pc_analysis"
    analysis_dir.mkdir(exist_ok=True)

    analysis_path = analysis_dir / "personality_pc_examples_analysis.txt"
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write("PCA Examples Analysis for PERSONALITY Embeddings\n")
        f.write("=" * 60 + "\n\n")

        for pc_idx in range(top_pcs):
            pc_num = pc_idx + 1
            pc_values = X_pca[:, pc_idx]

            top_indices = np.argsort(pc_values)[-10:]
            bottom_indices = np.argsort(pc_values)[:10]

            f.write(f"PC{pc_num} (Variance: {explained_var[pc_idx] * 100:.2f}%)\n")
            f.write("-" * 40 + "\n")

            f.write("TOP 10 Examples:\n")
            for i, idx in enumerate(reversed(top_indices)):
                char_name = df.iloc[idx]["character_name"]
                pc_value = pc_values[idx]
                f.write(f"  {i + 1:2d}. {char_name} (PC{pc_num}: {pc_value:.4f})\n")

            f.write("\nBOTTOM 10 Examples:\n")
            for i, idx in enumerate(bottom_indices):
                char_name = df.iloc[idx]["character_name"]
                pc_value = pc_values[idx]
                f.write(f"  {i + 1:2d}. {char_name} (PC{pc_num}: {pc_value:.4f})\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"PC analysis saved to: {analysis_path}")


def save_pca_results(
    df: pd.DataFrame,
    explained_var: np.ndarray,
    output_dir: Path,
    n_components: int,
) -> None:
    """Save PCA variance statistics and per-character PC loadings."""
    print("Saving personality PCA results...")

    variance_path = output_dir / "personality_explained_variance.txt"
    with variance_path.open("w", encoding="utf-8") as f:
        f.write("PCA Analysis Results for PERSONALITY Embeddings\n")
        f.write("=" * 50 + "\n\n")
        f.write("Explained Variance Ratios:\n")
        for i, var in enumerate(explained_var):
            f.write(f"PC_{i + 1}: {var:.6f}\n")
        f.write(f"\nCumulative Explained Variance: {explained_var.sum():.6f}\n")
        f.write(
            f"Total variance explained by top {len(explained_var)} PCs: "
            f"{explained_var.sum() * 100:.2f}%\n"
        )

        f.write("\nVariance Analysis:\n")
        f.write(f"PC1 variance: {explained_var[0] * 100:.2f}%\n")
        f.write(f"Top 10 PCs cumulative: {explained_var[:10].sum() * 100:.2f}%\n")
        f.write(f"Top 25 PCs cumulative: {explained_var[:25].sum() * 100:.2f}%\n")
        f.write(f"Top 50 PCs cumulative: {explained_var[:50].sum() * 100:.2f}%\n")
        f.write(
            f"Top {n_components} PCs cumulative: {explained_var.sum() * 100:.2f}%\n"
        )

    # Save top-N PCs CSV (character info + PC values + optional community metrics)
    pc_cols = [f"pca_{i + 1}" for i in range(n_components)]
    base_cols = ["character_json", "character_name"]

    extra_cols = []
    for col in ["community", "degree_centrality", "community_degree_centrality"]:
        if col in df.columns:
            extra_cols.append(col)

    result_cols = base_cols + pc_cols + extra_cols
    available_cols = [col for col in result_cols if col in df.columns]

    pcs_path = output_dir / "personality_top100_pcs.csv"
    df[available_cols].to_csv(pcs_path, index=False)

    alias_path = output_dir / "personality_top_pcs.csv"
    if alias_path != pcs_path:
        df[available_cols].to_csv(alias_path, index=False)

    print(f"Saved personality PCA results to: {pcs_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for PCA over personality embeddings."""
    parser = argparse.ArgumentParser(
        description="Run PCA on personality keyword embeddings and save summary outputs."
    )
    parser.add_argument(
        "--community-csv",
        type=Path,
        required=True,
        help="CSV file with community detection / metadata for characters.",
    )
    parser.add_argument(
        "--embeddings-jsonl",
        type=Path,
        required=True,
        help="JSONL file with per-character personality embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where PCA results will be written.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Number of principal components to compute (default: 100).",
    )
    parser.add_argument(
        "--top-pcs-to-analyze",
        type=int,
        default=10,
        help="Number of leading PCs for which to save example characters (default: 10).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for PCA reproducibility (default: 42).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("PCA Analysis for Personality Embeddings")
    print("=" * 50)

    print("Processing personality embeddings...")
    personality_df = load_personality_embeddings(
        community_csv=args.community_csv,
        embeddings_jsonl=args.embeddings_jsonl,
    )

    embeddings_list = personality_df["embedding"].tolist()

    personality_pca, personality_X_pca, personality_var, personality_cumvar = (
        perform_pca(
            embeddings=embeddings_list,
            n_components=args.n_components,
            random_state=args.random_state,
        )
    )

    # Add PC columns to dataframe
    for i in range(args.n_components):
        personality_df[f"pca_{i + 1}"] = personality_X_pca[:, i]

    save_pca_results(
        df=personality_df,
        explained_var=personality_var,
        output_dir=output_dir,
        n_components=args.n_components,
    )
    analyze_pc_examples(
        df=personality_df,
        X_pca=personality_X_pca,
        explained_var=personality_var,
        output_dir=output_dir,
        top_pcs=args.top_pcs_to_analyze,
    )

    # Persist PCA model for downstream analysis
    model_path = output_dir / "personality_pca_model.pkl"
    with model_path.open("wb") as f:
        import pickle

        pickle.dump(personality_pca, f)
    print(f"Personality PCA model saved to: {model_path}")

    print(f"Personality PCA complete. Total variance: {personality_cumvar[-1]:.3f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
