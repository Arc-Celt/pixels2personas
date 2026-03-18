"""
Principal Component Analysis for avatar (visual) embeddings.

Runs PCA on pre-normalized DINOv2 visual embeddings and saves:
- explained variance statistics
- per-character PC loadings
- example analyses for extreme characters on leading PCs
- optional avatar image copies for inspection.

Example usage:

python scripts/modeling/pca_avatar_embeddings.py \
  --embeddings-csv /path/to/char_visual_embeddings_normalized.csv \
  --image-dir /path/to/mal_char_ind_hr_images \
  --output-dir /path/to/output/pca_visual \
  --n-components 100 \
  --top-pcs-to-analyze 10
"""

import argparse
import ast
import os
import shutil
import warnings
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_normalized_embeddings(
    file_path: Path, embed_column: str = "visual_embedding"
) -> pd.DataFrame:
    """Load normalized visual embeddings from CSV file."""
    print(f"Loading embeddings from: {file_path}")
    df = pd.read_csv(file_path)

    if embed_column not in df.columns:
        raise ValueError(
            f"Expected embedding column '{embed_column}' not found in CSV."
        )

    # Convert serialized list back to Python list safely
    df[embed_column] = df[embed_column].apply(ast.literal_eval)

    if df.empty:
        raise ValueError("No embeddings found in the CSV.")

    dim = len(df.iloc[0][embed_column])
    print(f"Loaded {len(df)} embeddings with {dim} dimensions")

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


def find_image_file(image_filename: str, image_dir: Path) -> Path | None:
    """Find the corresponding image file for a character."""
    if not image_filename:
        return None

    image_path = image_dir / image_filename
    if image_path.exists():
        return image_path

    return None


def analyze_pc_examples(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    explained_var: np.ndarray,
    output_dir: Path,
    top_pcs: int,
    image_dir: Path | None = None,
) -> None:
    """Save top/bottom examples for leading PCs, optionally copying avatar images."""
    print(f"Analyzing top/bottom examples for top {top_pcs} PCs...")

    analysis_dir = output_dir / "visual_pc_analysis"
    analysis_dir.mkdir(exist_ok=True)

    analysis_path = analysis_dir / "visual_pc_examples_analysis.txt"
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write("PCA Examples Analysis for VISUAL Embeddings\n")
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
                char_name = df.iloc[idx].get("character_name", "")
                pc_value = pc_values[idx]
                f.write(f"  {i + 1:2d}. {char_name} (PC{pc_num}: {pc_value:.4f})\n")

            f.write("\nBOTTOM 10 Examples:\n")
            for i, idx in enumerate(bottom_indices):
                char_name = df.iloc[idx].get("character_name", "")
                pc_value = pc_values[idx]
                f.write(f"  {i + 1:2d}. {char_name} (PC{pc_num}: {pc_value:.4f})\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"PC analysis saved to: {analysis_path}")

    # Optionally copy avatar images for qualitative inspection
    if image_dir is not None and "image_file" in df.columns:
        print("Copying avatar images for visual PC analysis...")
        images_dir = analysis_dir / "avatar_images"
        images_dir.mkdir(exist_ok=True)

        for pc_idx in range(top_pcs):
            pc_num = pc_idx + 1
            pc_values = X_pca[:, pc_idx]

            pc_dir = images_dir / f"pc_{pc_num}"
            pc_dir.mkdir(exist_ok=True)

            top10_dir = pc_dir / "top10"
            bottom10_dir = pc_dir / "bottom10"
            top10_dir.mkdir(exist_ok=True)
            bottom10_dir.mkdir(exist_ok=True)

            # Top 10
            top_indices = np.argsort(pc_values)[-10:]
            for i, idx in enumerate(reversed(top_indices)):
                char_name = df.iloc[idx].get("character_name", "")
                image_filename = df.iloc[idx]["image_file"]

                image_path = find_image_file(image_filename, image_dir)
                if image_path:
                    original_filename = image_path.name
                    new_path = top10_dir / original_filename
                    try:
                        shutil.copy2(image_path, new_path)
                        print(
                            f"  Copied image for {char_name} (PC{pc_num} top {i + 1})"
                        )
                    except Exception as e:
                        print(f"  Failed to copy image for {char_name}: {e}")
                else:
                    print(f"  No image found for {char_name}")

            # Bottom 10
            bottom_indices = np.argsort(pc_values)[:10]
            for i, idx in enumerate(bottom_indices):
                char_name = df.iloc[idx].get("character_name", "")
                image_filename = df.iloc[idx]["image_file"]

                image_path = find_image_file(image_filename, image_dir)
                if image_path:
                    original_filename = image_path.name
                    new_path = bottom10_dir / original_filename
                    try:
                        shutil.copy2(image_path, new_path)
                        print(
                            f"  Copied image for {char_name} (PC{pc_num} bottom {i + 1})"
                        )
                    except Exception as e:
                        print(f"  Failed to copy image for {char_name}: {e}")
                else:
                    print(f"  No image found for {char_name}")
    else:
        print(
            "No image copying performed (either no image_dir or no image_file column)."
        )


def save_pca_results(
    df: pd.DataFrame,
    explained_var: np.ndarray,
    output_dir: Path,
    n_components: int,
) -> None:
    """Save PCA variance statistics and per-character PC loadings."""
    print("Saving visual PCA results...")

    variance_path = output_dir / "visual_explained_variance.txt"
    with variance_path.open("w", encoding="utf-8") as f:
        f.write("PCA Analysis Results for VISUAL Embeddings\n")
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

    pc_cols = [f"pca_{i + 1}" for i in range(n_components)]
    base_cols = ["character_json", "character_name"]
    extra_cols = []
    if "image_file" in df.columns:
        extra_cols.append("image_file")

    result_cols = base_cols + pc_cols + extra_cols
    available_cols = [col for col in result_cols if col in df.columns]

    primary_path = output_dir / "visual_top100_pcs.csv"
    df[available_cols].to_csv(primary_path, index=False)

    alias_path = output_dir / "visual_top_pcs.csv"
    if alias_path != primary_path:
        df[available_cols].to_csv(alias_path, index=False)

    print(f"Saved visual PCA results to: {primary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for PCA over avatar (visual) embeddings."""
    parser = argparse.ArgumentParser(
        description="Run PCA on avatar (visual) embeddings and save analysis outputs."
    )
    parser.add_argument(
        "--embeddings-csv",
        type=Path,
        required=True,
        help="CSV file with normalized visual embeddings and metadata.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=False,
        help="Directory with avatar images; if provided, copies example images per PC.",
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
    print("PCA Analysis for Avatar (Visual) Embeddings")
    print("=" * 50)

    visual_df = load_normalized_embeddings(args.embeddings_csv)
    embeddings_list = visual_df["visual_embedding"].tolist()

    visual_pca, visual_X_pca, visual_var, visual_cumvar = perform_pca(
        embeddings=embeddings_list,
        n_components=args.n_components,
        random_state=args.random_state,
    )

    for i in range(args.n_components):
        visual_df[f"pca_{i + 1}"] = visual_X_pca[:, i]

    save_pca_results(
        df=visual_df,
        explained_var=visual_var,
        output_dir=output_dir,
        n_components=args.n_components,
    )
    analyze_pc_examples(
        df=visual_df,
        X_pca=visual_X_pca,
        explained_var=visual_var,
        output_dir=output_dir,
        top_pcs=args.top_pcs_to_analyze,
        image_dir=args.image_dir,
    )

    # Persist PCA model for downstream analysis
    model_path = output_dir / "visual_pca_model.pkl"
    with model_path.open("wb") as f:
        import pickle

        pickle.dump(visual_pca, f)
    print(f"Visual PCA model saved to: {model_path}")

    print(f"Visual PCA complete. Total variance: {visual_cumvar[-1]:.3f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
