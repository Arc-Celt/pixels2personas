"""
Community detection on personality UMAP coordinates using a distance-thresholded
graph and the Leiden algorithm.

Takes 2D UMAP coordinates for characters, builds a similarity graph based on
pairwise distances, runs Leiden clustering, computes centrality, and writes
out a CSV with community assignments and centrality scores.

Example usage:

python scripts/modeling/personality_archetype_community_detection.py \
  --umap-csv /path/to/qwen3_32b_fp8_personality_umap_2d.csv \
  --output-csv /path/to/personality_communities_umap.csv \
  --cache-dir /path/to/cache \
  --distance-percentile 5 \
  --use-parallel
"""

import argparse
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import psutil
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_umap_data(file_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load UMAP 2D coordinates for community detection."""
    print(f"Loading UMAP data: {file_path}")
    df = pd.read_csv(file_path)

    required_cols = ["umap_x", "umap_y", "character_json", "character_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    coords = df[["umap_x", "umap_y"]].values.astype("float32")
    print(f"Loaded {len(df):,} points")

    return df, coords


def compute_distance_threshold(
    coords: np.ndarray,
    percentile: float,
    cache_dir: Path,
    sample_size_threshold: int = 20_000,
) -> float:
    """Calculate distance threshold from percentile using optional sampling and caching."""
    print(f"Computing distance threshold (p={percentile}%)...")

    cache_file = cache_dir / f"personality_distances_thresh_{percentile}.npz"

    if cache_file.exists():
        try:
            data = np.load(cache_file)
            threshold = float(data["threshold"][0])
            print(f"Loaded cached threshold: {threshold:.6f}")
            return threshold
        except Exception as e:
            print(f"[WARN] Failed to load threshold cache ({cache_file}): {e}")

    n_points = len(coords)

    if n_points > sample_size_threshold:
        n_samples = min(500_000, n_points // 10)

        rng = np.random.default_rng()
        sample_indices = rng.choice(n_points, size=n_samples, replace=False)
        sample_coords = coords[sample_indices]
        distances = pdist(sample_coords)
    else:
        distances = pdist(coords)

    threshold = float(np.percentile(distances, percentile))
    print(f"Distance threshold: {threshold:.6f}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        threshold=np.array([threshold]),
        percentile=np.array([percentile]),
    )

    return threshold


def process_chunk_vectorized(
    start_idx: int, chunk_coords: np.ndarray, all_coords: np.ndarray, threshold: float
) -> Tuple[list[tuple[int, int]], list[float]]:
    """Process a chunk of coordinates using vectorized distance computation."""
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for i, point_coord in enumerate(chunk_coords):
        global_idx = start_idx + i
        remaining_coords = all_coords[global_idx + 1 :]
        if len(remaining_coords) == 0:
            continue

        distances = np.sqrt(np.sum((remaining_coords - point_coord) ** 2, axis=1))
        valid_indices = np.where(distances <= threshold)[0]

        for local_idx in valid_indices:
            global_j = global_idx + 1 + local_idx
            distance = float(distances[local_idx])
            edges.append((global_idx, global_j))
            weights.append(1.0 / (1.0 + distance))

    return edges, weights


def build_distance_threshold_graph(
    coords: np.ndarray,
    threshold: float,
    cache_dir: Path,
    use_parallel: bool = True,
    max_workers: int | None = None,
    chunk_size_optimized: int = 5000,
) -> Tuple[list[tuple[int, int]], list[float]]:
    """Build a distance-thresholded graph using vectorized operations and caching."""
    n_points = len(coords)
    print(f"Building graph (n={n_points:,}, threshold={threshold:.6f})...")

    if max_workers is None:
        max_workers = min(16, mp.cpu_count())

    cache_file = cache_dir / f"personality_raw_distances_{n_points}.npz"

    if cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=True)
            all_distances = data["distances"]
            all_pairs = data["pairs"]
            print(f"Loaded cached distances ({len(all_pairs):,} pairs)")

            if len(all_pairs) > 0:
                max_node_idx = max(max(pair) for pair in all_pairs)
                if max_node_idx >= n_points:
                    print("[WARN] Distance cache does not match dataset size; recalculating")
                    raise ValueError("Cache mismatch with dataset size")

            valid_mask = all_distances <= threshold
            edges = [pair for pair, valid in zip(all_pairs, valid_mask) if valid]
            weights = [
                1.0 / (1.0 + dist)
                for dist, valid in zip(all_distances, valid_mask)
                if valid
            ]

            print(f"Edges after thresholding: {len(edges):,}")
            return edges, weights
        except Exception as e:
            print(f"[WARN] Failed to load distance cache ({cache_file}): {e}")

    if n_points > 10_000:
        chunk_data = []
        for start_idx in range(0, n_points, chunk_size_optimized):
            end_idx = min(start_idx + chunk_size_optimized, n_points)
            chunk_coords = coords[start_idx:end_idx]
            chunk_data.append((start_idx, chunk_coords, coords, threshold))

        all_edges: list[tuple[int, int]] = []
        all_weights: list[float] = []

        if use_parallel and len(chunk_data) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_chunk_vectorized,
                        start_idx,
                        chunk_coords,
                        coords,
                        threshold,
                    )
                    for (start_idx, chunk_coords, coords, threshold) in chunk_data
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing chunks",
                ):
                    edges, weights = future.result()
                    all_edges.extend(edges)
                    all_weights.extend(weights)
        else:
            for start_idx, chunk_coords, coords_ref, threshold_val in tqdm(
                chunk_data, desc="Processing chunks"
            ):
                edges, weights = process_chunk_vectorized(
                    start_idx, chunk_coords, coords_ref, threshold_val
                )
                all_edges.extend(edges)
                all_weights.extend(weights)

        cache_dir.mkdir(parents=True, exist_ok=True)
        all_pairs = np.array(all_edges, dtype=object)
        all_distances = np.array([1.0 / w - 1.0 for w in all_weights], dtype=float)
        np.savez(cache_file, pairs=all_pairs, distances=all_distances)
        print(f"Saved distance cache ({len(all_pairs):,} pairs)")

        edges, weights = all_edges, all_weights
    else:
        distances = pdist(coords)
        distance_matrix = squareform(distances)

        upper_triangle = np.triu(distance_matrix <= threshold, k=1)
        edge_indices = np.where(upper_triangle)

        edges = []
        weights = []
        for i, j in zip(edge_indices[0], edge_indices[1]):
            distance = float(distance_matrix[i, j])
            similarity_weight = 1.0 / (1.0 + distance)
            edges.append((i, j))
            weights.append(similarity_weight)

    print(f"Distance-thresholded graph: {len(edges):,} edges from {n_points:,} points")

    return edges, weights


def calculate_node_centrality(
    g: ig.Graph, communities: np.ndarray, df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate degree centrality for each node (global and within-community)."""
    degree_centrality = g.degree()
    df["degree_centrality"] = degree_centrality
    df["community_degree_centrality"] = 0.0

    unique_communities = sorted(set(communities))

    for community_id in unique_communities:
        community_nodes = [
            i for i, comm in enumerate(communities) if comm == community_id
        ]
        if len(community_nodes) < 2:
            continue

        subgraph = g.subgraph(community_nodes)
        community_degree = subgraph.degree()

        for local_idx, node_idx in enumerate(community_nodes):
            df.loc[node_idx, "community_degree_centrality"] = community_degree[
                local_idx
            ]

    return df


def run_umap_leiden_community_detection(
    df: pd.DataFrame,
    coords: np.ndarray,
    distance_percentile: float,
    cache_dir: Path,
    use_parallel: bool = True,
    max_workers: int | None = None,
    chunk_size_optimized: int = 5000,
) -> Tuple[pd.DataFrame, np.ndarray, float, float]:
    """Run UMAP-based Leiden community detection with a distance threshold."""
    if max_workers is None:
        max_workers = min(16, mp.cpu_count())

    start_time = time.time()
    distance_threshold = compute_distance_threshold(
        coords=coords,
        percentile=distance_percentile,
        cache_dir=cache_dir,
    )
    edges, weights = build_distance_threshold_graph(
        coords=coords,
        threshold=distance_threshold,
        cache_dir=cache_dir,
        use_parallel=use_parallel,
        max_workers=max_workers,
        chunk_size_optimized=chunk_size_optimized,
    )
    n_points = len(df)

    if len(edges) == 0:
        print("Warning: No edges below threshold; using single community.")
        communities = np.zeros(n_points, dtype=int)
        df["community"] = communities
        return df, communities, distance_threshold, 0.0

    g = ig.Graph()
    g.add_vertices(n_points)
    g.add_edges(edges)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=0.5,
    )
    communities = np.array(partition.membership)

    modularity = g.modularity(communities, weights="weight")
    df["community"] = communities
    df = calculate_node_centrality(g, communities, df)

    community_sizes = pd.Series(communities).value_counts().sort_values(ascending=False)
    n_communities = len(community_sizes)

    total_time = time.time() - start_time
    print(
        f"Done: {n_communities} communities "
        f"(largest={community_sizes.iloc[0]}, smallest={community_sizes.iloc[-1]}), "
        f"modularity={modularity:.3f}, time={total_time:.1f}s"
    )

    return df, communities, distance_threshold, modularity


def save_results(
    df: pd.DataFrame,
    threshold: float,
    modularity: float,
    output_csv: Path,
) -> None:
    """Save community detection results to CSV."""
    essential_cols = [
        "character_json",
        "character_name",
        "umap_x",
        "umap_y",
        "community",
    ]

    if "degree_centrality" in df.columns:
        essential_cols.append("degree_centrality")
    if "community_degree_centrality" in df.columns:
        essential_cols.append("community_degree_centrality")

    available_cols = [col for col in essential_cols if col in df.columns]
    output_df = df[available_cols].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    n_communities = len(set(df["community"]))
    print(f"Saved: {output_csv} (n={n_communities}, modularity={modularity:.3f})")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for UMAP-based community detection on personality embeddings."""
    parser = argparse.ArgumentParser(
        description=(
            "Run community detection on 2D personality UMAP coordinates using a "
            "distance-thresholded graph and the Leiden algorithm."
        )
    )
    parser.add_argument(
        "--umap-csv",
        type=Path,
        required=True,
        help="CSV file with 2D UMAP coordinates and character metadata.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for community assignments and centrality scores.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory for distance and threshold caches.",
    )
    parser.add_argument(
        "--distance-percentile",
        type=float,
        default=5.0,
        help="Percentile of pairwise distances used as threshold (default: 5).",
    )
    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="Enable parallel chunk processing for graph construction.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(16, mp.cpu_count()),
        help="Maximum number of worker threads when --use-parallel is set.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Chunk size for vectorized distance computation (default: 5000).",
    )
    parser.add_argument(
        "--sample-size-threshold",
        type=int,
        default=20_000,
        help=(
            "Number of points above which distance percentile is estimated via "
            "sampling instead of all-pairs computation (default: 20000)."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    umap_csv: Path = args.umap_csv
    if not umap_csv.exists():
        raise FileNotFoundError(f"UMAP CSV not found: {umap_csv}")

    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    df, coords = load_umap_data(umap_csv)
    df, communities, threshold, modularity = run_umap_leiden_community_detection(
        df=df,
        coords=coords,
        distance_percentile=args.distance_percentile,
        cache_dir=cache_dir,
        use_parallel=args.use_parallel,
        max_workers=args.max_workers,
        chunk_size_optimized=args.chunk_size,
    )
    save_results(
        df=df, threshold=threshold, modularity=modularity, output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
