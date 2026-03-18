"""
Community classification baselines (random and majority) for the 8-class
personality communities, using the same 70/10/20 stratified split as the
PC-based XGBoost and Random Forest classifiers.
- loads community labels and visual PCs CSVs
- aligns samples on `character_json`
- performs a single 70/10/20 stratified split
- evaluates:
  - uniform random predictions baseline
  - majority-class baseline
- optionally expands the results across a list of PC counts for table alignment.

Example:

python scripts/modeling/predict_archetype_from_visual_pcs__baseline.py \
  --community-csv /path/to/personality_communities_umap.csv \
  --visual-pcs-csv /path/to/visual_top100_pcs.csv \
  --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(community_csv: Path, visual_pcs_csv: Path) -> pd.DataFrame:
    logger.info("Loading community labels and aligning IDs…")
    df_comm = pd.read_csv(community_csv)
    df_pcs = pd.read_csv(visual_pcs_csv)

    for col in ["character_json", "character_name"]:
        if col in df_comm.columns:
            df_comm[col] = df_comm[col].astype(str)
    if "character_json" in df_pcs.columns:
        df_pcs["character_json"] = df_pcs["character_json"].astype(str)

    df_comm = df_comm[df_comm["community"].notna()].copy()
    df_comm["community"] = df_comm["community"].astype(int)
    df_comm = df_comm[(df_comm["community"] >= 0) & (df_comm["community"] <= 7)]

    df = pd.merge(df_comm, df_pcs[["character_json"]], on="character_json", how="inner")
    logger.info("Merged records: %s", f"{len(df):,}")
    return df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    f1_weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def random_uniform_baseline(y_test: np.ndarray, rng: np.random.RandomState) -> dict:
    classes = np.unique(y_test)
    y_pred = rng.choice(classes, size=len(y_test), replace=True)
    return evaluate(y_test, y_pred)


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    maj = np.bincount(y_train).argmax()
    y_pred = np.full_like(y_test, fill_value=maj)
    return evaluate(y_test, y_pred)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random and majority baselines for 8-class archetype prediction from visual PCs."
    )
    parser.add_argument(
        "--community-csv",
        type=Path,
        required=True,
        help="CSV with community labels (must contain 'character_json' and 'community').",
    )
    parser.add_argument(
        "--visual-pcs-csv",
        type=Path,
        required=True,
        help="CSV with visual PCA features (must contain 'character_json').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where baseline metrics CSVs will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expand-pcs",
        action="store_true",
        help="Replicate baseline rows across PCs (for table alignment).",
    )
    parser.add_argument(
        "--pc-list",
        type=str,
        default="20,50,100",
        help="Comma-separated PC counts when using --expand-pcs (default: 20,50,100).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    output_dir = args.output_dir / "pc_community_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.community_csv, args.visual_pcs_csv)
    y_all = df["community"].astype(int).values

    idx = np.arange(len(y_all))
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        idx, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,
        random_state=args.seed + 1,
        stratify=y_train_val,
    )
    logger.info(
        "Split sizes: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )

    rnd_metrics = random_uniform_baseline(y_test, rng)
    maj_metrics = majority_baseline(y_train, y_test)

    if args.expand_pcs:
        pcs_list = [int(x) for x in args.pc_list.split(",") if str(x).strip()]
        rows = []
        for p in pcs_list:
            rows.append(
                {
                    "model": "Baseline (random uniform)",
                    "pcs": p,
                    "accuracy": float(rnd_metrics["accuracy"]),
                    "f1_macro": float(rnd_metrics["f1_macro"]),
                    "f1_weighted": float(rnd_metrics["f1_weighted"]),
                }
            )
            rows.append(
                {
                    "model": "Baseline (majority)",
                    "pcs": p,
                    "accuracy": float(maj_metrics["accuracy"]),
                    "f1_macro": float(maj_metrics["f1_macro"]),
                    "f1_weighted": float(maj_metrics["f1_weighted"]),
                }
            )
        results_df = pd.DataFrame(rows).sort_values(["model", "pcs"])
        out_csv = output_dir / "pc_baseline_results.csv"
    else:
        results_df = pd.DataFrame(
            [
                {
                    "model": "Baseline (random uniform)",
                    "pcs": None,
                    "accuracy": float(rnd_metrics["accuracy"]),
                    "f1_macro": float(rnd_metrics["f1_macro"]),
                    "f1_weighted": float(rnd_metrics["f1_weighted"]),
                },
                {
                    "model": "Baseline (majority)",
                    "pcs": None,
                    "accuracy": float(maj_metrics["accuracy"]),
                    "f1_macro": float(maj_metrics["f1_macro"]),
                    "f1_weighted": float(maj_metrics["f1_weighted"]),
                },
            ]
        )
        out_csv = output_dir / "baseline_results.csv"

    results_df.to_csv(out_csv, index=False)
    logger.info("Saved baseline results: %s", out_csv)


if __name__ == "__main__":
    main()
