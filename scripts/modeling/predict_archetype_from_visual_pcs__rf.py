"""
Random Forest community classifier on PCA visual features for interpretability
for the 8-class personality communities.

This script:
- merges a community CSV (with `character_json`, `community`) and a visual PCs CSV
  (with `character_json`, `pca_1..pca_N`)
- performs a single 70/10/20 stratified split
- trains RandomForestClassifier baselines for PCs = 20, 50, 100
- computes SHAP-based feature importances (optional)
- saves metrics, feature importances, SHAP statistics, and models.

Example:

python scripts/modeling/predict_archetype_from_visual_pcs__rf.py \
  --community-csv /path/to/personality_communities_umap.csv \
  --visual-pcs-csv /path/to/visual_top100_pcs.csv \
  --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(
    community_csv: Path, visual_pcs_csv: Path
) -> tuple[pd.DataFrame, list[str]]:
    logger.info("Loading community labels and visual PCs…")
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

    df = pd.merge(df_comm, df_pcs, on="character_json", how="inner")
    logger.info("Merged records: %s", f"{len(df):,}")

    pc_cols = sorted(
        [c for c in df.columns if c.startswith("pca_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not pc_cols:
        raise ValueError("No PCA columns found. Expected 'pca_1', 'pca_2', …")
    return df, pc_cols


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )[2]
    cm = confusion_matrix(y_true, y_pred)

    num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    zero_pred_classes = np.where(pred_counts == 0)[0]
    if len(zero_pred_classes) > 0:
        logger.warning(
            "No predictions made for classes: %s", zero_pred_classes.tolist()
        )

    true_counts = np.bincount(y_true, minlength=num_classes)
    logger.info(
        "True class distribution: %s", dict(zip(range(num_classes), true_counts))
    )
    logger.info(
        "Predicted class distribution: %s", dict(zip(range(num_classes), pred_counts))
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_macro": f1,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random Forest on PCA visual features for 8-class community classification."
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
        help="CSV with visual PCA features (must contain 'character_json' and 'pca_*' columns).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where models, metrics, and plots will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--test-single-pc",
        type=int,
        default=None,
        help="Test only a single PC value (e.g., 100) instead of [20, 50, 100].",
    )
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Compute SHAP values for interpretability on the test set.",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=2000,
        help="Max number of test samples to explain with SHAP (default: 2000).",
    )
    parser.add_argument(
        "--shap-topk",
        type=int,
        default=20,
        help="Top-K PCs to display in SHAP bar plot (default: 20).",
    )
    parser.add_argument(
        "--shap-seed",
        type=int,
        default=42,
        help="Random seed for SHAP subsampling (default: 42).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    out_dir = args.output - dir / "pc_community_rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, pc_cols = load_data(args.community_csv, args.visual_pcs_csv)
    X_all = df[pc_cols]
    y_all = df["community"].astype(int).values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
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

    train_counts = np.bincount(y_train, minlength=8)
    logger.info("Training class distribution: %s", dict(zip(range(8), train_counts)))
    logger.info("Using class_weight='balanced' for RandomForestClassifier")

    if args.test_single_pc is not None:
        if args.test_single_pc <= X_all.shape[1]:
            pcs_list = [args.test_single_pc]
            logger.info("Testing single PC value: %d", args.test_single_pc)
        else:
            logger.warning(
                "Requested PC value %d exceeds available PCs %d",
                args.test_single_pc,
                X_all.shape[1],
            )
            pcs_list = [min(100, X_all.shape[1])]
    else:
        pcs_list = [p for p in [20, 50, 100] if p <= X_all.shape[1]]
        if not pcs_list:
            pcs_list = [min(20, X_all.shape[1])]

    results = []
    pc100_test_metrics = None

    for n in pcs_list:
        logger.info("Training RandomForest (PCs=%d)…", n)
        X_train_pc = X_train.iloc[:, :n].to_numpy(dtype=np.float32)
        X_val_pc = X_val.iloc[:, :n].to_numpy(dtype=np.float32)
        X_test_pc = X_test.iloc[:, :n].to_numpy(dtype=np.float32)

        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            class_weight="balanced",
        )
        rf.fit(X_train_pc, y_train)
        y_pred = rf.predict(X_test_pc)
        metrics = evaluate(y_test, y_pred)

        if n == 100:
            pc100_test_metrics = metrics

        results.append(
            {
                "pcs": n,
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
            }
        )

        joblib.dump(
            {"model": rf, "pc_columns": pc_cols[:n], "model_name": "random_forest"},
            out_dir / f"rf_pc{n}_model.joblib",
        )

        importances = rf.feature_importances_
        feat_names = pc_cols[:n]
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df.sort_values("importance", ascending=False).to_csv(
            out_dir / f"rf_pc{n}_importances.csv", index=False
        )

        if args.enable_shap:
            rs = np.random.RandomState(args.shap_seed)
            num_samples = X_test_pc.shape[0]
            max_samples = max(1, int(args.shap_max_samples))
            if num_samples > max_samples:
                idx = rs.choice(num_samples, size=max_samples, replace=False)
                Xte_shap = X_test_pc[idx]
            else:
                Xte_shap = X_test_pc

            logger.info(
                "Computing SHAP values for RF (PCs=%d) on %d samples…",
                n,
                Xte_shap.shape[0],
            )
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(Xte_shap)

            if isinstance(shap_values, list):
                abs_per_class = [np.abs(sv) for sv in shap_values]
                abs_stacked = np.stack(abs_per_class, axis=0)
                abs_sum_classes = np.sum(abs_stacked, axis=0)
                mean_abs = abs_sum_classes.mean(axis=0)
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            mean_abs = np.asarray(mean_abs).flatten()

            shap_df = pd.DataFrame(
                {"feature": feat_names, "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_csv_path = out_dir / f"rf_pc{n}_shap_mean_abs.csv"
            shap_df.to_csv(shap_csv_path, index=False)
            logger.info("Saved SHAP mean |value| per PC: %s", shap_csv_path)

            topk = max(1, int(args.shap_topk))
            plot_df = shap_df.head(topk).copy()
            plt.figure(figsize=(8, 6))
            plt.barh(
                plot_df["feature"][::-1],
                plot_df["mean_abs_shap"][::-1],
                color="#4C72B0",
            )
            plt.xlabel("Mean |SHAP|")
            plt.ylabel("PC feature")
            plt.title(f"RF SHAP Top-{topk} (PCs={n})")
            plt.tight_layout()
            shap_plot_path = out_dir / f"rf_pc{n}_shap_top{topk}_bar.pdf"
            plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Saved SHAP bar plot: %s", shap_plot_path)

    res_df = pd.DataFrame(results).sort_values("pcs")
    res_df.to_csv(out_dir / "rf_results.csv", index=False)
    logger.info("Saved RF results: %s", out_dir / "rf_results.csv")

    if pc100_test_metrics is not None:
        results_json = {
            "task": "pc_community_classification",
            "selected_model": "random_forest",
            "selected_num_pcs": 100,
            "dataset_sizes": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "test_metrics": pc100_test_metrics,
        }

        results_json_path = out_dir / "results.json"
        with results_json_path.open("w") as f:
            json.dump(convert_numpy(results_json), f, indent=2)
        logger.info("Saved results.json: %s", results_json_path)


if __name__ == "__main__":
    main()
