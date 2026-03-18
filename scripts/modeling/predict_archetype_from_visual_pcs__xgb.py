"""
Train a community classifier using PCA visual features (PCs) as predictors with XGBoost
for the 8-class personality communities.

The script:
- merges a community CSV (with `character_json`, `community`) and a visual PCs CSV
  (with `character_json`, `pca_1`..`pca_N`)
- performs a stratified 70/10/20 train/val/test split
- optionally runs hyperparameter search
- trains an XGBoost multi-class classifier
- saves metrics, confusion-matrix heatmaps, per-PC models, and a summary CSV.

Example:

python scripts/modeling/predict_archetype_from_visual_pcs__xgb.py \
  --community-csv /path/to/personality_communities_umap.csv \
  --visual-pcs-csv /path/to/visual_top100_pcs.csv \
  --output-dir /path/to/results \
  --num-pcs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


COMMUNITY_LABELS = [
    "0: Energetic Extroverts",
    "1: Steadfast Leaders",
    "2: Cold-hearted Villains",
    "3: Passionate Strivers",
    "4: Kind Caregivers",
    "5: Justice Keepers",
    "6: Reserved Introverts",
    "7: Arrogant Tsunderes",
]


def plot_confusion_matrix_heatmap(
    confusion: np.ndarray, output_dir: Path, split_name: str = "test"
) -> None:
    row_sums = confusion.sum(axis=1, keepdims=True)
    percent_matrix = np.divide(confusion, row_sums, where=row_sums != 0) * 100.0

    annot = np.empty_like(confusion, dtype=object)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            annot[i, j] = f"{percent_matrix[i, j]:.1f}%\n({confusion[i, j]})"

    def to_two_lines(name: str) -> str:
        base = name.split(":", 1)[1].strip() if ":" in name else name.strip()
        words = base.split()
        if len(words) <= 2:
            return "\n".join(words)
        mid = (len(words) + 1) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

    compact_labels = [to_two_lines(lbl) for lbl in COMMUNITY_LABELS]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        percent_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=compact_labels,
        yticklabels=compact_labels,
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"fontsize": 15},
    )
    plt.xlabel("Predicted Prototype", fontsize=24, fontweight="bold")
    plt.ylabel("True Prototype", fontsize=24, fontweight="bold")
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.tight_layout()

    path = output_dir / f"{split_name}_confusion_matrix_percentage.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved percentage-based confusion matrix: %s", path)


def load_data(community_csv: Path, visual_pcs_csv: Path) -> pd.DataFrame:
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

    pc_cols = [c for c in df.columns if c.startswith("pca_")]
    if not pc_cols:
        raise ValueError(
            "No PCA columns found. Expected columns like 'pca_1', 'pca_2', …"
        )

    logger.info("Found %d PCA feature columns", len(pc_cols))
    return df


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    return {
        int(cls): float(n_samples / (n_classes * count))
        for cls, count in zip(classes, counts)
    }


def apply_class_weights(y: np.ndarray, weight_map: dict[int, float]) -> np.ndarray:
    return np.array([weight_map[int(y_i)] for y_i in y], dtype=float)


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
    weighted_f1 = precision_recall_fscore_support(
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
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }


def build_xgb(use_gpu: bool, n_jobs: int) -> xgb.XGBClassifier:
    params = dict(
        objective="multi:softprob",
        num_class=8,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    params["device"] = "cuda" if use_gpu else "cpu"
    return xgb.XGBClassifier(**params)


def get_xgb_param_grid() -> dict:
    return {
        "learning_rate": [0.02, 0.05, 0.07],
        "max_depth": [6, 8, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_lambda": [1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.5, 1.0],
        "gamma": [0, 0.1, 0.2],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCA-based community classification (XGBoost) for 8-class setup."
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
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=100,
        help="Maximum number of PCs to consider (used when `--test-single-pc` is unset).",
    )
    parser.add_argument(
        "--cv-verbose",
        type=int,
        default=1,
        help="Verbosity for RandomizedSearchCV (0=silent, 1=candidates, 2=folds).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for hyperparameter search.",
    )
    parser.add_argument(
        "--cv-iter",
        type=int,
        default=100,
        help="Number of candidates in RandomizedSearch.",
    )
    parser.add_argument(
        "--skip-hyperopt",
        action="store_true",
        help="Skip hyperparameter optimization for faster training (use defaults).",
    )
    parser.add_argument(
        "--test-single-pc",
        type=int,
        default=None,
        help="Test only a single PC count (e.g., 100) instead of a grid [20, 50, 100].",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU-accelerated tree building if available.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Threads for XGBoost training (-1 to use all cores).",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Path to existing model file (joblib) to load and evaluate without retraining.",
    )
    parser.add_argument(
        "--num-pcs-for-model",
        type=int,
        default=None,
        help="Number of PCs to use when loading an existing model (required with --load-model).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = np.random.RandomState(args.seed)
    output_dir: Path = args.output_dir / "pc_community"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.community_csv, args.visual_pcs_csv)
    pc_cols = sorted(
        [c for c in df.columns if c.startswith("pca_")],
        key=lambda x: int(x.split("_")[1]),
    )
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

    if args.load_model:
        if args.num_pcs_for_model is None:
            raise ValueError("--num-pcs-for-model is required when using --load-model")

        logger.info("Loading existing model from: %s", args.load_model)
        model_data = joblib.load(args.load_model)
        estimator = model_data["model"]
        best_n = args.num_pcs_for_model

        Xtest_np = X_test.iloc[:, :best_n].to_numpy(dtype=np.float32)
        y_pred_test = estimator.predict(Xtest_np)
        test_metrics = evaluate(y_test, y_pred_test)

        plot_confusion_matrix_heatmap(
            test_metrics["confusion_matrix"], output_dir, f"test_pc{best_n}"
        )

        results = {
            "task": "pc_community_classification",
            "selected_model": "xgb",
            "selected_num_pcs": best_n,
            "dataset_sizes": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "test_metrics": test_metrics,
        }

        results_json_path = output_dir / "results.json"
        with results_json_path.open("w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        logger.info("Saved results.json: %s", results_json_path)
        logger.info("Test Accuracy: %.4f", test_metrics["accuracy"])
        logger.info("Test Weighted F1: %.4f", test_metrics["weighted_f1"])
        return

    train_counts = np.bincount(y_train, minlength=8)
    logger.info("Training class distribution: %s", dict(zip(range(8), train_counts)))

    weight_map = compute_class_weights(y_train)
    logger.info("Class weights (for balancing): %s", weight_map)
    sample_weights_train = apply_class_weights(y_train, weight_map)
    sample_weights_val = apply_class_weights(y_val, weight_map)
    y_full = np.concatenate([y_train, y_val])
    sample_weights_full = apply_class_weights(y_full, weight_map)

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
            pcs_list = [min(args.num_pcs, X_all.shape[1])]
    else:
        pcs_list = [p for p in [20, 50, 100] if p <= X_all.shape[1]]
        if not pcs_list:
            pcs_list = [min(20, X_all.shape[1])]

    all_results = []

    for best_n in pcs_list:
        logger.info("Using PCs=%d", best_n)

        X_train_pc = X_train.iloc[:, :best_n].to_numpy(dtype=np.float32)
        X_val_pc = X_val.iloc[:, :best_n].to_numpy(dtype=np.float32)
        X_test_pc = X_test.iloc[:, :best_n].to_numpy(dtype=np.float32)

        estimator = build_xgb(use_gpu=args.use_gpu, n_jobs=args.n_jobs)

        if not args.skip_hyperopt:
            param_grid = get_xgb_param_grid()
            if param_grid:
                rs_start = time.time()
                rand_search = RandomizedSearchCV(
                    estimator,
                    param_distributions=param_grid,
                    n_iter=args.cv_iter,
                    scoring="f1_weighted",
                    cv=args.cv_folds,
                    n_jobs=-1,
                    verbose=args.cv_verbose,
                    random_state=args.seed,
                )
                rand_search.fit(X_train_pc, y_train, sample_weight=sample_weights_train)
                estimator = rand_search.best_estimator_
                rs_time = time.time() - rs_start
                logger.info(
                    "RandomizedSearch best params (PCs=%d): %s (took %.1f min)",
                    best_n,
                    rand_search.best_params_,
                    rs_time / 60.0,
                )
        else:
            logger.info("Skipping hyperparameter optimization (using defaults)")

        if isinstance(estimator, xgb.XGBClassifier):
            xgb_params = estimator.get_params()
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda" if args.use_gpu else "cpu"

            dtrain = xgb.DMatrix(X_train_pc, label=y_train, weight=sample_weights_train)
            dval = xgb.DMatrix(X_val_pc, label=y_val, weight=sample_weights_val)

            train_params = {
                "objective": "multi:softprob",
                "num_class": 8,
                "eta": xgb_params.get("learning_rate", 0.05),
                "max_depth": xgb_params.get("max_depth", 8),
                "subsample": xgb_params.get("subsample", 0.8),
                "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
                "reg_lambda": xgb_params.get("reg_lambda", 2.0),
                "reg_alpha": xgb_params.get("reg_alpha", 0.0),
                "eval_metric": "mlogloss",
                "n_jobs": args.n_jobs,
                "tree_method": "hist",
                "device": "cuda" if args.use_gpu else "cpu",
            }

            num_boost_round = 1500
            logger.info(
                "Starting early stopping to select n_estimators (with class weights)…"
            )
            early_start = time.time()
            bst = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
            )
            best_iter = getattr(bst, "best_iteration", None)
            if best_iter is None:
                best_iter = getattr(bst, "best_ntree_limit", None)
            if best_iter is None:
                best_iter = num_boost_round
            xgb_params["n_estimators"] = int(best_iter)
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda" if args.use_gpu else "cpu"
            logger.info(
                "Early stopping selected n_estimators=%d (took %.1fs)",
                best_iter,
                time.time() - early_start,
            )

            final_model = xgb.XGBClassifier(**xgb_params)
            Xfull_np = np.vstack([X_train_pc, X_val_pc])
            yfull = np.concatenate([y_train, y_val], axis=0)
            final_model.fit(Xfull_np, yfull, sample_weight=sample_weights_full)
            estimator = final_model
        else:
            Xfull_np = np.vstack([X_train_pc, X_val_pc])
            yfull = np.concatenate([y_train, y_val], axis=0)
            try:
                estimator.fit(Xfull_np, yfull, sample_weight=sample_weights_full)
            except TypeError:
                estimator.fit(Xfull_np, yfull)

        Xtest_np = X_test_pc
        y_pred_test = estimator.predict(Xtest_np)
        test_metrics = evaluate(y_test, y_pred_test)

        plot_confusion_matrix_heatmap(
            test_metrics["confusion_matrix"], output_dir, f"test_pc{best_n}"
        )

        joblib.dump(
            {"model": estimator, "pc_columns": pc_cols[:best_n], "model_name": "xgb"},
            output_dir / f"pc_comm_model_pc{best_n}.joblib",
        )

        all_results.append(
            {
                "pcs": best_n,
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1_macro": float(test_metrics["f1"]),
                "f1_weighted": float(test_metrics["weighted_f1"]),
            }
        )

        if best_n == 100:
            results = {
                "task": "pc_community_classification",
                "selected_model": "xgb",
                "selected_num_pcs": best_n,
                "dataset_sizes": {
                    "train": len(X_train),
                    "val": len(X_val),
                    "test": len(X_test),
                },
                "test_metrics": test_metrics,
            }

            results_json_path = output_dir / "results.json"
            with results_json_path.open("w") as f:
                json.dump(convert_numpy(results), f, indent=2)
            logger.info("Saved results.json: %s", results_json_path)

    results_df = pd.DataFrame(all_results).sort_values("pcs")
    results_path = output_dir / "pc_grid_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved PC grid results: %s", results_path)


if __name__ == "__main__":
    main()
