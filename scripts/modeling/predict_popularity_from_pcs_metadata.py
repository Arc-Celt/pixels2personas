"""
Character popularity prediction using avatar and personality PCs plus metadata.
- builds a per-character feature table by merging:
  - aggregated character info (including favorites, animeography, gender)
  - aggregated anime info (anime_favorites, year)
  - personality PCA features (top 100 PCs)
  - avatar PCA features (top 100 PCs)
- trains multiple regression models (Dummy baseline, Poisson regression, Histogram-based Gradient Boosted Trees)
  on log-transformed favorites, across multiple feature subsets and PC counts,
  and evaluates with MSE, MAE, and mean Poisson deviance.
- saves fitted models, feature-column definitions, raw results, and a
  summary CSV of metrics.

Example:

python scripts/modeling/predict_popularity_from_pcs_metadata.py \
  --char-info-jsonl /path/to/character_info_agg_with_favorites_gender_updated.jsonl \
  --anime-info-csv /path/to/anime_info_agg_with_time.csv \
  --personality-pcs-csv /path/to/personality_top100_pcs.csv \
  --visual-pcs-csv /path/to/visual_top100_pcs.csv \
  --output-dir /path/to/results/popularity
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sns.set_theme(style="whitegrid")


def save_models_and_data(
    fitted_models: Dict[str, object],
    common_df: pd.DataFrame,
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
    feature_columns: Dict[str, List[str]] | None = None,
) -> Path:
    models_dir = output_dir / "fitted_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, model in fitted_models.items():
        model_path = models_dir / f"{key}_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

    data_path = models_dir / "common_df.pkl"
    common_df.to_pickle(data_path)

    results_path = models_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    if feature_columns is not None:
        feat_cols_path = models_dir / "feature_columns.json"
        with feat_cols_path.open("w") as f:
            json.dump(feature_columns, f, indent=2)

    return models_dir


def build_feature_table(
    char_info_jsonl: Path,
    anime_info_csv: Path,
    personality_pcs_csv: Path,
    visual_pcs_csv: Path,
) -> pd.DataFrame:
    char_df = pd.read_json(char_info_jsonl, lines=True)

    def extract_first_anime_json(row):
        if isinstance(row.get("animeography"), list) and row["animeography"]:
            return row["animeography"][0].get("anime_json", np.nan)
        return np.nan

    def extract_role(row):
        if isinstance(row.get("animeography"), list) and row["animeography"]:
            roles = [entry.get("role", "") for entry in row["animeography"]]
            return "Main" if "Main" in roles else "Supporting"
        return "Unknown"

    char_df["anime_json"] = char_df.apply(extract_first_anime_json, axis=1)
    char_df["role"] = char_df.apply(extract_role, axis=1)

    anime_df = pd.read_csv(anime_info_csv)
    personality_pca = pd.read_csv(personality_pcs_csv)
    visual_pca = pd.read_csv(visual_pcs_csv)

    cols_to_drop = ["community", "degree_centrality", "community_degree_centrality"]
    personality_pca = personality_pca.drop(
        columns=[c for c in cols_to_drop if c in personality_pca.columns],
        errors="ignore",
    )

    char_df = char_df.merge(
        anime_df[["anime_json", "anime_favorites"]],
        on="anime_json",
        how="left",
    )

    char_df = char_df.drop_duplicates(subset="character_json", keep="first")
    visual_pca = visual_pca.drop_duplicates(subset="character_json", keep="first")
    personality_pca = personality_pca.drop_duplicates(
        subset="character_json", keep="first"
    )

    visual_pca = visual_pca.rename(
        columns={f"pca_{i}": f"visual_pca_{i}" for i in range(1, 101)}
    )
    personality_pca = personality_pca.rename(
        columns={f"pca_{i}": f"personality_pca_{i}" for i in range(1, 101)}
    )

    main_df = visual_pca.copy()

    meta_cols_to_merge = [
        "character_json",
        "favorites",
        "anime_favorites",
        "year",
        "gender",
        "role",
    ]
    meta_cols_to_merge = [c for c in meta_cols_to_merge if c in char_df.columns]
    main_df = main_df.merge(
        char_df[meta_cols_to_merge], on="character_json", how="left"
    )
    main_df = main_df[main_df["favorites"].notna() & (main_df["favorites"] >= 0)].copy()
    main_df = main_df.merge(personality_pca, on="character_json", how="left")

    personality_cols = [c for c in main_df.columns if c.startswith("personality_pca_")]
    if personality_cols:
        main_df[personality_cols] = main_df[personality_cols].fillna(0.0)

    visual_cols_all = [c for c in main_df.columns if c.startswith("visual_pca_")]
    if visual_cols_all:
        main_df[visual_cols_all] = main_df[visual_cols_all].fillna(0.0)

    if "gender" in main_df.columns:
        gender_series = main_df["gender"].astype(str).str.upper()
        gender_series = gender_series.where(gender_series.isin(["M", "F"]), "Unknown")
        gender_dummies = pd.get_dummies(gender_series, prefix="gender", dtype=float)
        main_df = pd.concat([main_df, gender_dummies], axis=1)

    if "role" in main_df.columns:
        role_series = main_df["role"].astype(str)
        role_series = role_series.where(
            role_series.isin(["Main", "Supporting"]), "Unknown"
        )
        role_dummies = pd.get_dummies(role_series, prefix="role", dtype=float)
        main_df = pd.concat([main_df, role_dummies], axis=1)

    meta_cols = (
        ["anime_favorites", "year"]
        + [c for c in main_df.columns if c.startswith("gender_")]
        + [c for c in main_df.columns if c.startswith("role_")]
    )
    for col in meta_cols:
        if col in main_df.columns:
            main_df[col] = pd.to_numeric(main_df[col], errors="coerce")

    main_df = main_df.dropna(subset=meta_cols)
    return main_df


def run_model(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_runs: int = 20,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, float]:
    results: Dict[str, List[float]] = defaultdict(list)
    for i in tqdm(range(n_runs), desc="Runs"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e12, neginf=1e-12)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mask = y_pred > 0
        mpd = (
            mean_poisson_deviance(y_test[mask], np.clip(y_pred[mask], 1e-12, 1e12))
            if mask.any()
            else np.nan
        )
        results["mse"].append(mse)
        results["mae"].append(mae)
        results["mpd"].append(mpd)
    return {k: float(np.mean(v)) for k, v in results.items()}


def build_feature_sets(common_df: pd.DataFrame, n_pcs: int, meta_cols: List[str]):
    avatar_pcs_n = [
        c
        for c in [f"visual_pca_{i}" for i in range(1, n_pcs + 1)]
        if c in common_df.columns
    ]
    personality_pcs_n = [
        c
        for c in [f"personality_pca_{i}" for i in range(1, n_pcs + 1)]
        if c in common_df.columns
    ]

    return {
        f"avatar_pcs_{n_pcs}": avatar_pcs_n,
        f"personality_pcs_{n_pcs}": personality_pcs_n,
        f"both_pcs_{n_pcs}": avatar_pcs_n + personality_pcs_n,
        f"both_pcs+meta_{n_pcs}": avatar_pcs_n
        + personality_pcs_n
        + [c for c in meta_cols if c in common_df.columns],
        f"avatar_pcs+meta_{n_pcs}": avatar_pcs_n
        + [c for c in meta_cols if c in common_df.columns],
        f"personality_pcs+meta_{n_pcs}": personality_pcs_n
        + [c for c in meta_cols if c in common_df.columns],
        f"meta_only_{n_pcs}": [c for c in meta_cols if c in common_df.columns],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict character popularity from avatar/personality PCs and metadata."
    )
    parser.add_argument(
        "--char-info-jsonl",
        type=Path,
        required=True,
        help="JSONL file with aggregated character info (must include 'favorites').",
    )
    parser.add_argument(
        "--anime-info-csv",
        type=Path,
        required=True,
        help="CSV with aggregated anime information.",
    )
    parser.add_argument(
        "--personality-pcs-csv",
        type=Path,
        required=True,
        help="CSV with personality PCA features (top PCs).",
    )
    parser.add_argument(
        "--visual-pcs-csv",
        type=Path,
        required=True,
        help="CSV with avatar/visual PCA features (top PCs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store models, results, and plots.",
    )
    parser.add_argument(
        "--train-new-models",
        action="store_true",
        help="Train new models instead of only using existing ones.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Number of random train/test splits per configuration (default: 20).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test size fraction for each run (default: 0.1).",
    )
    parser.add_argument(
        "--pc-counts",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="List of PC counts to evaluate (default: 20 50 100).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    common_df = build_feature_table(
        char_info_jsonl=args.char_info_jsonl,
        anime_info_csv=args.anime_info_csv,
        personality_pcs_csv=args.personality_pcs_csv,
        visual_pcs_csv=args.visual_pcs_csv,
    )

    meta_cols = (
        ["anime_favorites", "year"]
        + [c for c in common_df.columns if c.startswith("gender_")]
        + [c for c in common_df.columns if c.startswith("role_")]
    )

    y = np.log1p(common_df["favorites"].values)

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "PoissonRegressor": PoissonRegressor(alpha=1e-2, max_iter=300),
        "HGBR_Poisson": HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, object] = {}
    feature_columns: Dict[str, List[str]] = {}

    if args.train_new_models:
        for n_pcs in args.pc_counts:
            feature_sets = build_feature_sets(common_df, n_pcs, meta_cols)

            for feat_name, feat_cols in feature_sets.items():
                if not feat_cols:
                    continue
                X = common_df[feat_cols].values
                if X.shape[1] == 0 or X.shape[0] == 0:
                    continue
                for model_name, base_model in models.items():
                    key = f"{feat_name}__{model_name}"
                    if model_name == "PoissonRegressor":
                        estimator = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "regressor",
                                    PoissonRegressor(alpha=1e-2, max_iter=300),
                                ),
                            ]
                        )
                    else:
                        estimator = clone(base_model)

                    cfg_desc = f"{key} (X shape={X.shape})"
                    print(f"Running {cfg_desc}…")
                    results[key] = run_model(
                        X,
                        y,
                        estimator,
                        n_runs=args.n_runs,
                        test_size=args.test_size,
                    )

                    estimator.fit(X, y)
                    fitted_models[key] = estimator
                    feature_columns[key] = feat_cols

        save_models_and_data(
            fitted_models=fitted_models,
            common_df=common_df,
            results=results,
            output_dir=output_dir,
            feature_columns=feature_columns,
        )

    if results:
        rows = []
        for key, metrics in results.items():
            if "__" in key:
                feature_set, model_name = key.split("__", 1)
            else:
                feature_set, model_name = key, ""
            row = {"feature_set": feature_set, "model": model_name}
            row.update(metrics)
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
