"""
SHAP-based feature importance analysis for character popularity regression.

This script:
- loads previously trained popularity models and data (from the
  popularity regression pipeline)
- selects the best HistGradientBoostingRegressor configuration
- computes SHAP values for interpretability
- plots:
  - SHAP summary beeswarm
  - SHAP importance bar charts
- saves SHAP importances to CSV.

Example:

python scripts/modeling/feature_importance_shap_popularity.py \
  --models-dir /path/to/results/popularity/fitted_models \
  --model-key both_pcs+meta_50__HGBR_Poisson
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

sns.set_theme(style="whitegrid")

AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 18
X_LABEL_SIZE = 20
LEGEND_FONT_SIZE = 20
FONTWEIGHT = "bold"


def load_models_and_data(
    models_dir: Path,
) -> Tuple[pd.DataFrame | None, Dict, Dict | None, Dict | None]:
    """Load previously saved models, data, and results."""
    models_dir = Path(models_dir)

    data_path = models_dir / "common_df.pkl"
    if data_path.exists():
        common_df = pd.read_pickle(data_path)
    else:
        print(f"Data file not found: {data_path}")
        return None, None, None, None

    results_path = models_dir / "results.json"
    if results_path.exists():
        with results_path.open("r") as f:
            results = json.load(f)
    else:
        results = None

    models: Dict[str, object] = {}
    for model_file in models_dir.glob("*_model.pkl"):
        key = model_file.stem.replace("_model", "")
        with model_file.open("rb") as f:
            models[key] = pickle.load(f)

    feat_cols_path = models_dir / "feature_columns.json"
    feature_columns = None
    if feat_cols_path.exists():
        with feat_cols_path.open("r") as f:
            feature_columns = json.load(f)

    return common_df, models, results, feature_columns


def categorize_feature(feature_name: str) -> str:
    if feature_name.startswith("visual_pca_"):
        return "Avatar PCs"
    if feature_name.startswith("personality_pca_"):
        return "Personality PCs"
    if feature_name in ["anime_favorites", "year"] or feature_name.startswith(
        ("gender_", "role_")
    ):
        return "Metadata"
    return "Metadata"


def analyze_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    max_samples: int | None = None,
) -> Tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    print("Calculating SHAP values…")

    is_sampled = max_samples is not None and len(X) > max_samples
    if is_sampled:
        sample_idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        sample_idx = None

    cache_dir = output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    size_tag = f"sampled_{max_samples}" if is_sampled else f"full_{len(X_sample)}"
    cache_path = cache_dir / f"shap_values_{size_tag}.npz"

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            shap_values_arr = data["shap_values"]
            X_sample_cached = data["X_sample"]
            if X_sample_cached.shape == X_sample.shape:
                shap_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "shap_importance": np.abs(shap_values_arr).mean(axis=0),
                        "feature_type": [categorize_feature(f) for f in feature_names],
                    }
                ).sort_values("shap_importance", ascending=False)
                return shap_df, shap_values_arr, X_sample
        except Exception:
            pass

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap_values_arr = shap_values.values

        try:
            np.savez_compressed(
                cache_path, shap_values=shap_values_arr, X_sample=X_sample
            )
        except Exception:
            pass

        mean_shap = np.abs(shap_values_arr).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": mean_shap,
                "feature_type": [categorize_feature(f) for f in feature_names],
            }
        ).sort_values("shap_importance", ascending=False)

        return shap_df, shap_values_arr, X_sample

    except Exception as e:
        print(f"SHAP calculation failed: {e}")
        return None, None, None


def to_display(names: List[str]) -> List[str]:
    display_map = {
        "anime_favorites": "Anime favorites",
        "year": "Release year",
    }
    pretty: List[str] = []
    for n in names:
        if n in display_map:
            pretty.append(display_map[n])
            continue
        if n.startswith("visual_pca_"):
            idx = n.split("_")[-1]
            pretty.append(f"Avatar PC {idx}")
            continue
        if n.startswith("personality_pca_"):
            idx = n.split("_")[-1]
            pretty.append(f"Personality PC {idx}")
            continue
        if n.startswith("gender_"):
            gval = n[len("gender_") :]
            label = {"M": "Male", "F": "Female"}.get(
                gval.upper(), gval.replace("_", " ").title() or "Gender"
            )
            pretty.append(f"Gender: {label}")
            continue
        if n.startswith("role_"):
            role_name = n.replace("role_", "").replace("_", " ").title()
            pretty.append(f"Role: {role_name}")
            continue
        pretty.append(n)
    return pretty


def plot_shap_summary(
    shap_values: np.ndarray | None,
    X_sample: np.ndarray | None,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    if shap_values is None or X_sample is None:
        return

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names,
        max_display=10,
        show=False,
    )
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=X_LABEL_SIZE,
        fontweight=FONTWEIGHT,
    )
    axes = plt.gcf().axes
    if len(axes) > 1:
        cax = axes[-1]
        cax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_raw.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    clean_feature_names = to_display(feature_names)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        clean_feature_names,
        max_display=10,
        show=False,
    )
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel(
        "SHAP value (impact on model output)",
        fontsize=X_LABEL_SIZE,
        fontweight=FONTWEIGHT,
    )
    axes = plt.gcf().axes
    if len(axes) > 1:
        cax = axes[-1]
        cax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_clean.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_importance(
    shap_df: pd.DataFrame, output_dir: Path, top_n: int = 10
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = shap_df.sort_values("shap_importance", ascending=False).head(top_n)
    top_features = top_features.iloc[::-1]

    bars = ax.barh(range(len(top_features)), top_features["shap_importance"])

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"], fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel(
        "SHAP Importance (mean |SHAP value|)",
        fontsize=X_LABEL_SIZE,
        fontweight=FONTWEIGHT,
    )
    ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x", alpha=0.3)

    palette = sns.color_palette("colorblind", n_colors=3)
    category_to_color = {
        "Metadata": palette[0],
        "Avatar PCs": palette[1],
        "Personality PCs": palette[2],
    }

    for bar, feature_type in zip(bars, top_features["feature_type"]):
        bar.set_color(category_to_color.get(feature_type, palette[0]))

    ordered_types = ["Metadata", "Avatar PCs", "Personality PCs"]
    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=category_to_color[t], edgecolor="none", label=t
        )
        for t in ordered_types
    ]
    ax.legend(
        handles=legend_handles,
        title="Feature type",
        loc="lower right",
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_raw.pdf", dpi=300, bbox_inches="tight")

    clean_labels = to_display(top_features["feature"].tolist())
    ax.set_yticklabels(clean_labels, fontsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_clean.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHAP feature-importance analysis for popularity regression models."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing fitted_models (common_df.pkl, *_model.pkl, feature_columns.json).",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="both_pcs+meta_50__HGBR_Poisson",
        help="Key of the model to explain (matches *_model.pkl prefix).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use for SHAP (None uses all).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    common_df, models, results, feature_columns = load_models_and_data(args.models_dir)
    if common_df is None or feature_columns is None:
        print(
            "Required data or feature mapping not found; run the popularity regression first."
        )
        return

    if args.model_key not in models:
        print(
            f"Model key '{args.model_key}' not found. Available keys: {list(models.keys())[:10]}…"
        )
        return

    model = models[args.model_key]
    feature_names = feature_columns.get(args.model_key, [])
    if not feature_names:
        print(f"No feature names found for model key '{args.model_key}'")
        return

    X = common_df[feature_names].values
    print(f"Analyzing SHAP values for model '{args.model_key}'")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(X)}")

    plot_output_dir = args.models_dir / "feature_importance_analysis"
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    shap_df, shap_values, X_sample = analyze_shap_values(
        model, X, feature_names, plot_output_dir, max_samples=args.max_samples
    )
    if shap_df is not None:
        print("Top 10 most important features (SHAP):")
        for _, row in shap_df.head(10).iterrows():
            print(
                f"  {row['feature']}: {row['shap_importance']:.4f} ({row['feature_type']})"
            )

        plot_shap_summary(shap_values, X_sample, feature_names, plot_output_dir)
        plot_shap_importance(shap_df, plot_output_dir)
        shap_df.to_csv(plot_output_dir / "shap_importance.csv", index=False)

    print(f"All SHAP results saved to: {plot_output_dir}")


if __name__ == "__main__":
    main()
