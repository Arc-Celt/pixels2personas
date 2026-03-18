"""
Replot confusion matrices from JSON result files with consistent styling.

Input:
- Results JSON containing `test_metrics.confusion_matrix` (period classifier and/or PC-community classifier)

Output:
- avatar_confusion_matrix.pdf
- pc_community_confusion_matrix.pdf

Example:

python scripts/plotting/confusion_matrices.py \
  --period-results-json /path/to/dinov2_period/results.json \
  --pc-community-results-json /path/to/pc_community_new/results.json \
  --output-dir /path/to/output_dir
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PERIOD_ORDER: List[str] = [
    "1917-1980",
    "1981-1985",
    "1986-1990",
    "1991-1995",
    "1996-2000",
    "2001-2005",
    "2006-2010",
    "2011-2015",
    "2016-2020",
    "2021-2025",
]

COMMUNITY_LABELS: List[str] = [
    "0: Energetic Extroverts",
    "1: Steadfast Leaders",
    "2: Cold-hearted Antagonists",
    "3: Passionate Strivers",
    "4: Kind Caregivers",
    "5: Justice Keepers",
    "6: Reserved Introverts",
    "7: Arrogant Tsunderes",
]


def _to_two_lines(name: str) -> str:
    base = name.split(":", 1)[1].strip() if ":" in name else name.strip()
    words = base.split()
    if len(words) <= 2:
        return "\n".join(words)
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def load_results(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


def plot_confusion_matrix_heatmap(
    confusion_matrix: Sequence[Sequence[int]],
    output_path: Path,
    xlabels: List[str],
    ylabels: List[str],
    xlabel: str,
    ylabel: str,
    axis_label_size: int = 28,
    tick_label_size: int = 20,
    fontweight: str = "bold",
) -> None:
    sns.set_theme(style="whitegrid")

    cm = np.array(confusion_matrix)

    # Match the paper plotting: Predicted on y, True on x
    cm = cm.T

    row_sums = cm.sum(axis=1, keepdims=True)
    percent_matrix = np.divide(cm, row_sums, where=row_sums != 0) * 100.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{percent_matrix[i, j]:.1f}%\n({cm[i, j]})"

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        percent_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=xlabels,
        yticklabels=ylabels,
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"fontsize": 15},
    )
    plt.xlabel(xlabel, fontsize=axis_label_size, fontweight=fontweight)
    plt.ylabel(ylabel, fontsize=axis_label_size, fontweight=fontweight)
    plt.xticks(rotation=45, fontsize=tick_label_size)
    plt.yticks(rotation=0, fontsize=tick_label_size)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replot confusion matrices for paper figures."
    )
    p.add_argument(
        "--period-results-json", type=Path, help="Results JSON for period classifier."
    )
    p.add_argument(
        "--pc-community-results-json",
        type=Path,
        help="Results JSON for PC-community classifier.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.period_results_json is None and args.pc_community_results_json is None:
        raise SystemExit(
            "Provide at least one of --period-results-json or --pc-community-results-json."
        )

    if args.period_results_json is not None:
        results = load_results(args.period_results_json)
        cm = results["test_metrics"]["confusion_matrix"]
        out_path = out_dir / "avatar_confusion_matrix.pdf"
        plot_confusion_matrix_heatmap(
            cm,
            output_path=out_path,
            xlabels=PERIOD_ORDER,
            ylabels=PERIOD_ORDER,
            xlabel="True Period",
            ylabel="Predicted Period",
        )
        print(f"Saved: {out_path}")

    if args.pc_community_results_json is not None:
        results = load_results(args.pc_community_results_json)
        cm = results["test_metrics"]["confusion_matrix"]
        compact_labels = [_to_two_lines(lbl) for lbl in COMMUNITY_LABELS]
        out_path = out_dir / "pc_community_confusion_matrix.pdf"
        plot_confusion_matrix_heatmap(
            cm,
            output_path=out_path,
            xlabels=compact_labels,
            ylabels=compact_labels,
            xlabel="True Archetype",
            ylabel="Predicted Archetype",
            axis_label_size=28,
            tick_label_size=18,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
