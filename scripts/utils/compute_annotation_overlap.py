"""
Compute overlap metrics between human personality keyword annotations
and model-predicted keywords.
- loads original model outputs (with `id` and `model_keywords`)
- loads human annotation JSONL (with `instance_id` or `id`, `user_id`,
  and `label_annotations.valid_keywords`)
- computes per-instance precision/recall/F1/Jaccard between human and model
  keyword sets
- aggregates metrics per user and overall
- writes a JSON report with per-user and overall statistics.

Example:

python scripts/utils/compute_annotation_overlap.py \
  --original-jsonl /path/to/keyword_annotation_qwen3_32b_fp8.jsonl \
  --annotated-jsonl /path/to/annotated_instances.jsonl \
  --output-json /path/to/annotation_overlap_metrics.json
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load a JSONL file."""
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_model_keywords(instance: Dict) -> Set[str]:
    """Extract model keywords from original data instance."""
    return set(instance.get("model_keywords", []))


def extract_annotated_keywords(instance: Dict) -> Set[str]:
    """Extract annotated keywords from annotation instance, excluding sentinel 'None of the above'."""
    label_annotations = instance.get("label_annotations", {})
    valid_keywords = label_annotations.get("valid_keywords", {})

    keywords: Set[str] = set()
    for keyword in valid_keywords.keys():
        if keyword != "None of the above":
            keywords.add(keyword)
    return keywords


def compute_metrics(predicted: Set[str], actual: Set[str]) -> Dict[str, float]:
    """Compute precision, recall, F1, and Jaccard similarity."""
    if not predicted and not actual:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaccard": 1.0,
        }

    intersection = predicted & actual
    union = predicted | actual

    precision = len(intersection) / len(predicted) if predicted else 0.0
    recall = len(intersection) / len(actual) if actual else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def compute_aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute aggregate mean/std statistics across all instances."""
    if not all_metrics:
        return {}

    aggregate: Dict[str, List[float]] = defaultdict(list)
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregate[key].append(float(value))

    result: Dict[str, float] = {}
    for key, values in aggregate.items():
        if not values:
            continue
        mean_val = sum(values) / len(values)
        var = sum((x - mean_val) ** 2 for x in values) / len(values)
        result[f"{key}_mean"] = mean_val
        result[f"{key}_std"] = var**0.5

    return result


def compute_user_metrics(
    user_annotations: List[Dict], original_lookup: Dict[str, Dict]
) -> Dict:
    """Compute overlap metrics for a single annotator."""
    user_metrics: List[Dict[str, float]] = []
    per_instance_results: List[Dict] = []
    total_instances = 0

    for annotated_instance in user_annotations:
        instance_id = annotated_instance.get("instance_id") or annotated_instance.get(
            "id"
        )
        if not instance_id:
            continue
        if instance_id not in original_lookup:
            continue

        original_instance = original_lookup[instance_id]

        model_keywords = extract_model_keywords(original_instance)
        annotated_keywords = extract_annotated_keywords(annotated_instance)

        metrics = compute_metrics(predicted=annotated_keywords, actual=model_keywords)

        total_instances += 1
        user_metrics.append(metrics)

        per_instance_results.append(
            {
                "instance_id": instance_id,
                "model_keywords": sorted(list(model_keywords)),
                "annotated_keywords": sorted(list(annotated_keywords)),
                "metrics": metrics,
            }
        )

    aggregate_metrics = compute_aggregate_metrics(user_metrics)

    return {
        "total_instances": total_instances,
        "aggregate_metrics": aggregate_metrics,
        "per_instance": per_instance_results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute overlap metrics between human annotations and model keywords."
    )
    parser.add_argument(
        "--original-jsonl",
        type=Path,
        required=True,
        help="JSONL file with original model outputs (must contain 'id' and 'model_keywords').",
    )
    parser.add_argument(
        "--annotated-jsonl",
        type=Path,
        required=True,
        help="JSONL file with human annotations (must contain 'user_id' and label_annotations).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write JSON report with per-user and overall metrics.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    original_data = load_jsonl(args.original_jsonl)
    original_lookup = {str(item["id"]): item for item in original_data}

    annotated_data = load_jsonl(args.annotated_jsonl)

    user_to_annotations: Dict[str, List[Dict]] = defaultdict(list)
    for annotated_instance in annotated_data:
        user_id = annotated_instance.get("user_id")
        if user_id:
            user_to_annotations[str(user_id)].append(annotated_instance)

    user_results: Dict[str, Dict] = {}
    for user_id in sorted(user_to_annotations.keys()):
        user_annotations = user_to_annotations[user_id]
        user_result = compute_user_metrics(user_annotations, original_lookup)
        user_results[user_id] = user_result

        agg = user_result["aggregate_metrics"]
        print(
            f"{user_id}: "
            f"Precision={agg.get('precision_mean', 0.0):.3f}, "
            f"Recall={agg.get('recall_mean', 0.0):.3f}, "
            f"F1={agg.get('f1_mean', 0.0):.3f}, "
            f"Jaccard={agg.get('jaccard_mean', 0.0):.3f}"
        )

    all_instance_metrics: List[Dict[str, float]] = []
    for user_result in user_results.values():
        for inst_result in user_result["per_instance"]:
            if "metrics" in inst_result:
                all_instance_metrics.append(inst_result["metrics"])
    overall_metrics = compute_aggregate_metrics(all_instance_metrics)
    total_all_instances = sum(r["total_instances"] for r in user_results.values())

    print(
        "\nOverall: "
        f"Precision={overall_metrics.get('precision_mean', 0.0):.3f}, "
        f"Recall={overall_metrics.get('recall_mean', 0.0):.3f}, "
        f"F1={overall_metrics.get('f1_mean', 0.0):.3f}, "
        f"Jaccard={overall_metrics.get('jaccard_mean', 0.0):.3f}"
    )

    output_data = {
        "overall": {
            "total_instances": total_all_instances,
            "aggregate_metrics": overall_metrics,
        },
        "per_user": user_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
