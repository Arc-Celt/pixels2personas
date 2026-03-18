"""
Compute inter-annotator agreement (micro metrics) for keyword multi-selection.

Treats one annotator as reference and the other as prediction, and computes:
- global precision / recall / F1 over all pairs
- mean per-instance Jaccard similarity

Inputs:
- Two annotated JSONL files.

Output:
- JSON file containing summary metrics.

Example:

python scripts/utils/compute_inter_annotator_agreement.py \
  --reference-jsonl /path/to/reference_annotated_instances.jsonl \
  --pred-jsonl /path/to/predicted_annotated_instances.jsonl \
  --output-json /path/to/inter_annotator_agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL records from a file."""
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def build_map(records: List[Dict]) -> Dict[str, Set[str]]:
    """Map instance id -> set of valid keywords."""
    m: Dict[str, Set[str]] = {}
    for r in records:
        iid = r.get("id") or r.get("instance_id")
        if not iid:
            continue
        la = r.get("label_annotations", {})
        vk = la.get("valid_keywords", {})
        kws = set(vk.keys()) if isinstance(vk, dict) else set()
        m[str(iid)] = kws
    return m


def compute_metrics(
    reference_map: Dict[str, Set[str]],
    pred_map: Dict[str, Set[str]],
    reference_name: str,
    pred_name: str,
) -> Dict:
    """Compute global micro P/R/F1 and mean per-instance Jaccard."""
    common = set(reference_map.keys()) & set(pred_map.keys())
    all_keywords: Set[str] = set()
    for s in reference_map.values():
        all_keywords.update(s)
    for s in pred_map.values():
        all_keywords.update(s)

    tp = fp = fn = 0
    jaccards: List[float] = []

    for iid in sorted(common):
        g = reference_map.get(iid, set())
        p = pred_map.get(iid, set())

        inter = len(g & p)
        uni = len(g | p)
        j = 1.0 if uni == 0 else inter / uni
        jaccards.append(j)

        for kw in all_keywords:
            in_g = kw in g
            in_p = kw in p
            if in_g and in_p:
                tp += 1
            elif not in_g and in_p:
                fp += 1
            elif in_g and not in_p:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_j = sum(jaccards) / len(jaccards) if jaccards else 0.0

    return {
        "total_common_instances": len(common),
        "total_keywords": len(all_keywords),
        "global": {
            "reference": reference_name,
            "prediction": pred_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
        },
        "mean_jaccard": mean_j,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement for keyword multi-selection."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=False,
        default=Path("annotation_output/keyword_annotation_qwen3_32b_fp8"),
        help="Base directory containing per-annotator JSONL files.",
    )
    parser.add_argument(
        "--reference-id",
        type=str,
        required=True,
        help="Identifier for the reference annotator (subdirectory name).",
    )
    parser.add_argument(
        "--pred-id",
        type=str,
        required=True,
        help="Identifier for the predicted annotator (subdirectory name).",
    )
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        required=True,
        help=(
            "Path to the reference JSONL file. "
            "If set, overrides --base-dir/--reference-id."
        ),
    )
    parser.add_argument(
        "--pred-jsonl",
        type=Path,
        required=True,
        help="Path to the prediction JSONL file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=False,
        default=Path("output/inter_annotator_agreement.json"),
        help="Output JSON file for metrics.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.reference_jsonl is not None:
        ref_path = args.reference_jsonl
    else:
        ref_path = args.base_dir / args.reference_id / "annotated_instances.jsonl"

    if args.pred_jsonl is not None:
        pred_path = args.pred_jsonl
    else:
        pred_path = args.base_dir / args.pred_id / "annotated_instances.jsonl"

    gold_records = load_jsonl(ref_path)
    pred_records = load_jsonl(pred_path)

    gold_map = build_map(gold_records)
    pred_map = build_map(pred_records)

    res = compute_metrics(
        reference_map=gold_map,
        pred_map=pred_map,
        reference_name=args.reference_id,
        pred_name=args.pred_id,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\nInter-annotator agreement metrics:")
    print(
        f"Instances compared: {res['total_common_instances']} | "
        f"Unique keywords: {res['total_keywords']}"
    )
    g = res["global"]
    print(
        f"Precision: {g['precision']:.3f} | "
        f"Recall: {g['recall']:.3f} | "
        f"F1: {g['f1']:.3f}"
    )
    print(f"Mean Jaccard Similarity: {res['mean_jaccard']:.3f}")


if __name__ == "__main__":
    main()
