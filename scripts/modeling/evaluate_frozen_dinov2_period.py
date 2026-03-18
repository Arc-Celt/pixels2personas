"""
Evaluate a frozen DINOv2 backbone for period classification as a baseline.

This script:
- loads a period classification dataset arranged as {split}/{period}/*.jpg
- builds a DINOv2 backbone with a randomly initialized classification head
  (backbone is frozen)
- evaluates on the test split and reports accuracy, macro F1, weighted F1
- saves a confusion-matrix heatmap and results JSON.

Example:

python scripts/modeling/evaluate_frozen_dinov2_period.py \
  --data-dir /path/to/classification_period \
  --output-dir /path/to/results \
  --model-name /path/to/dinov2-base
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import Dinov2Config, Dinov2Model
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PERIOD_LABELS = [
    "1917_1980",
    "1981_1985",
    "1986_1990",
    "1991_1995",
    "1996_2000",
    "2001_2005",
    "2006_2010",
    "2011_2015",
    "2016_2020",
    "2021_2025",
]


class PeriodDataset(Dataset):
    """Dataset for period classification from directory structure."""

    def __init__(self, data_dir: Path, split: str = "test", transform=None) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.periods = PERIOD_LABELS
        self.period_to_idx = {p: i for i, p in enumerate(self.periods)}

        self.samples: list[Tuple[str, int]] = []
        for period, label in self.period_to_idx.items():
            period_dir = self.data_dir / split / period
            if not period_dir.exists():
                continue
            for img_path in period_dir.glob("*.jpg"):
                self.samples.append((str(img_path), label))

        logger.info("Loaded %d samples for split='%s'", len(self.samples), split)

        labels = [label for _, label in self.samples]
        if labels:
            class_counts = pd.Series(labels).value_counts().sort_index()
            for period_idx, count in class_counts.items():
                period_name = self.periods[period_idx]
                logger.info("%s - %s: %d", split, period_name, count)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Error loading image %s: %s", img_path, e)
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


class FrozenDINOv2Classifier(nn.Module):
    """Frozen DINOv2 backbone with a small classification head."""

    def __init__(self, model_name: str, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        config = Dinov2Config.from_pretrained(model_name)
        feature_dim = config.hidden_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        logger.info(
            "Created frozen DINOv2 classifier: %d -> %d classes",
            feature_dim,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        features = outputs.last_hidden_state
        cls_features = features[:, 0]
        return self.classifier(cls_features)


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_test_loader(
    data_dir: Path, batch_size: int = 32, num_workers: int = 4
) -> DataLoader:
    dataset = PeriodDataset(data_dir=data_dir, split="test", transform=get_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def evaluate_frozen(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> dict:
    """Evaluate frozen backbone with random head parameters."""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions_arr = np.array(all_predictions)
    all_labels_arr = np.array(all_labels)

    accuracy = accuracy_score(all_labels_arr, all_predictions_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_arr, all_predictions_arr, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        all_labels_arr, all_predictions_arr, average="weighted", zero_division=0
    )[2]
    cm = confusion_matrix(all_labels_arr, all_predictions_arr)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }
    logger.info(
        "Frozen DINOv2 results - Acc: %.4f, F1: %.4f, Weighted F1: %.4f",
        accuracy,
        f1,
        weighted_f1,
    )
    return metrics


def plot_confusion_matrix_heatmap(
    confusion: np.ndarray, output_dir: Path, model_name: str = "frozen_dinov2"
) -> None:
    period_order = [
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

    row_sums = confusion.sum(axis=1, keepdims=True)
    percent_matrix = np.divide(confusion, row_sums, where=row_sums != 0) * 100.0

    annot = np.empty_like(confusion, dtype=object)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            annot[i, j] = f"{percent_matrix[i, j]:.1f}%\n({confusion[i, j]})"

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        percent_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=period_order,
        yticklabels=period_order,
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.xlabel("Predicted Period", fontsize=26, fontweight="bold")
    plt.ylabel("True Period", fontsize=26, fontweight="bold")
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.title("Frozen DINOv2 Confusion Matrix", fontsize=28, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / f"{model_name}_confusion_matrix.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix plot to %s", plot_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen DINOv2 backbone for period classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory of the period dataset (expects {split}/{period}/*.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path or HuggingFace identifier for the DINOv2 model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    torch.manual_seed(args.seed)

    output_dir: Path = args.output_dir / "frozen_dinov2_period"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    test_loader = create_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = FrozenDINOv2Classifier(
        model_name=args.model_name, num_classes=len(PERIOD_LABELS)
    ).to(device)

    metrics = evaluate_frozen(model, test_loader, device)
    plot_confusion_matrix_heatmap(metrics["confusion_matrix"], output_dir)

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    results = {
        "task": "period_classification_frozen_backbone",
        "model_name": args.model_name,
        "args": vars(args),
        "test_metrics": metrics,
        "note": "Backbone is frozen; classifier head has random weights.",
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    logger.info("Frozen DINOv2 evaluation complete.")


if __name__ == "__main__":
    main()
