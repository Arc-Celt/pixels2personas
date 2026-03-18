"""
Fine-tune DINOv2 for anime character personality archetype (community) classification.

Expected dataset structure:
- data_dir/
  - train/{id_label}/*.jpg
  - val/{id_label}/*.jpg
  - test/{id_label}/*.jpg

where `{id_label}` matches the 8 archetype IDs and names:
- 0_Energetic_Extroverts
- 1_Steadfast_Leaders
- 2_Cold-hearted_Villains
- 3_Passionate_Strivers
- 4_Kind_Caregivers
- 5_Justice_Keepers
- 6_Reserved_Introverts
- 7_Arrogant_Tsunderes

Example:

python scripts/modeling/finetune_dinov2_archetype.py \
  --data-dir /path/to/classification_community \
  --output-dir /path/to/results \
  --model-name /path/to/dinov2-base \
  --epochs 50 \
  --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import Dinov2Config, Dinov2Model, get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ARCHETYPE_LABELS = [
    "0_Energetic_Extroverts",
    "1_Steadfast_Leaders",
    "2_Cold-hearted_Villains",
    "3_Passionate_Strivers",
    "4_Kind_Caregivers",
    "5_Justice_Keepers",
    "6_Reserved_Introverts",
    "7_Arrogant_Tsunderes",
]


class ArchetypeDataset(Dataset):
    """Dataset for archetype (community) classification."""

    def __init__(self, data_dir: Path, split: str, transform=None) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.classes = ARCHETYPE_LABELS
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = self.data_dir / split / cls
            if not cls_dir.exists():
                continue
            label = self.class_to_idx[cls]
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((str(img_path), label))

        logger.info("Loaded %d samples for split='%s'", len(self.samples), split)

        labels = [label for _, label in self.samples]
        class_counts = pd.Series(labels).value_counts().sort_index()
        for class_idx, count in class_counts.items():
            logger.info("%s - %s: %d", split, self.classes[class_idx], count)
        missing = set(range(len(self.classes))) - set(class_counts.index)
        for idx in missing:
            logger.warning("%s - %s: 0 (MISSING)", split, self.classes[idx])

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


class DINOv2ArchetypeClassifier(nn.Module):
    """DINOv2 backbone with MLP head for 8-way archetype classification."""

    def __init__(self, model_name: str, num_classes: int = 8) -> None:
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        config = Dinov2Config.from_pretrained(model_name)
        feature_dim = config.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        logger.info(
            "Created archetype classifier: %d -> %d classes", feature_dim, num_classes
        )

    def unfreeze_layers(self, n_layers: int) -> None:
        if not hasattr(self.backbone, "encoder") or not hasattr(
            self.backbone.encoder, "layer"
        ):
            logger.warning(
                "Backbone encoder structure not as expected; skipping unfreeze."
            )
            return
        total_layers = len(self.backbone.encoder.layer)
        for i in range(total_layers - 1, max(-1, total_layers - 1 - n_layers), -1):
            for _, param in self.backbone.encoder.layer[i].named_parameters():
                param.requires_grad = True
            logger.info("Unfroze transformer block index %d", i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        features = outputs.last_hidden_state
        cls_features = features[:, 0]
        return self.classifier(cls_features)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    )
                ],
                p=0.5,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def create_data_loaders(
    data_dir: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    train_t, val_t = get_transforms()
    train_ds = ArchetypeDataset(data_dir, "train", train_t)
    val_ds = ArchetypeDataset(data_dir, "val", val_t)
    test_ds = ArchetypeDataset(data_dir, "test", val_t)

    train_labels = [label for _, label in train_ds.samples]
    num_classes = len(ARCHETYPE_LABELS)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    class_weights = np.power(class_weights, 0.5)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader, test_loader, class_weights


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    accumulation_steps: int = 2,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item() * accumulation_steps:.4f}",
                    "Acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate_baselines(all_labels: list[int], num_classes: int) -> dict:
    from collections import Counter
    import random

    counts = Counter(all_labels)
    majority_class = counts.most_common(1)[0][0]
    majority_preds = [majority_class] * len(all_labels)

    random.seed(42)
    random_preds = [random.randint(0, num_classes - 1) for _ in all_labels]

    maj_acc = accuracy_score(all_labels, majority_preds)
    maj_prec, maj_rec, maj_f1, _ = precision_recall_fscore_support(
        all_labels, majority_preds, average="macro"
    )

    rnd_acc = accuracy_score(all_labels, random_preds)
    rnd_prec, rnd_rec, rnd_f1, _ = precision_recall_fscore_support(
        all_labels, random_preds, average="macro"
    )

    logger.info("Baseline Majority - Acc: %.4f, F1: %.4f", maj_acc, maj_f1)
    logger.info("Baseline Random   - Acc: %.4f, F1: %.4f", rnd_acc, rnd_f1)

    return {
        "majority_vote": {
            "accuracy": maj_acc,
            "precision": maj_prec,
            "recall": maj_rec,
            "f1": maj_f1,
        },
        "random_sampling": {
            "accuracy": rnd_acc,
            "precision": rnd_prec,
            "recall": rnd_rec,
            "f1": rnd_f1,
        },
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"{split_name} Eval"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    weighted_f1 = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )[2]
    cm = confusion_matrix(all_labels, all_preds)

    baselines = evaluate_baselines(all_labels, num_classes=len(ARCHETYPE_LABELS))
    logger.info("%s - Loss: %.4f, Acc: %.4f, F1: %.4f", split_name, avg_loss, acc, f1)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "baselines": baselines,
    }


def plot_confusion_matrix_heatmap(
    confusion_matrix_arr: np.ndarray, output_dir: Path, split_name: str = "test"
) -> None:
    class_labels = [
        "0: Energetic Extroverts",
        "1: Steadfast Leaders",
        "2: Cold-hearted Villains",
        "3: Passionate Strivers",
        "4: Kind Caregivers",
        "5: Justice Keepers",
        "6: Reserved Introverts",
        "7: Arrogant Tsunderes",
    ]

    row_sums = confusion_matrix_arr.sum(axis=1, keepdims=True)
    percent_matrix = (
        np.divide(confusion_matrix_arr, row_sums, where=row_sums != 0) * 100.0
    )

    annot = np.empty_like(confusion_matrix_arr, dtype=object)
    for i in range(confusion_matrix_arr.shape[0]):
        for j in range(confusion_matrix_arr.shape[1]):
            annot[i, j] = f"{percent_matrix[i, j]:.1f}%\n({confusion_matrix_arr[i, j]})"

    AXIS_LABEL_SIZE = 24
    TICK_LABEL_SIZE = 16

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        percent_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"fontsize": 12},
    )

    plt.xlabel("Predicted", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    plt.ylabel("True", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    plt.xticks(rotation=45, fontsize=TICK_LABEL_SIZE, ha="right")
    plt.yticks(rotation=0, fontsize=TICK_LABEL_SIZE)
    plt.tight_layout()

    plot_path = output_dir / f"{split_name}_confusion_matrix_percentage.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved percentage-based confusion matrix: %s", plot_path)


def add_unfrozen_params_to_optimizer(
    model: DINOv2ArchetypeClassifier,
    optimizer: optim.Optimizer,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    existing_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = []
    if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layer"):
        for layer in model.backbone.encoder.layer:
            for p in layer.parameters():
                if p.requires_grad and id(p) not in existing_param_ids:
                    new_params.append(p)
    if new_params:
        optimizer.add_param_group(
            {"params": new_params, "lr": lr, "weight_decay": weight_decay}
        )
    return optimizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv2 for 8-class personality archetype classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory of the archetype dataset (expects train/val/test splits).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoints and metrics will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Path or HuggingFace identifier for the DINOv2 model.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay-backbone", type=float, default=0.05)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze-epochs",
        type=int,
        nargs="*",
        default=[2, 4, 6, 8, 10, 12],
        help="Epochs at which to progressively unfreeze deeper backbone layers.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    torch.manual_seed(args.seed)

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir / "dinov2_archetype"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir, args.batch_size, args.num_workers
    )

    model = DINOv2ArchetypeClassifier(
        model_name=args.model_name, num_classes=len(ARCHETYPE_LABELS)
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("Model compiled for faster training")

    class_weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights_tensor)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = optim.AdamW(
        [
            {
                "params": head_params,
                "lr": args.head_lr,
                "weight_decay": args.weight_decay_head,
            },
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
                "weight_decay": args.weight_decay_backbone,
            },
        ]
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.accumulation_steps))
    num_training_steps = updates_per_epoch * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = GradScaler("cuda")

    best_weighted_f1 = 0.0
    best_metrics = None
    patience = 10
    no_improve = 0
    min_epochs = 10

    logger.info("Starting fine-tuning…")
    for epoch in range(1, args.epochs + 1):
        if epoch in args.unfreeze_epochs:
            layers_to_unfreeze = (epoch // 2) * 2
            model.unfreeze_layers(layers_to_unfreeze)
            optimizer = add_unfrozen_params_to_optimizer(
                model,
                optimizer,
                lr=args.backbone_lr,
                weight_decay=args.weight_decay_backbone,
            )
            logger.info(
                "Progressive unfreezing: unfroze last %d transformer blocks at epoch %d",
                layers_to_unfreeze,
                epoch,
            )

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            scaler,
            accumulation_steps=args.accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, split_name="Val")

        if val_metrics["weighted_f1"] > best_weighted_f1:
            best_weighted_f1 = val_metrics["weighted_f1"]
            best_metrics = val_metrics
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                },
                output_dir / "best_model.pth",
            )
            logger.info(
                "New best Weighted F1: %.4f (epoch %d)",
                best_weighted_f1,
                epoch,
            )
        else:
            no_improve += 1
            logger.info("No improvement for %d/%d epochs", no_improve, patience)

        logger.info(
            "Epoch %d/%d - Train Acc: %.2f%%, Val Weighted F1: %.4f",
            epoch,
            args.epochs,
            train_acc,
            val_metrics["weighted_f1"],
        )

        if epoch >= min_epochs and no_improve >= patience:
            logger.info(
                "Early stopping after %d epochs without improvement (epoch %d)",
                patience,
                epoch,
            )
            break

    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, split_name="Test")

    plot_confusion_matrix_heatmap(
        test_metrics["confusion_matrix"], output_dir, split_name="test"
    )

    results = {
        "task": "archetype_classification",
        "model_name": args.model_name,
        "args": vars(args),
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with (output_dir / "results.json").open("w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    logger.info("Archetype classification fine-tuning completed.")
    logger.info("Best Val Weighted F1: %.4f", best_weighted_f1)
    logger.info("Test Accuracy: %.4f", test_metrics["accuracy"])
    logger.info("Test Weighted F1: %.4f", test_metrics["weighted_f1"])

    logger.info("Baseline comparison:")
    logger.info(
        "DINOv2 - Acc: %.4f, Weighted F1: %.4f",
        test_metrics["accuracy"],
        test_metrics["weighted_f1"],
    )
    logger.info(
        "Majority - Acc: %.4f, F1: %.4f",
        test_metrics["baselines"]["majority_vote"]["accuracy"],
        test_metrics["baselines"]["majority_vote"]["f1"],
    )
    logger.info(
        "Random - Acc: %.4f, F1: %.4f",
        test_metrics["baselines"]["random_sampling"]["accuracy"],
        test_metrics["baselines"]["random_sampling"]["f1"],
    )


if __name__ == "__main__":
    main()
