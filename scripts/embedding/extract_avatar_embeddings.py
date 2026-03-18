"""
Utilities for extracting DINOv2 image embeddings.

Example:

python scripts/embedding/extract_avatar_embeddings.py \
  --model-name /path/to/dinov2-base \
  --dataset-csv /path/to/character_year_labeled_dataset.csv \
  --output-jsonl /path/to/char_visual_embeddings.jsonl \
  --file-type avatar
"""

from __future__ import annotations

import argparse
import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def load_model(model_name: str) -> Tuple[object, object, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Loading DINOv2 model from {model_name} (device: {device})...")
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor, device


def load_image_from_row(row: pd.Series, file_path_col: str) -> Optional[Dict]:
    try:
        raw = row.get(file_path_col)
        if not isinstance(raw, str) or not raw:
            return None
        img_path = Path(raw)
        if not img_path.exists():
            return None
        img = Image.open(img_path).convert("RGB")
        return {"image": img, "meta": row.to_dict()}
    except Exception:
        return None


@torch.no_grad()
def generate_embeddings(
    batch: List[Dict],
    model,
    processor,
    device: torch.device,
    pool: str = "mean",
) -> List[Dict]:
    images = [item["image"] for item in batch]
    metas = [item["meta"] for item in batch]

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)

    hidden = outputs.last_hidden_state  # [B, seq, dim]
    if pool == "mean":
        feats = hidden.mean(dim=1)
    elif pool == "cls":
        feats = hidden[:, 0]
    else:
        raise ValueError(f"Unknown pooling mode: {pool}")

    feats = feats.detach().cpu().tolist()

    out: List[Dict] = []
    for meta, emb in zip(metas, feats):
        meta = dict(meta)
        meta["embedding"] = emb
        out.append(meta)
    return out


def iter_chunks(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def extract_embeddings_to_jsonl(
    df: pd.DataFrame,
    model,
    processor,
    device: torch.device,
    output_jsonl: Path,
    *,
    file_path_col: str,
    batch_size: int,
    load_chunk_size: int,
    max_workers: int,
    pool: str,
    flush_every: int = 1000,
) -> Tuple[int, int]:
    processed = 0
    skipped = 0
    buffer: List[Dict] = []

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for start, end in tqdm(
            list(iter_chunks(len(df), load_chunk_size)),
            desc="Embedding chunks",
            unit="chunk",
        ):
            chunk = df.iloc[start:end]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(load_image_from_row, row, file_path_col)
                    for _, row in chunk.iterrows()
                ]
                for fut in as_completed(futures):
                    item = fut.result()
                    if item is None:
                        skipped += 1
                        continue
                    buffer.append(item)

                    if len(buffer) >= batch_size:
                        batch_out = generate_embeddings(
                            buffer, model, processor, device, pool=pool
                        )
                        for rec in batch_out:
                            f_out.write(json.dumps(rec) + "\n")
                        processed += len(batch_out)
                        buffer.clear()

                        if processed % flush_every == 0:
                            f_out.flush()

                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

        if buffer:
            batch_out = generate_embeddings(buffer, model, processor, device, pool=pool)
            for rec in batch_out:
                f_out.write(json.dumps(rec) + "\n")
            processed += len(batch_out)
            buffer.clear()

    return processed, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings for images listed in a CSV."
    )
    p.add_argument(
        "--model-name", type=str, required=True, help="Path or HF id for DINOv2 model."
    )
    p.add_argument(
        "--dataset-csv",
        type=Path,
        required=True,
        help="CSV containing image paths and metadata.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL to write embeddings to.",
    )
    p.add_argument(
        "--file-path-col",
        type=str,
        default="file_path",
        help="CSV column containing image file paths.",
    )
    p.add_argument(
        "--file-type-col",
        type=str,
        default="file_type",
        help="Optional column for file type filtering.",
    )
    p.add_argument(
        "--file-type",
        type=str,
        default=None,
        help="If set, filter dataset rows where file_type_col == this value.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--load-chunk-size", type=int, default=200)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--pool",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Token pooling mode for image features.",
    )
    p.add_argument(
        "--keep-cols",
        type=str,
        nargs="*",
        default=None,
        help="If provided, keep only these metadata columns plus embedding.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    df = pd.read_csv(args.dataset_csv)
    if args.file_type is not None and args.file_type_col in df.columns:
        df = df[df[args.file_type_col].astype(str) == str(args.file_type)].copy()

    if args.file_path_col not in df.columns:
        raise SystemExit(f"Missing file path column: {args.file_path_col}")

    if args.keep_cols is not None:
        keep = [c for c in args.keep_cols if c in df.columns]
        for key_col in ["character_json", "character_name", args.file_type_col]:
            if key_col in df.columns and key_col not in keep:
                keep.append(key_col)
        if args.file_path_col not in keep:
            keep.append(args.file_path_col)
        df = df[keep].copy()

    tqdm.write(f"Rows to process: {len(df):,}")
    model, processor, device = load_model(args.model_name)

    processed, skipped = extract_embeddings_to_jsonl(
        df=df,
        model=model,
        processor=processor,
        device=device,
        output_jsonl=args.output_jsonl,
        file_path_col=args.file_path_col,
        batch_size=int(args.batch_size),
        load_chunk_size=int(args.load_chunk_size),
        max_workers=int(args.max_workers),
        pool=str(args.pool),
    )

    tqdm.write(f"Done: {processed:,} embedded, {skipped:,} skipped -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
