"""
Utilities for extracting personality keyword embeddings using Qwen3-Embedding-0.6B model.

Example usage:
python scripts/embedding/extract_personality_embeddings.py \
  --model-path /path/to/embedding/model/Qwen3-Embedding-0.6B \
  --input-path /path/to/input/jsonl/file.jsonl \
  --output-path /path/to/output/jsonl/file.jsonl \
  --batch-size 32
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings using the last token, handling left padding correctly."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def load_qwen_embedding_model(model_path: str) -> Tuple[AutoModel, AutoTokenizer]:
    """Load the Qwen3 embedding model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(
        f"Loading model from {model_path} (device: {device})...", file=sys.stderr
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def extract_personality_embeddings(
    input_path: Path,
    output_path: Path,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
) -> Tuple[int, int]:
    """
    Process a JSONL file and generate embeddings for entries with English keywords.

    The input JSONL is expected to contain one JSON object per line, where each
    object has a `personality_keywords` field with an `English` list.
    """
    processed_count = 0
    skipped_count = 0
    results = []
    texts = []

    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with input_path.open("r", encoding="utf-8") as f:
        pbar = tqdm(
            f,
            total=total_lines,
            desc=f"Reading {input_path.name}",
            file=sys.stderr,
            unit="lines",
        )
        for line_num, line in enumerate(pbar, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                kw_obj = data.get("personality_keywords")
                if isinstance(kw_obj, dict):
                    english_keywords = kw_obj.get("English") or kw_obj.get(
                        "english", []
                    )
                elif isinstance(kw_obj, list):
                    english_keywords = kw_obj
                else:
                    english_keywords = []

                english_keywords = [
                    str(k).strip() for k in english_keywords if str(k).strip()
                ]

                if not english_keywords:
                    skipped_count += 1
                    continue

                text = ", ".join(english_keywords)

                results.append(
                    {
                        "character_json": data["character_json"],
                        "character_name": data["character_name"],
                    }
                )
                texts.append(text)
                processed_count += 1
                pbar.set_postfix(
                    {"processed": processed_count, "skipped": skipped_count}
                )

            except json.JSONDecodeError as e:
                tqdm.write(
                    f"Warning: JSON decode error on line {line_num}: {e}",
                    file=sys.stderr,
                )
            except Exception as e:
                tqdm.write(
                    f"Warning: Error processing line {line_num}: {e}",
                    file=sys.stderr,
                )
        pbar.close()

    embeddings_list = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        pbar = tqdm(
            range(0, len(texts), batch_size),
            total=n_batches,
            desc="Generating embeddings",
            file=sys.stderr,
            unit="batch",
        )
        for i in pbar:
            batch_texts = texts[i : i + batch_size]

            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict.to(model.device)

            outputs = model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings_list.append(embeddings)
            current_count = min((len(embeddings_list) * batch_size), len(texts))
            pbar.set_postfix({"embeddings": current_count})
        pbar.close()

    if embeddings_list:
        embeddings_tensor = torch.cat(embeddings_list, dim=0)
        normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)

        for i, result in enumerate(results):
            result["embedding"] = normalized_embeddings[i].cpu().tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tqdm.write(
        f"{input_path.name}: {processed_count} processed, {skipped_count} skipped",
        file=sys.stderr,
    )
    return processed_count, skipped_count


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract personality keyword embeddings using a Qwen3 embedding model.\n\n"
            "The input JSONL is expected to contain one JSON object per line with a "
            "'personality_keywords' field under the 'English' key."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Path or identifier for the Qwen3 embedding model "
            "(e.g. 'Qwen/Qwen3-Embedding-0.6B' or a local directory)."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the input JSONL file with personality annotations.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the output JSONL file to write embeddings to.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32).",
    )
    return parser


def main(argv=None) -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    model, tokenizer = load_qwen_embedding_model(args.model_path)

    tqdm.write(f"Starting to process {args.input_path}...", file=sys.stderr)
    processed, skipped = extract_personality_embeddings(
        input_path=args.input_path,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    tqdm.write(f"\nSummary: {processed} processed, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
