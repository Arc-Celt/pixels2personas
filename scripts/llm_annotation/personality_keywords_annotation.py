"""
Annotate anime character personalities using a Qwen3-32B-FP8 model via vLLM.

Example usage:
python scripts/llm_annotation/personality_keywords_annotation.py \
  --model-path /path/to/vllm/Qwen3-32B-FP8 \
  --char-bio-dir /path/to/data/anime/char_bio_json \
  --output-dir /path/to/data/anime/extracted \
  --prompt-md scripts/llm_annotation/config/annotation_prompt.md \
  --qwen-config scripts/llm_annotation/config/qwen_params.yaml \
  --output-file char_personality_qwen3_32b_fp8.jsonl \
  --batch-size 10000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_processed_characters(output_file: Path) -> Set[str]:
    """Load the set of already-processed character JSON filenames from the output file."""
    processed: Set[str] = set()
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        char_json = data.get("character_json", "")
                        if char_json:
                            processed.add(char_json)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARN] Error reading existing output file: {e}")
    return processed


def load_character_data(character_file: Path) -> Dict:
    """Load character data from JSON file."""
    with character_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt(prompt_path: Path | None = None) -> str:
    """Load system prompt from a markdown file."""
    if prompt_path is None:
        prompt_path = Path(__file__).with_name("config") / "annotation_prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def load_qwen_config(config_path: Path | None = None) -> Dict:
    """Load LLM and sampling parameters from YAML config."""
    if config_path is None:
        config_path = Path(__file__).with_name("config") / "qwen_params.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_prompt(
    character_file: Path,
    char_data: Dict,
    tokenizer: AutoTokenizer,
    system_prompt: str,
) -> str:
    """Create the full prompt for a character using tokenizer's chat template."""
    user_content = f"""Now analyze this character:

Character name: {char_data.get("name", "")}
Biography: {char_data.get("biography", "")}
Character JSON: {character_file.name}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt_str


def parse_response(response_text: str, character_file: Path) -> Dict:
    """Robustly extract JSON from model output using regex and JSON object finding."""
    try:
        # Clean markdown code blocks
        text = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Find the first '{' and last '}' to extract JSON object
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            result = json.loads(json_str)
        else:
            result = json.loads(text)

        result["character_json"] = character_file.name
        return result

    # If JSON parsing fails, return a structured error response
    except Exception as e:
        print(f"[WARN] Failed to parse JSON for {character_file.name}: {e}")
        print(f"[WARN] Response text: {response_text[:200]}...")
        return {
            "character_name": "",
            "personality_keywords": {"Japanese": [], "English": []},
            "gender": "Unknown",
            "character_json": character_file.name,
            "_parse_error": True,
        }


def compute_chunk_indices(
    total_files: int, chunk_count: int, chunk_id: int
) -> Tuple[int, int]:
    """Compute [start, end) indices for the given chunk configuration."""
    if chunk_count <= 1:
        return 0, total_files

    chunk_size = total_files // chunk_count
    remainder = total_files % chunk_count
    start = chunk_id * chunk_size + min(chunk_id, remainder)
    end = start + chunk_size + (1 if chunk_id < remainder else 0)
    return start, end


def annotate_characters(
    model_path: Path,
    char_bio_dir: Path,
    output_dir: Path,
    output_file_name: str,
    prompt_md: Path,
    qwen_config: Path,
    batch_size: int = 100000,
    chunk_count: int = 1,
    chunk_id: int = 0,
) -> None:
    """
    Annotate character bios with personality information using vLLM.

    This function mirrors the behavior of the original script but with all
    paths and configuration provided via arguments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_output_file = output_dir / output_file_name

    # Load already-processed characters
    processed_chars = load_processed_characters(base_output_file)

    # Get all files and filter out already-processed ones
    all_files = sorted(char_bio_dir.glob("*.json"))
    files = [f for f in all_files if f.name not in processed_chars]

    env_chunk_count = os.getenv("SLURM_ARRAY_TASK_COUNT") or os.getenv("CHUNK_COUNT")
    env_chunk_id = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("CHUNK_ID")
    if env_chunk_count is not None:
        chunk_count = int(env_chunk_count)
    if env_chunk_id is not None:
        chunk_id = int(env_chunk_id)

    if chunk_count > 1:
        start, end = compute_chunk_indices(len(files), chunk_count, chunk_id)
        files = files[start:end]
        print(
            f"Chunking enabled: chunk_id={chunk_id}/{chunk_count-1}, "
            f"processing indices [{start}, {end}) -> {len(files)} files"
        )

    total_all = len(all_files)
    total_remaining = len(files)
    already_processed = len(processed_chars)

    print(f"Found {total_all} character files in: {char_bio_dir}")
    if already_processed > 0:
        print(f"Already processed: {already_processed} characters")
        print(f"Remaining to process: {total_remaining} characters")
    else:
        print(f"Processing all {total_remaining} characters")

    output_file = base_output_file
    if chunk_count > 1:
        output_file = (
            output_dir / f"{output_file_name}.chunk_{chunk_id}_of_{chunk_count}.jsonl"
        )

    print(f"Writing annotations to: {output_file}")
    print(f"Model path: {model_path}")
    print(f"Batch size: {batch_size}")
    print()

    if total_remaining == 0:
        print("All characters have already been processed!")
        return

    # Load shared prompt/config from files
    system_prompt = load_system_prompt(prompt_md)
    qwen_cfg = load_qwen_config(qwen_config)

    # Initialize tokenizer for chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # Initialize vLLM
    print("Initializing vLLM model...")
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=qwen_cfg["llm"]["tensor_parallel_size"],
        gpu_memory_utilization=qwen_cfg["llm"]["gpu_memory_utilization"],
        max_model_len=qwen_cfg["llm"]["max_model_len"],
        max_num_seqs=qwen_cfg["llm"]["max_num_seqs"],
    )

    # Configure sampling parameters for JSON output
    sampling_params = SamplingParams(
        temperature=qwen_cfg["sampling"]["temperature"],
        max_tokens=qwen_cfg["sampling"]["max_tokens"],
        top_p=qwen_cfg["sampling"]["top_p"],
    )

    print("Model loaded. Starting batch processing...\n")
    start_time = time.time()

    # Load all character data and create prompts
    print("Loading character data and creating prompts...")
    prompts: List[str] = []
    file_list: List[Path] = []
    empty_bio_results: List[Dict] = []

    for char_file in tqdm(files, desc="Preparing prompts", unit="char"):
        char_data = load_character_data(char_file)

        # Skip empty biographies
        bio = (char_data.get("biography") or "").strip()
        if not bio:
            empty_bio_results.append(
                {
                    "character_name": char_data.get("name", ""),
                    "personality_keywords": {
                        "Japanese": [],
                        "English": [],
                    },
                    "gender": "Unknown",
                    "character_json": char_file.name,
                }
            )
            continue

        prompt = create_prompt(char_file, char_data, tokenizer, system_prompt)
        prompts.append(prompt)
        file_list.append(char_file)

    print(
        f"Created {len(prompts)} prompts "
        f"({len(empty_bio_results)} skipped with empty biography). "
        "Starting inference...\n"
    )

    processed = 0
    failed = 0

    with output_file.open("a", encoding="utf-8") as out_f:
        for result in empty_bio_results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed += 1

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        pbar = tqdm(
            total=len(prompts) + len(empty_bio_results),
            desc="Annotating characters",
            unit="char",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]",
        )

        pbar.update(len(empty_bio_results))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            batch_files = file_list[start_idx:end_idx]

            outputs = llm.generate(batch_prompts, sampling_params)

            for output, char_file in zip(outputs, batch_files):
                try:
                    response_text = output.outputs[0].text
                    result = parse_response(response_text, char_file)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed += 1
                    pbar.update(1)
                except Exception as e:
                    failed += 1
                    pbar.update(1)
                    tqdm.write(f"[ERROR] Failed on {char_file.name}: {e}")

            if (processed + failed) % 100 == 0:
                out_f.flush()
                elapsed = time.time() - start_time
                rate = (processed + failed) / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    {
                        "success": processed,
                        "failed": failed,
                        "rate": f"{rate:.2f}/s",
                    }
                )

        pbar.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(
        f"Done. Success: {processed}, Failed: {failed}, "
        f"Remaining processed: {total_remaining}"
    )
    if already_processed > 0:
        print(
            f"Previously processed: {already_processed}, "
            f"Total in file: {already_processed + processed}"
        )
    print(f"Total time: {total_time/60:.2f} minutes")
    if processed > 0:
        print(f"Average per successful sample: {total_time/processed:.3f} seconds")
    print(f"Output written to: {output_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for character personality annotation."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate anime character personalities using a Qwen3-32B-FP8 model via vLLM."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the vLLM-compatible model (e.g. /path/to/Qwen3-32B-FP8).",
    )
    parser.add_argument(
        "--char-bio-dir",
        type=Path,
        required=True,
        help="Directory containing character biography JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where annotation JSONL files will be written.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="char_personality_qwen3_32b_fp8.jsonl",
        help="Base output JSONL filename (default: char_personality_qwen3_32b_fp8.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of prompts to send per vLLM batch (default: 10000).",
    )
    parser.add_argument(
        "--chunk-count",
        type=int,
        default=1,
        help="Number of chunks to split the work into (default: 1, i.e., no chunking).",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=0,
        help="Zero-based index of the current chunk (default: 0).",
    )
    parser.add_argument(
        "--prompt-md",
        type=Path,
        default=Path(__file__).with_name("config") / "annotation_prompt.md",
        help="Path to annotation prompt markdown (default: scripts/llm_annotation/config/annotation_prompt.md).",
    )
    parser.add_argument(
        "--qwen-config",
        type=Path,
        default=Path(__file__).with_name("config") / "qwen_params.yaml",
        help="Path to Qwen YAML config (default: scripts/llm_annotation/config/qwen_params.yaml).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.prompt_md.exists():
        raise FileNotFoundError(f"Prompt markdown not found: {args.prompt_md}")
    if not args.qwen_config.exists():
        raise FileNotFoundError(f"Qwen config not found: {args.qwen_config}")

    annotate_characters(
        model_path=args.model_path,
        char_bio_dir=args.char_bio_dir,
        output_dir=args.output_dir,
        output_file_name=args.output_file,
        prompt_md=args.prompt_md,
        qwen_config=args.qwen_config,
        batch_size=args.batch_size,
        chunk_count=args.chunk_count,
        chunk_id=args.chunk_id,
    )


if __name__ == "__main__":
    main()
