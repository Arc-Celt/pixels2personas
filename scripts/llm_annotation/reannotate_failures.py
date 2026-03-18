"""
Re-annotate anime character personalities for entries that previously failed
JSON parsing, using a Qwen3-32B-FP8 model via vLLM.

Example usage:

python scripts/llm_annotation/reannotate_failures.py \
  --model-path /path/to/Qwen3-32B-FP8 \
  --char-bio-dir /path/to/data/anime/char_bio_json \
  --input-annotations /path/to/input/jsonl/file.jsonl \
  --prompt-md scripts/llm_annotation/config/annotation_prompt.md \
  --qwen-config scripts/llm_annotation/config/qwen_params.yaml \
  --batch-size 100
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_character_data(character_file: Path) -> Dict:
    """Load character data from JSON file."""
    with character_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt(prompt_path: Path | None = None) -> str:
    """Load system prompt from markdown config file."""
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


def find_error_entries(annotations_file: Path) -> list[str]:
    """Find `character_json` filenames for entries with parse errors."""
    error_chars: list[str] = []
    if not annotations_file.exists():
        print(f"[WARN] Annotations file not found: {annotations_file}")
        return error_chars

    print(f"Scanning {annotations_file} for parse errors...")
    with annotations_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("_parse_error") is True:
                    char_json = data.get("character_json", "")
                    if char_json:
                        error_chars.append(char_json)
            except json.JSONDecodeError:
                continue

    print(f"Found {len(error_chars)} entries with parse errors")
    return error_chars


def reannotate_errors(
    model_path: Path,
    char_bio_dir: Path,
    annotations_file: Path,
    prompt_md: Path,
    qwen_config: Path,
    batch_size: int = 100,
) -> None:
    """
    Re-annotate characters that had parse errors in an existing annotations file.

    This will:
    - Scan the annotations file for entries where `_parse_error` is True.
    - Re-run the model on the corresponding character bios.
    - Update the in-memory annotation dict and overwrite the annotations file
      with corrected entries.
    """
    # Find error entries
    error_chars = find_error_entries(annotations_file)

    if not error_chars:
        print("No parse errors found. Nothing to reannotate.")
        return

    # Load all existing entries into memory
    print(f"Loading existing entries from {annotations_file}...")
    existing_entries: Dict[str, Dict] = {}
    if annotations_file.exists():
        with annotations_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    char_json = data.get("character_json", "")
                    if char_json:
                        existing_entries[char_json] = data
                except json.JSONDecodeError:
                    continue

    # Find character files for error entries
    print(f"Finding character files for {len(error_chars)} error entries...")
    char_files: list[Path] = []
    for char_json in error_chars:
        char_file = char_bio_dir / char_json
        if char_file.exists():
            char_files.append(char_file)
        else:
            print(f"[WARN] Character file not found: {char_file}")

    if not char_files:
        print("No character files found for error entries.")
        return

    print(f"Found {len(char_files)} character files to reannotate")

    system_prompt = load_system_prompt(prompt_md)
    qwen_cfg = load_qwen_config(qwen_config)

    # Initialize tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    print("Initializing vLLM model...")
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=qwen_cfg["llm"]["tensor_parallel_size"],
        gpu_memory_utilization=qwen_cfg["llm"]["gpu_memory_utilization"],
        max_model_len=qwen_cfg["llm"]["max_model_len"],
        max_num_seqs=qwen_cfg["llm"]["max_num_seqs"],
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=qwen_cfg["sampling"]["temperature"],
        max_tokens=qwen_cfg["sampling"]["max_tokens"],
        top_p=qwen_cfg["sampling"]["top_p"],
    )

    print("Model loaded. Starting reannotation...\n")
    start_time = time.time()

    # Load character data and create prompts
    print("Preparing prompts...")
    prompts: List[str] = []
    file_list: List[Path] = []

    for char_file in tqdm(char_files, desc="Preparing prompts", unit="char"):
        char_data = load_character_data(char_file)
        prompt = create_prompt(char_file, char_data, tokenizer, system_prompt)
        prompts.append(prompt)
        file_list.append(char_file)

    # Process in batches
    processed = 0
    failed = 0
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    pbar = tqdm(
        total=len(prompts),
        desc="Reannotating errors",
        unit="char",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_files = file_list[start_idx:end_idx]

        # Run inference
        outputs = llm.generate(batch_prompts, sampling_params)

        # Update entries
        for output, char_file in zip(outputs, batch_files):
            try:
                response_text = output.outputs[0].text
                result = parse_response(response_text, char_file)
                # Update existing entry
                existing_entries[char_file.name] = result
                processed += 1
                pbar.update(1)
            except Exception as e:
                failed += 1
                pbar.update(1)
                tqdm.write(f"[ERROR] Failed on {char_file.name}: {e}")

    pbar.close()

    # Write updated entries back to file
    print(f"\nWriting updated entries to {annotations_file}...")
    with annotations_file.open("w", encoding="utf-8") as out_f:
        for char_json, entry in sorted(existing_entries.items()):
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Done. Success: {processed}, Failed: {failed}")
    print(f"Total time: {total_time/60:.2f} minutes")
    if processed > 0:
        print(f"Average per successful sample: {total_time/processed:.3f} seconds")
    print(f"Updated file: {annotations_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for personality annotation re-run on error entries."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-annotate anime character personalities for entries with parse errors "
            "using a Qwen3-32B-FP8 model via vLLM."
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
        "--input-annotations",
        type=Path,
        required=True,
        help="Path to the JSONL annotations file to scan and update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of prompts to send per vLLM batch (default: 100).",
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

    reannotate_errors(
        model_path=args.model_path,
        char_bio_dir=args.char_bio_dir,
        annotations_file=args.input_annotations,
        prompt_md=args.prompt_md,
        qwen_config=args.qwen_config,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
