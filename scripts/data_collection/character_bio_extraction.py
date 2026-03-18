"""
Extract structured character biographies and animeography from HTML pages.

Example usage:
python scripts/data_collection/character_bio_extraction.py \
  --input-dir /path/to/mal_char_ind_html_pages \
  --output-dir /path/to/output/char_bio_json_directory \
  --max-workers 4
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from bs4 import BeautifulSoup


def extract_biography_from_html(soup: BeautifulSoup) -> Tuple[Optional[str], str]:
    """Extract character name and biography text from a BeautifulSoup HTML tree."""
    header = soup.find("h2", attrs={"class": "normal_header"})
    if not header:
        return None, ""

    name = header.get_text(strip=True)

    # Find the next section header
    next_header = header.find_next("div", attrs={"class": "normal_header"})
    if not next_header:
        next_header = header.find_next("h2", attrs={"class": "normal_header"})

    # Collect all elements between header and next_header
    bio_elements: List = []
    current = header.next_sibling

    section_headers = [
        "Voice Actors",
        "Animeography",
        "Mangaography",
        "Recent Featured Articles",
    ]

    while current is not None:
        # Check if current is a div/h2 with normal_header class
        if hasattr(current, "name") and current.name in ["div", "h2"]:
            classes = current.get("class", [])
            is_normal_header = False
            if isinstance(classes, list):
                is_normal_header = "normal_header" in classes
            elif isinstance(classes, str):
                is_normal_header = "normal_header" in classes

            # Check if it's a known section header
            if is_normal_header:
                text_content = current.get_text(strip=True)
                if text_content in section_headers:
                    break
                if any(text_content.startswith(h) for h in section_headers):
                    break
                if not text_content or len(text_content) < 3:
                    break

        if next_header:
            try:
                if current is next_header or current == next_header:
                    break
            except Exception:
                pass

        bio_elements.append(current)
        current = current.next_sibling

    processed_elements: List = []
    for elem in bio_elements:
        if (
            hasattr(elem, "name")
            and elem.name == "div"
            and hasattr(elem, "get")
            and elem.get("class") == ["spoiler"]
        ):
            spoiler_content = elem.find("span", attrs={"class": "spoiler_content"})
            if spoiler_content:
                spoiler_text = spoiler_content.get_text(separator=" ", strip=True)
                spoiler_text = re.sub(
                    r"\b(Hide|Show)\s+spoiler\b",
                    "",
                    spoiler_text,
                    flags=re.IGNORECASE,
                ).strip()
                if spoiler_text:
                    processed_elements.append(spoiler_text + " ")
            continue

        processed_elements.append(elem)

    # Remove script, style, and ad divs
    filtered_elements: List = []
    for elem in processed_elements:
        if isinstance(elem, str):
            filtered_elements.append(elem)
            continue

        if hasattr(elem, "name"):
            if elem.name in ["script", "style"]:
                continue
            if elem.name == "div" and hasattr(elem, "get") and elem.get("class"):
                classes = elem.get("class")
                if isinstance(classes, list):
                    classes_str = " ".join(classes)
                else:
                    classes_str = str(classes)
                if (
                    "ad-" in classes_str
                    or "sUaidzctQfngSNMH" in classes_str
                    or "ad-sas" in classes_str
                ):
                    continue

        filtered_elements.append(elem)

    # Extract all text from filtered elements
    bio_text_parts: List[str] = []
    for elem in filtered_elements:
        if isinstance(elem, str):
            if elem.strip():
                bio_text_parts.append(elem.strip())
        elif hasattr(elem, "get_text"):
            text = elem.get_text(separator=" ", strip=True)
            if text:
                bio_text_parts.append(text)
        elif hasattr(elem, "strip"):
            text = str(elem).strip()
            if text and not text.isspace():
                bio_text_parts.append(text)

    biography = " ".join(bio_text_parts)
    biography = (
        biography.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    )
    biography = re.sub(r"[\s\t\n\r]+", " ", biography).strip()

    if "no biography written" in biography.lower():
        biography = ""

    return name, biography


def extract_animeography_from_html(soup: BeautifulSoup) -> list[dict]:
    """Extract animeography entries from a BeautifulSoup HTML tree."""
    anime_list: list[dict] = []
    anime_section = soup.find(
        "div", attrs={"class": "normal_header"}, string="Animeography"
    )
    if not anime_section:
        return anime_list

    table = anime_section.find_next("table")
    if not table:
        return anime_list

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        anime_link_tag = cells[1].find("a", href=True)
        role_tag = cells[1].find("small")

        if anime_link_tag:
            anime_title = anime_link_tag.text.strip()
            anime_url = anime_link_tag["href"]
            role = role_tag.text.strip() if role_tag else "Unknown"
            anime_list.append(
                {
                    "title": anime_title,
                    "url": anime_url,
                    "role": role,
                }
            )

    return anime_list


def process_single_file(input_path: Path, output_path: Path) -> str:
    """Process a single HTML file and write the extracted data to JSON."""
    try:
        html_content = input_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")

        name, bio = extract_biography_from_html(soup)
        animeography = extract_animeography_from_html(soup)

        if name:
            data = {
                "name": name,
                "biography": bio,
                "animeography": animeography,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return f"Saved: {output_path.name}"

        return f"Skipped (no name found): {input_path.name}"

    except Exception as e:
        return f"Error processing {input_path.name}: {e}"


def process_all_html_files_parallel(
    input_dir: Path, output_dir: Path, max_workers: Optional[int] = None
) -> None:
    """Process all HTML files in a directory in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path]] = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            input_path = input_dir / filename
            output_filename = Path(filename).with_suffix(".json").name
            output_path = output_dir / output_filename
            tasks.append((input_path, output_path))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file, input_path, output_path)
            for input_path, output_path in tasks
        ]
        for future in as_completed(futures):
            print(future.result())


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for character biography extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract character biographies and animeography from HTML pages "
            "into structured JSON files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing HTML files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON files will be written.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker processes (default: 4).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    process_all_html_files_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
