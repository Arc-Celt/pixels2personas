import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_main_html(file_path, output_dir):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")
    except Exception as e:
        return f"❌ Error reading {file_path}: {e}"

    info_section = soup.find("div", class_="pure-1 md-3-5")
    if not info_section:
        return f"⚠️ Info section not found in {file_path}"

    # Description
    description_tag = info_section.find("div")
    description = description_tag.get_text(strip=True) if description_tag else ""

    # Notes
    notes_tag = info_section.find("div", class_="notes")
    notes = notes_tag.get_text(strip=True).replace("Source:", "").strip() if notes_tag else ""

    # Tags
    tag_container = info_section.find("div", class_="tags")
    tags = []
    if tag_container:
        tags = [li.get_text(strip=True) for li in tag_container.find_all("li")]

    # Watch Status
    status_select = info_section.find("select", class_="changeStatus")
    selected_option = status_select.find("option", selected=True) if status_select else None
    status = selected_option.get_text(strip=True) if selected_option else "Unwatched"

    result = {
        "description": description,
        "notes": notes,
        "tags": tags,
        "status": status
    }

    base_name = Path(file_path).stem
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return f"✅ Parsed: {file_path}"


def process_main_folder(input_dir, output_dir, max_workers=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = [
        f for f in input_dir.rglob("*.html")
        if "main" in f.stem.lower()
    ]

    if not html_files:
        print("No main HTML files found.")
        return

    print(f"🔍 Found {len(html_files)} 'main' HTML files to process.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(parse_main_html, str(f), str(output_dir)): f for f in html_files
        }

        for future in as_completed(futures):
            print(future.result())


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    input_directory = "output/anime-planet"      # Folder containing 'main' HTMLs
    output_directory = "output/ap_main"
    process_main_folder(input_directory, output_directory, max_workers=6)
