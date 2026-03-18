import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_character_section(soup):
    return soup.find("div", class_="pure-1 md-4-5")


def parse_character_section(section_div):
    sections = section_div.find_all("h3", class_="sub")
    if not sections:
        return None

    all_characters = []

    for section in sections:
        category = section.get_text(strip=True)  # e.g., "Main Characters"
        table = section.find_next("table")
        if not table:
            continue

        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            name_tag = cols[1].find("a", class_="name")
            name = name_tag.get_text(strip=True) if name_tag else "N/A"

            img_tag = cols[0].find("img")
            img_url = img_tag["src"] if img_tag else "N/A"

            tag_list = []
            tags_section = cols[1].find("div", class_="tags")
            if tags_section:
                tag_list = [li.get_text(strip=True) for li in tags_section.find_all("li")]

            actor_div = cols[-1].find("a", class_="tooltip")
            va_name = actor_div.get_text(strip=True) if actor_div else "N/A"

            all_characters.append({
                "name": name,
                "image_url": img_url,
                "tags": tag_list,
                "japanese_voice_actor": va_name,
                "category": category
            })

    return all_characters if all_characters else None


def process_file(file_path, output_dir):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")
    except Exception as e:
        return f"❌ Error reading {file_path}: {e}"

    section_div = extract_character_section(soup)
    if not section_div:
        return f"⚠️ No character section found in {file_path}"

    data = parse_character_section(section_div)
    if not data:
        return f"⚠️ No characters parsed in {file_path}"

    base_name = Path(file_path).stem
    output_path = os.path.join(output_dir, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return f"✅ Processed {file_path}"


def process_folder(input_dir, output_dir, max_workers=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = [
        f for f in input_dir.rglob("*.html")
        if "characters" in f.stem.lower()
    ]

    if not html_files:
        print("No valid character HTML files found.")
        return

    print(f"🔍 Found {len(html_files)} character HTML files.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, str(file_path), str(output_dir)): file_path
            for file_path in html_files
        }

        for future in as_completed(futures):
            print(future.result())



# === USAGE EXAMPLE ===
if __name__ == "__main__":
    input_directory = "output/anime-planet"      # Folder containing raw HTMLs
    output_directory = "output/ap_characters"    # Output for parsed JSON files
    process_folder(input_directory, output_directory, max_workers=6)
