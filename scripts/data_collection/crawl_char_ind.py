import os
import json
import requests
import time
import re

# --- Config ---
INPUT_DIR = "output/parsed_mal_char"
OUTPUT_DIR = "output"
CHAR_DIR = os.path.join(OUTPUT_DIR, "mal_char_ind_pages")
IMG_DIR = os.path.join(OUTPUT_DIR, "mal_char_ind_images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "img_page_char_index.jsonl")
PLACEHOLDER = "questionmark_23"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- Create directories ---
os.makedirs(CHAR_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---
def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()

def extract_char_id(url):
    match = re.search(r'/character/(\d+)', url)
    return match.group(1) if match else "unknown"

def download_raw_html(url, dest_path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"📄 Saved HTML: {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to fetch HTML from {url}: {e}")
        return False

def download_image(url, dest_path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"🖼️  Saved image: {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to fetch image from {url}: {e}")
        return False

# --- Load already processed entries from JSONL ---
completed_htmls = set()
if os.path.exists(JSONL_PATH):
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                completed_htmls.add(obj.get("character_html"))
            except Exception:
                continue

# --- Open JSONL file in append mode ---
with open(JSONL_PATH, "a", encoding="utf-8") as jsonl_file:

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Could not read {filename}: {e}")
            continue

        for character in data.get("characters", []):
            char_name = character.get("name", "unknown")
            char_url = character.get("url")
            img_url = character.get("image")

            if not char_url:
                continue

            char_id = extract_char_id(char_url)
            file_stem = f"{safe_filename(char_name)}_{char_id}"

            html_filename = f"{file_stem}.html"
            image_filename = f"{file_stem}.jpg" if img_url and PLACEHOLDER not in img_url else ""

            # Skip if already processed
            if html_filename in completed_htmls:
                print(f"⏩ Skipping (already processed): {html_filename}")
                continue

            html_path = os.path.join(CHAR_DIR, html_filename)
            image_path = os.path.join(IMG_DIR, image_filename) if image_filename else ""

            # Download raw HTML
            if not os.path.exists(html_path):
                if download_raw_html(char_url, html_path):
                    time.sleep(0.5)
                else:
                    continue  # skip entry if HTML failed

            # Download image if needed
            if image_filename and not os.path.exists(image_path):
                if download_image(img_url, image_path):
                    time.sleep(0.5)

            # Write to JSONL
            jsonl_entry = {
                "character_name": char_name,
                "character_html": html_filename,
                "image_file": image_filename,
                "source_json": filename
            }
            jsonl_file.write(json.dumps(jsonl_entry) + "\n")
            jsonl_file.flush()
            completed_htmls.add(html_filename)  # mark as done in session
