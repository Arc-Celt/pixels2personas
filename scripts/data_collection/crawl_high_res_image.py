import os
import requests
import time
from bs4 import BeautifulSoup
import re
import json

# --- Config ---
HTML_DIR = "output/mal_char_ind_pages"
IMG_DIR = "output/mal_char_ind_hr_images"
MISSING_LOG = os.path.join(IMG_DIR, "missing_images.jsonl")
HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(IMG_DIR, exist_ok=True)

# Load already logged missing files
logged_missing = set()
if os.path.exists(MISSING_LOG):
    with open(MISSING_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                logged_missing.add(entry.get("html_file"))
            except json.JSONDecodeError:
                continue  # skip bad lines

def extract_image_url(soup):
    img_tag = soup.find("img", class_="portrait-225x350")
    if img_tag and img_tag.has_attr("data-src"):
        return img_tag["data-src"]
    return None

def download_image(url, dest_path):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print(f"🖼️  Saved image: {dest_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to download image: {e}")
        return False

# Main processing loop
for html_filename in os.listdir(HTML_DIR):
    if not html_filename.endswith(".html"):
        continue

    if html_filename in logged_missing:
        print(f"⏩ Skipping (previously logged missing): {html_filename}")
        continue

    html_path = os.path.join(HTML_DIR, html_filename)
    image_filename = html_filename.replace(".html", ".jpg")
    image_path = os.path.join(IMG_DIR, image_filename)

    if os.path.exists(image_path):
        print(f"⏩ Image already exists: {image_path}")
        continue

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"[!] Failed to read {html_filename}: {e}")
        continue

    img_url = extract_image_url(soup)
    if not img_url:
        print(f"🚫 No image found in: {html_filename}")
        with open(MISSING_LOG, "a", encoding="utf-8") as logf:
            logf.write(json.dumps({
                "html_file": html_filename,
                "reason": "no_image_tag"
            }) + "\n")
        continue

    if download_image(img_url, image_path):
        time.sleep(1)
