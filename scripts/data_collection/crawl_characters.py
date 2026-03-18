import os
import requests
from bs4 import BeautifulSoup
import time
import csv
import re

# Directories
anime_html_dir = 'output/myanimelist'
characters_html_dir = 'output/mal_char'
os.makedirs(characters_html_dir, exist_ok=True)

# Mapping output
mapping_file = os.path.join(characters_html_dir, 'character_page_mapping.csv')
csv_header = ['anime_html_file', 'characters_url', 'downloaded_characters_file']

# Delay (in seconds)
delay = 2

# Helper to sanitize filenames
def sanitize_filename(url_or_name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', url_or_name)

def extract_characters_url(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    link = soup.find('a', string=lambda s: s and 'Characters & Staff' in s)
    if link and link.has_attr('href'):
        return link['href']
    return None

def process_anime_pages(anime_html_dir, characters_html_dir, mapping_file):
    with open(mapping_file, 'w', newline='', encoding='utf-8') as mapfile:
        writer = csv.writer(mapfile)
        writer.writerow(csv_header)

        for filename in os.listdir(anime_html_dir):
            if not filename.endswith('.html'):
                continue

            file_path = os.path.join(anime_html_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            characters_url = extract_characters_url(content)
            if not characters_url:
                print(f"No characters URL found in {filename}")
                continue

            try:
                print(f"Fetching: {characters_url}")
                response = requests.get(characters_url, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Failed to fetch {characters_url}: {e}")
                continue

            # Save characters page HTML
            sanitized_name = sanitize_filename(characters_url)
            characters_file = f'{sanitized_name}.html'
            characters_path = os.path.join(characters_html_dir, characters_file)

            with open(characters_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print(f"Saved characters page to {characters_path}")

            # Write mapping
            writer.writerow([filename, characters_url, characters_file])

            time.sleep(delay)

# Run it
process_anime_pages(anime_html_dir, characters_html_dir, mapping_file)
