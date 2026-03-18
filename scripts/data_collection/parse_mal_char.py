import os
import re
import json
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


def extract_anime_metadata(soup):
    data = {}

    def get_text_after_label(label):
        tag = soup.find('span', string=re.compile(f"^{re.escape(label)}"))
        if not tag:
            return None
        next_tag = tag.find_next_sibling()
        return next_tag.text.strip() if next_tag and next_tag.name == 'a' else tag.next_sibling.strip() if tag.next_sibling else None

    def get_multiple_links(label):
        tag = soup.find('span', string=re.compile(f"^{re.escape(label)}"))
        if not tag:
            return []
        parent = tag.find_parent('div', class_='spaceit_pad')
        return [a.text.strip() for a in parent.find_all('a')]

    def get_score():
        tag = soup.find('span', itemprop='ratingValue')
        if tag:
            text = tag.text.strip()
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def get_number(label):
        tag = soup.find('span', string=re.compile(f"^{re.escape(label)}"))
        if tag:
            nums = re.findall(r'[\d,]+', tag.parent.text)
            return int(nums[0].replace(',', '')) if nums else None
        return None

    def extract_related():
        related = {}
        related_section = soup.find('table', class_='anime_detail_related_anime')
        if not related_section:
            return related

        for entry in related_section.find_all('tr'):
            rel_type = entry.find('th').text.strip(':')
            related[rel_type] = []

            for rel_entry in entry.find_all('td'):
                title_div = rel_entry.find('div', class_='title')
                if not title_div:
                    continue
                title = title_div.find('a')
                if not title:
                    continue
                related[rel_type].append({
                    'title': title.text.strip(),
                    'url': title['href']
                })

        return related

    # Alternative Titles
    data['synonyms'] = get_text_after_label("Synonyms:")
    data['japanese_title'] = get_text_after_label("Japanese:")

    # Information fields
    for field in ["Type:", "Episodes:", "Status:", "Aired:", "Duration:", "Rating:", "Source:"]:
        data[field.strip(':').lower()] = get_text_after_label(field)

    data['producers'] = get_multiple_links("Producers:")
    data['licensors'] = get_multiple_links("Licensors:")
    data['studios'] = get_multiple_links("Studios:")

    # Handle both 'Genres:' and 'Genre:'
    data['genres'] = get_multiple_links("Genres:") or get_multiple_links("Genre:")
    data['themes'] = get_multiple_links("Theme:")
    data['demographic'] = get_multiple_links("Demographic:") or None

    # Stats
    data['score'] = get_score()
    data['ranked'] = get_text_after_label("Ranked:")
    data['popularity'] = get_text_after_label("Popularity:")
    data['members'] = get_number("Members:")
    data['favorites'] = get_number("Favorites:")

    # Synopsis
    para = soup.find('p', itemprop='description')
    data['synopsis'] = para.text.strip() if para else None

    # Background
    bg_header = soup.find('h2', id='background')
    if bg_header:
        bg = bg_header.find_next_sibling(text=True)
        data['background'] = bg.strip() if bg else None

    # Related Entries
    related = extract_related()
    if related:
        data['related_entries'] = related

    return data


def parse_characters(soup):
    characters = []

    char_tables = soup.select("table.js-anime-character-table")
    for table in char_tables:
        # Character info block
        char_td = table.find_all("td")[1]
        name_tag = char_td.select_one("h3.h3_character_name")
        char_name = name_tag.text.strip() if name_tag else None
        char_url = name_tag.find_parent("a")["href"] if name_tag else None

        role_tag = char_td.select("div.spaceit_pad")[1]
        char_role = role_tag.text.strip() if role_tag else None

        favorites_tag = char_td.select("div.spaceit_pad")[-1]
        favorites_text = favorites_tag.text.strip() if favorites_tag else "0 Favorites"
        char_favorites = int(favorites_text.replace(',', '').split(" ")[0]) if favorites_text else 0


        # Character image
        char_img_td = table.find_all("td")[0]
        char_img_tag = char_img_td.select_one("img")
        char_image = char_img_tag["data-src"] if char_img_tag and char_img_tag.has_attr("data-src") else None

        # Voice actor block
        va_td = table.find_all("td")[2]
        va_rows = va_td.select("tr.js-anime-character-va-lang")
        voice_actors = []
        for row in va_rows:
            va_name_tag = row.select_one("a")
            va_name = va_name_tag.text.strip() if va_name_tag else None
            va_url = va_name_tag["href"] if va_name_tag else None
            language_tag = row.select_one("div.js-anime-character-language")
            language = language_tag.text.strip() if language_tag else None

            va_img_tag = row.find_next("img")
            va_image = va_img_tag["data-src"] if va_img_tag and va_img_tag.has_attr("data-src") else None

            voice_actors.append({
                "name": va_name,
                "url": va_url,
                "language": language,
                "image": va_image
            })

        characters.append({
            "name": char_name,
            "url": char_url,
            "role": char_role,
            "favorites": char_favorites,
            "image": char_image,
            "voice_actors": voice_actors
        })

    return characters

def parse_staff(soup):
    staff_data = []

    # Find the anchor <a name="staff"></a>
    staff_anchor = soup.find('a', attrs={'name': 'staff'})
    if not staff_anchor:
        return staff_data

    # Start from the sibling right after the anchor
    current = staff_anchor.next_sibling

    # Skip siblings until the first table is found
    while current and (getattr(current, 'name', None) != 'table'):
        current = current.next_sibling

    # Now parse all consecutive tables
    while current and current.name == 'table':
        tds = current.find_all('td')
        if len(tds) >= 2:
            entry = {}

            # Extract image URL from first td
            img_tag = tds[0].find('img')
            entry['image'] = img_tag.get('data-src') if img_tag and img_tag.has_attr('data-src') else None

            # Extract name and URL from second td
            name_tag = tds[1].find('a')
            entry['name'] = name_tag.text.strip() if name_tag else None
            entry['url'] = name_tag['href'] if name_tag else None

            # Extract roles from <small> inside div.spaceit_pad (second td)
            role_div = tds[1].find('div', class_='spaceit_pad')
            if role_div:
                small = role_div.find('small')
                if small:
                    entry['roles'] = [role.strip() for role in small.text.split(',')]
                else:
                    entry['roles'] = []
            else:
                entry['roles'] = []

            staff_data.append(entry)

        current = current.next_sibling

    return staff_data



def parse_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

    data = extract_anime_metadata(soup)
    data['characters'] = parse_characters(soup)
    data['staff'] = parse_staff(soup)
    return data


def process_single_file(in_path, out_path):
    data = parse_html_file(in_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✔ Parsed {os.path.basename(in_path)} → {os.path.basename(out_path)}")


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [fn for fn in os.listdir(input_dir) if fn.endswith('.html')]

    with ThreadPoolExecutor() as executor:
        futures = []
        for fn in files:
            in_path = os.path.join(input_dir, fn)
            out_name = os.path.splitext(fn)[0] + '.json'
            out_path = os.path.join(output_dir, out_name)
            futures.append(executor.submit(process_single_file, in_path, out_path))

        for future in futures:
            future.result()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Parse MyAnimeList HTML to JSON")
    parser.add_argument('input_dir', help='Folder with HTML files')
    parser.add_argument('output_dir', help='Folder where JSON files will be saved')
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)
