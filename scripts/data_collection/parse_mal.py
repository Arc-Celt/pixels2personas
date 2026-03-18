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
    entries = []
    container = soup.find('div', class_='detail-characters-list')
    if not container:
        return entries

    for table in container.find_all('table', recursive=True):
        char = {}
        name_tag = table.select_one('h3.h3_characters_voice_actors a')
        if not name_tag:
            continue
        char['character_name'] = name_tag.text.strip()
        char['character_url'] = name_tag['href']
        role_tag = table.select_one('div.spaceit_pad small')
        char['role'] = role_tag.text.strip() if role_tag else None
        va_tag = table.select_one('td.va-t.ar.pl4.pr4 a')
        char['voice_actor'] = va_tag.text.strip() if va_tag else None
        char['voice_actor_url'] = va_tag['href'] if va_tag else None
        lang_tag = table.select_one('td.va-t.ar.pl4.pr4 small')
        char['language'] = lang_tag.text.strip() if lang_tag else None
        img_tag = table.select_one('td.ac.borderClass img')
        char['character_image'] = img_tag['data-src'] if img_tag and img_tag.get('data-src') else None
        va_img = table.select_one('td[valign="top"] img')
        char['voice_actor_image'] = va_img['data-src'] if va_img and va_img.get('data-src') else None
        entries.append(char)

    return entries

def parse_staff(soup):
    staff_data = []

    # Find the anchor <a name="staff"></a>
    staff_anchor = soup.find('a', attrs={'name': 'staff'})
    if not staff_anchor:
        return staff_data

    # Find the next sibling div with class 'detail-characters-list clearfix'
    container = None
    sibling = staff_anchor.next_sibling
    while sibling:
        if getattr(sibling, 'name', None) == 'div' and sibling.get('class'):
            classes = sibling.get('class')
            if 'detail-characters-list' in classes and 'clearfix' in classes:
                container = sibling
                break
        sibling = sibling.next_sibling

    if not container:
        return staff_data

    for table in container.find_all('table'):
        entry = {}

        # Image URL
        img_td = table.find('td', class_='ac borderClass')
        img_tag = img_td.find('img') if img_td else None
        entry['image'] = img_tag['data-src'] if img_tag and img_tag.has_attr('data-src') else None

        # Name and URL extraction
        # The second <td> with class containing 'borderClass'
        tds = table.find_all('td', class_=lambda c: c and 'borderClass' in c)
        if len(tds) < 2:
            continue
        name_td = tds[1]
        person_link = name_td.find('a')
        if person_link:
            name_text = person_link.get_text(strip=True)
            entry['name'] = name_text if name_text else None
            entry['url'] = person_link.get('href', None)
        else:
            entry['name'] = None
            entry['url'] = None

        # Role extraction from <small> inside div.spaceit_pad
        role_div = name_td.find('div', class_='spaceit_pad')
        role_text = None
        if role_div:
            small = role_div.find('small')
            if small:
                role_text = small.get_text(strip=True)
        entry['role'] = role_text

        staff_data.append(entry)

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
