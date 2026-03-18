import os
import json
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from bs4 import NavigableString, Tag

def extract_biography_from_html(soup):
    # 1. Locate the Header
    header = soup.find("h2", class_="normal_header")
    if not header:
        return None, ""
    
    name = header.get_text(strip=True)
    bio_parts = []
    
    # 2. Iterate over EVERYTHING after the header
    # We use list() to ensure we get a stable list of all siblings
    for element in header.next_siblings:
        
        # --- CASE 1: Text Nodes ---
        # If name is None, it is raw text. We Keep it.
        if element.name is None:
            text = str(element).strip()
            if text:
                bio_parts.append(text)
            continue
            
        # --- CASE 2: Tags ---
        tag_name = element.name
        
        # A. Handle Newlines
        if tag_name == "br":
            bio_parts.append("\n")
            continue
            
        # B. Handle Divs (The tricky part)
        if tag_name == "div":
            classes = element.get("class", []) or []
            style = element.get("style", "") or ""

            # Check for Spoilers (Keep this!)
            if "spoiler" in classes:
                spoiler_span = element.find("span", class_="spoiler_content")
                if spoiler_span:
                    # Remove buttons so they don't appear in text
                    for btn in spoiler_span.find_all("input"):
                        btn.decompose()
                    # Get text with proper newlines
                    s_text = spoiler_span.get_text(separator="\n", strip=True)
                    bio_parts.append(f"\n[Spoiler: {s_text}]\n")
                continue
            
            # Check for STOP Signals (End of Bio)
            
            # 1. "Voice Actors" header -> STOP
            if "normal_header" in classes:
                break
                
            # 2. The Ad Container (from your Conan logs) -> STOP
            # It usually has inline padding style or specific ad classes
            if "padding" in style and "display: inline-block" in style:
                break
            
            # 3. Generic "Safety Valve"
            # If we hit a div that has text but ISN'T a spoiler, it's likely the footer/ads.
            # We break to be safe.
            if element.get_text(strip=True):
                break
                
            # If it's an empty div (invisible spacer), ignore and continue
            continue

        # C. Handle other headers or tables -> STOP
        if tag_name in ["h2", "h3", "table", "form"]:
            break
            
        # D. Handle Formatting (b, i, span) -> Keep Text
        # If it's not a div/br/table, it's likely formatting like <b>Bold</b>
        text = element.get_text(strip=True)
        if text:
            bio_parts.append(text)

    # 3. Clean up the Result
    # Join with spaces, but fix the newlines we manually added
    full_text = " ".join(bio_parts)
    
    # Remove spaces around newlines (e.g., " \n " -> "\n")
    full_text = full_text.replace(" \n ", "\n").replace("\n ", "\n").replace(" \n", "\n")
    
    # Fix excessive newlines (3+ enters become 2)
    import re
    full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
    
    # Final check for empty bio placeholder
    if "no biography written" in full_text.lower():
        full_text = ""
        
    return name, full_text.strip()


def extract_animeography_from_html(soup):
    anime_list = []
    anime_section = soup.find("div", class_="normal_header", string="Animeography")
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
            anime_url = anime_link_tag['href']
            role = role_tag.text.strip() if role_tag else "Unknown"
            anime_list.append({
                "title": anime_title,
                "url": anime_url,
                "role": role
            })

    return anime_list

def process_single_file(args):
    input_path, output_path = args
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, "html.parser")

            name, bio = extract_biography_from_html(soup)
            animeography = extract_animeography_from_html(soup)

            if name:
                data = {
                    "name": name,
                    "biography": bio,
                    "animeography": animeography
                }
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(data, json_file, indent=2, ensure_ascii=False)
                return f"✅ Saved: {os.path.basename(output_path)}"
            else:
                return f"⚠️ Skipped (no name found): {os.path.basename(input_path)}"
    except Exception as e:
        return f"❌ Error processing {os.path.basename(input_path)}: {e}"

def process_all_html_files_parallel(input_dir, output_dir, max_workers=None):
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            tasks.append((input_path, output_path))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result())

# Update these paths as needed
input_folder = "output/mal_char_ind_pages"
output_folder = "output/char_bio_new_json"

# Run with default CPU core count (or specify max_workers=4, etc.)
process_all_html_files_parallel(input_folder, output_folder, max_workers=6)
