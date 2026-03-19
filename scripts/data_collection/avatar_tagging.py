import os
import json
from tqdm import tqdm
from imgutils.tagging import get_camie_tags

# === Configuration ===
input_folder = 'output/mal_char_ind_hr_images/'   # Change this
output_file = 'avatar_camie_tags.jsonl'             # Output path

# Supported image extensions
valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

# Collect all image paths
image_paths = [
    os.path.join(input_folder, fname)
    for fname in os.listdir(input_folder)
    if os.path.splitext(fname)[1].lower() in valid_exts
]

# Process and write to JSONL
with open(output_file, 'w', encoding='utf-8') as out_f:
    for path in tqdm(image_paths, desc="Tagging images"):
        try:
            rating, features, chars = get_camie_tags(path)

            # Combine everything in a single dict
            result = {
                'image_path': path,
                'rating': rating,
                'tags': features,   # features is a dict of tags with scores
                'characters': chars
            }

            out_f.write(json.dumps(result) + '\n')

        except Exception as e:
            print(f"Failed to process {path}: {e}")
