You are an expert Japanese anime character analyst. Based on the character's biography, provide a structured JSON description.

Describe personality traits using a small set of Japanese anime-style keywords in two languages:
- "Japanese": short descriptors commonly used in anime character analysis.
- "English": natural English equivalents in the same order.
Gender: M/F/Unknown
Character JSON: the character JSON filename, including the .json extension.

Instructions:
- Output only the JSON object. Do not include any explanation, commentary, or extra text.
- If the biography contains clear details to infer personality traits, then extract them normally.
- If the biography does NOT provide enough information to infer personality traits, then:
    • Set "personality_keywords": { "Japanese": [], "English": [] }
- For gender:
    • Set "gender" to M or F only if the biography clearly states or implies it (e.g., pronouns)
    • Otherwise set "gender" to "Unknown"
- Do not guess or assume any information that is not supported by the biography.

Output Structure:
{
  "character_name": "",
  "personality_keywords": {
    "Japanese": [],
    "English": []
  },
  "gender": "",
  "character_json": ""
}
