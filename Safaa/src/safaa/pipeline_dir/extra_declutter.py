import re
import pandas as pd
import json


# # === Regex patterns ===
#
# # Copyright symbols + years + holders
# COPYRIGHT_RE = re.compile(
#     r"""
#     (?:                              # Start non-capturing group
#         ©                            # Symbol ©
#         | \(c\)                      # or (c)
#         | copyright                  # or the word 'copyright'
#     )
#     [\s,:]*                          # Optional space or punctuation
#     (?P<years>(\d{4})(?:[\-,–]\d{4})?(?:,\s*\d{4})*)?  # Year or range
#     [\s,]*(?:by)?[\s]*               # optional 'by'
#     (?P<holder>
#         (?!the\s+copyright)          # negative lookahead to avoid 'the copyright law'
#         (?:(?!\b(all|rights?|reserved)\b)[\w\s\.\-&(),]*){1,3}  # Try 1-3 phrases
#     )
#     """,
#     re.IGNORECASE | re.VERBOSE
# )
#
# # License-related phrases
# LICENSE_RE = re.compile(
#     r"""
#     (?:
#         licensed\s+under\s+(?:the\s+)?     # 'licensed under (the)?'
#         (?P<license>[^.\n]+?)              # license name (up to period or newline)
#         (?:\s+license)?                    # optional word 'license'
#     )
#     |
#     (?:
#         released\s+under\s+(?:the\s+)?     # 'released under ...'
#         (?P<released_license>[^.\n]+?)
#         (?:\s+license)?
#     )
#     |
#     (?:
#         distributed\s+under\s+(?:the\s+)?  # 'distributed under ...'
#         (?P<distributed_license>[^.\n]+?)
#         (?:\s+license)?
#     )
#     """,
#     re.IGNORECASE | re.VERBOSE
# )
#
# # === Core function ===
#
# def declutter_text(text: str) -> Dict[str, List[Dict[str, str]]]:
#     cleaned = {
#         "copyrights": [],
#         "licenses": []
#     }
#
#     # Normalize double quotes
#     text = text.replace('“', '"').replace('”', '"')
#
#     # Extract copyright
#     for match in COPYRIGHT_RE.finditer(text):
#         years = match.group("years")
#         holder = match.group("holder")
#
#         # Basic cleanup
#         if holder:
#             holder = holder.strip(" ,.\n").strip()
#         if years:
#             years = years.strip(" ,.\n").strip()
#
#         # Only if meaningful
#         if holder or years:
#             cleaned["copyrights"].append({
#                 "year": years if years else "",
#                 "holder": holder if holder else ""
#             })
#
#     # Extract licenses
#     for match in LICENSE_RE.finditer(text):
#         for key in ["license", "released_license", "distributed_license"]:
#             license_text = match.group(key)
#             if license_text:
#                 cleaned["licenses"].append(license_text.strip(" ,.\n"))
#
#     return cleaned


COPYRIGHT_RE = re.compile(
    r"copyright\s*\(c\)?\s*(?P<years>[\d{4},\s–-]+)?\s*(?P<holder>[^\n,\.]+)",
    re.IGNORECASE
)

LICENSE_RE = re.compile(
    r"licensed\s+under\s+the\s+(?P<license>[\w\s\-\.]+?)(?:\s+license)?[\.]", re.IGNORECASE
)


def declutter_text(text):
    cleaned = {"copyrights": [], "licenses": []}

    for match in COPYRIGHT_RE.finditer(text):
        years = match.group("years")
        holder = match.group("holder")

        if years and holder:
            # Clean up extraneous symbols/spacing
            cleaned["copyrights"].append({
                "year": years.strip(" ,"),
                "holder": holder.strip(" ,.")
            })

    for match in LICENSE_RE.finditer(text):
        cleaned["licenses"].append(match.group("license").strip())

    return cleaned



df = pd.read_csv('data/preprocessed_copyrights.csv')
data = df['original_content']

decluttered_outputs = [declutter_text(text) for text in data]

decluttered_texts = [json.dumps(result) for result in decluttered_outputs]

new_df = pd.DataFrame({
    'original_content': data,
    'decluttered_content': decluttered_texts
})

new_df.to_csv("data/extra_declutter.csv", index=False)
