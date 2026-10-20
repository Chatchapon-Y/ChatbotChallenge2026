"""Count the workshop posters per year on the Innovation Academy site. Lv4.

/workshop/ lists every past workshop as a poster image under a
"Year 20xx" heading. The page text alone loses which poster sits under
which heading, so an LLM reading chunks cannot count them (the dev set
asks for 2025 and the answer is 31). This reads the page's HTML once at
build time and saves {year: number of posters} to data/poster_counts.json,
which bot/answer.py reads. No API call, no quota.

Run from submission_repo (any network):

    python -m build.counts
"""
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://innoacademy.engg.hku.hk/workshop/"
OUT = Path("data/poster_counts.json")
HEADERS = {"User-Agent": "HKU Chatbot Challenge 2026, Team 1 (student project)"}
YEAR = re.compile(r"^\s*Year\s+(20\d\d)\s*$")
# The same poster at another size (poster-1024x724.jpg) is one poster.
SIZE = re.compile(r"-\d+x\d+(?=\.\w+$)")


def footer_image(src: str, year: str) -> bool:
    """The page footer (social logos, a recent photo) follows the last
    section, so it would count as that year's posters. A logo is never a
    poster, and a poster is not uploaded years after its workshop."""
    if "logo" in src.lower():
        return True
    m = re.search(r"/uploads/(20\d\d)/", src)
    return bool(m) and int(m.group(1)) > int(year) + 1


def count_posters(html: str) -> dict[str, int]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("#content") or soup
    posters, year = {}, None
    for el in root.find_all(["h2", "img"]):
        if el.name == "h2":
            m = YEAR.match(el.get_text(" ", strip=True))
            year = m.group(1) if m else None
        elif year:
            src = el.get("data-src") or el.get("src") or ""
            if src and not src.startswith("data:") and not footer_image(src, year):
                posters.setdefault(year, set()).add(SIZE.sub("", src.split("?")[0]))
    return {y: len(s) for y, s in sorted(posters.items(), reverse=True)}


if __name__ == "__main__":
    html = requests.get(URL, headers=HEADERS, timeout=30).text
    counts = count_posters(html)
    OUT.write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(counts)
