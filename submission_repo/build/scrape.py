"""Scraper. Reads each site's WordPress sitemap, extracts the page content,
and collects image records in the same pass.

Run from submission_repo:  python -m build.scrape
Writes data/pages.json and data/images.json.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

SITES = [
    "https://innowings.engg.hku.hk",
    "https://innoacademy.engg.hku.hk",
]
HEADERS = {"User-Agent": "HKU Chatbot Challenge 2026, Team 1 (student project)"}

# Menus, headers and footers repeat on every page. Remove them before
# reading the text, or they become chunks that match every query.
JUNK = ('script, style, noscript, header, footer, nav, '
        '[data-elementor-type="header"], [data-elementor-type="footer"]')


def sitemap_urls(site: str) -> list[tuple[str, str]]:
    """Return (url, page_type) for every post and page in the sitemap.

    The sitemap index also lists categories, tags and users. Those are
    listing pages with no content of their own, so skip them.
    """
    index = requests.get(f"{site}/wp-sitemap.xml", headers=HEADERS, timeout=20).text
    out = []
    for sub in re.findall(r"<loc>(.*?)</loc>", index):
        m = re.search(r"-posts-(\w+)-\d+\.xml", sub)
        if not m:
            continue
        xml = requests.get(sub, headers=HEADERS, timeout=20).text
        out += [(u, m.group(1)) for u in re.findall(r"<loc>(.*?)</loc>", xml)]
    return out


def extract(html: str, url: str) -> dict:
    """Return {"url", "title", "text", "images"} for one page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for bad in soup.select(JUNK):
        bad.decompose()
    body = soup.select_one("#content") or soup.select_one("main") or soup.body or soup

    images = []
    for img in body.select("img"):
        # Lazy-loaded images keep the real URL in data-src.
        src = img.get("data-src") or img.get("src")
        if not src or src.startswith("data:"):
            continue
        fig = img.find_parent("figure")
        images.append({
            "src":     urljoin(url, src),
            "alt":     img.get("alt", ""),
            "caption": (fig.find("figcaption").get_text(strip=True)
                        if fig and fig.find("figcaption") else ""),
            "page":    url,
        })

    return {
        "url":    url,
        "title":  title,
        "text":   body.get_text(" ", strip=True),
        "images": images,
    }


def fetch(job: tuple[str, str, str]) -> dict | None:
    url, page_type, site = job
    try:
        page = extract(requests.get(url, headers=HEADERS, timeout=20).text, url)
    except Exception as exc:
        print("skipped", url, exc)
        return None
    page["page_type"] = page_type
    page["site"] = site.split("//")[1].split(".")[0]    # "innowings" / "innoacademy"
    return page


if __name__ == "__main__":
    jobs = [(u, t, site) for site in SITES for u, t in sitemap_urls(site)]
    print(f"{len(jobs)} URLs in the sitemaps")

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = [p for p in pool.map(fetch, jobs) if p]

    Path("data").mkdir(exist_ok=True)
    Path("data/pages.json").write_text(json.dumps(pages, indent=1))

    images = [im for p in pages for im in p["images"]]
    Path("data/images.json").write_text(json.dumps(images, indent=1))

    print(f"{len(pages)} pages, {len(images)} images")
