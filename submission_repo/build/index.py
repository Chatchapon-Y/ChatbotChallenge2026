"""Indexer. STUB. Workshop 1 block 4.

Chunk the scraped text and write it to the store. Run this file
directly, after build/scrape.py, to (re)build data/chroma.

Store metadata now. Level 4 questions need to filter by year and page
type, and adding a field later means rebuilding everything.
"""
import json
import re
from pathlib import Path

from bot.store import add_to_store, get_store

CHUNK_SIZE, OVERLAP = 800, 100


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunks with overlap.

    Overlap exists because a fact split across a boundary is lost: a
    date at char 998 and its event name at char 1002 land in different
    chunks and neither answers the question.

    TODO try a better strategy once this works: split on headings
    TODO first, fall back to characters last, and prepend the section
    TODO heading to each chunk.
    """
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def guess_year(page: dict) -> str:
    """First 20xx found in the URL, then the title, then the opening text.

    Inno Academy URLs often carry the date (/20231106_workshop/, /pitching_2025/).
    Returns "" when nothing matches, because Chroma metadata cannot be None.
    """
    for source in (page["url"], page.get("title", ""), page["text"][:300]):
        m = re.search(r"20[12]\d", source)
        if m:
            return m.group(0)
    return ""


# WordPress shows this instead of the content on password-protected pages.
PASSWORD_TEXT = re.compile(
    r"This content is password protected\.?\s*"
    r"(To view it please enter your password below:\s*Password:)?", re.I)

# A sentence on this many pages or more is a widget or a repeated notice
# ("Seats for on-site participants are limited."), not page content.
BOILERPLATE_MIN_PAGES = 20
MIN_PAGE_CHARS = 50

# The challenge's own announcement pages list the dev set questions with
# their answers. Indexing them makes the dev set score look better than
# the bot is, and the hidden test questions are not on them.
EXCLUDE_URLS = ("/aichallenge/", "/ai-chatbot-challenge-inno-trivia")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def title_words(page: dict) -> set[str]:
    # "RoboMaster 2019 – Innovation Wing" -> {"robomaster", "2019"}
    title = page.get("title", "").split(" – ")[0].lower()
    return {w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3}


def clean_pages(pages: list[dict]) -> list[dict]:
    """Strip password notices and repeated boilerplate, drop empty pages.

    A repeated sentence is kept once, on the page whose title shares the
    most words with it (the filters project keeps its own description),
    and removed everywhere else.
    """
    pages = [p for p in pages if not any(u in p["url"] for u in EXCLUDE_URLS)]
    sentences = [split_sentences(PASSWORD_TEXT.sub(" ", p["text"])) for p in pages]

    where = {}
    for i, sents in enumerate(sentences):
        for s in set(sents):
            if len(s) > 30:
                where.setdefault(s, []).append(i)
    boiler = {s: pages_with for s, pages_with in where.items()
              if len(pages_with) >= BOILERPLATE_MIN_PAGES}

    owner = {}
    for s, pages_with in boiler.items():
        words = {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}
        owner[s] = max(pages_with, key=lambda i: len(words & title_words(pages[i])))

    cleaned, before, after = [], 0, 0
    for i, (page, sents) in enumerate(zip(pages, sentences)):
        text = " ".join(s for s in sents if s not in boiler or owner[s] == i)
        before += len(page["text"])
        after += len(text)
        if len(text) >= MIN_PAGE_CHARS:
            cleaned.append({**page, "text": text})

    print(f"boilerplate sentences: {len(boiler)}   "
          f"pages kept: {len(cleaned)}/{len(pages)}   "
          f"text: {before:,} -> {after:,} chars")
    return cleaned


def build_index(pages: list[dict], reset: bool = True):
    texts, metas = [], []
    for page in pages:
        year = guess_year(page)
        for i, piece in enumerate(chunk(page["text"])):
            # Prefix the page title so a chunk from the middle of a page
            # still says what page it belongs to.
            texts.append(f"{page.get('title', '')}\n{piece}")
            metas.append({
                "url":       page["url"],
                "title":     page.get("title", ""),
                "position":  i,
                "kind":      "text",
                "year":      year,
                "page_type": page.get("page_type", ""),
                "site":      page.get("site", ""),
            })
    store = get_store(reset=reset)
    add_to_store(store, texts, metas)
    print(f"indexed {len(texts)} chunks")
    return store


if __name__ == "__main__":
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    pages = clean_pages(pages)
    # Saved so you can open it and check what the index actually holds.
    Path("data/pages_clean.json").write_text(json.dumps(pages, indent=1), encoding="utf-8")
    build_index(pages)
