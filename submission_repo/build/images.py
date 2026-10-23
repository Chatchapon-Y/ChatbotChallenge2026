"""Image pipeline. Workshop 2 blocks 2 and 3.

Describe every collected image once, cache the result, and add the
descriptions to the same store as the scraped text.

Run from submission_repo, on HKU Wi-Fi or HKUVPN:

    python -m build.images test [image url] [low]
                                              describe one image, print it
                                              (optionally at a reasoning effort:
                                              minimal, low, medium, high)
    python -m build.images 50 pitching        describe up to 50 new images from
                                              pages with "pitching" in the URL,
                                              then index everything cached
    python -m build.images                    describe every image, then index
    python -m build.images index              index the cache, no vision calls
                                              (changed chunks keep their old
                                              embedding; add "fresh" to re-embed)
    python -m build.images merge their.json   add a teammate's cached descriptions
    (local_vlm/descriptions_local.json is NOT merged: index reads it as is,
    behind the gpt-5-mini descriptions, labelled as rough)

`python -m build.index` wipes data/chroma, image descriptions included.
Run `python -m build.images index` after it; the cache makes that cheap.

Never describe an image while answering a question. Ingestion time is
unlimited; runtime is 30 seconds.
"""
import base64
import hashlib
import json
import mimetypes
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

from bot.llm import VISION_DEPLOYMENT, VISION_URL, _client
from bot.store import add_to_store, get_store
from build.index import EXCLUDE_URLS, guess_year

# The model writes about whatever is named here and skips the rest, so
# every kind of fact a question could ask for is named. Text comes first
# because most images on the two sites are posters, slides and award
# photos, where the answer is a name or a number printed in the image.
DESCRIPTION_PROMPT = """You are extracting facts from an image on the HKU Innovation Wing website so that a search engine can find them later. Do not write a caption, a summary or alternative versions. Output these five sections, in this order, as plain text.

TEXT: Copy every piece of visible text exactly as written, word for word: titles, names of people, teams, projects and companies, awards and rankings, dates, numbers, signs, labels, logos, banners, slides, screens. Keep the original spelling and capitalisation, reading unusual names letter by letter. Keep each name next to the award, rank or role it belongs to. Write [unreadable] in place of any part you cannot make out. Never guess or correct spelling.
TYPE: One of: poster, slide, certificate, group photo, event photo, room photo, equipment photo, diagram, chart, screenshot, logo, portrait, other.
PLACE: The room or area, only as a sign or visible text names it. Otherwise "not shown".
OBJECTS: Every distinct kind of object or person with an exact count ("11 chairs", "4 people holding certificates"), never "some" or "several". If items are partly hidden, count the ones you can see and say "at least". Name machines and tools specifically.
LAYOUT: What is next to, behind, in front of or on what. Mention colours and materials only where they identify something.

Report only what is visible in the image. Do not add outside knowledge. If a section has nothing, write "none". Do not add anything else."""

CACHE_PATH = Path("data/descriptions.json")
OCR_PATH = Path("data/ocr.json")            # written by build/ocr.py
COLOURS_PATH = Path("data/colours.json")    # written by build/colours.py
# Descriptions from a local model (qwen2.5vl:7b, local_vlm/README.md).
# Kept apart from the gpt-5-mini cache: its text is good but its counts
# are not (11 people on a poster came back as 4), so its chunks say so.
LOCAL_PATH = Path("local_vlm/descriptions_local.json")
LOCAL_NOTE = "Rough description by a small local model; its counts may be wrong."
SECTIONS = ("TEXT:", "TYPE:", "PLACE:", "OBJECTS:", "LAYOUT:")
OCR_MAX_CHARS = 3000
# An image whose OCR text is at least this long is a poster, slide or
# certificate: OCR already copied what it says, so it skips the vision
# model and the quota goes to photos, diagrams and logos, where the
# answer is a count or an object rather than printed words.
OCR_ENOUGH = 200
TMP_DIR = Path("data/_tmp")
HEADERS = {"User-Agent": "HKU Chatbot Challenge 2026, Team 1 (student project)"}

# The dev set's Lv3 question: the three winning teams are printed in this
# image and nowhere in the page text.
TEST_IMAGE = "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/03/Pitching-Winner-1024x724.jpg"

SITES = ("innowings.engg.hku.hk", "innoacademy.engg.hku.hk")
SKIP_EXT = {".svg", ".gif", ".bmp"}     # icons and animations, not content
# An image on this many pages is a logo or a sidebar widget.
MAX_PAGES = 20
# WordPress serves resized copies named photo-300x200.jpg next to the
# original photo.jpg.
SIZE = re.compile(r"-(\d+)x(\d+)(?=\.\w+$)")
NO_SIZE = 10 ** 6   # "side" of a URL with no size in its name, i.e. an original
MIN_BYTES = 5000    # smaller files are icons and spacers
MIN_PIXELS = 250    # an image whose longest side is shorter is an icon or button
# An original can be a camera file of many MB, slow to upload and no
# easier to read. Above this, use the resized copy the page shows.
MAX_BYTES = 4_000_000

WORKERS = 8         # images described at the same time (one takes 30-45s)
SAVE_EVERY = 20     # write the cache this often, so a dropped VPN loses little
# The key has a token quota (about 1,000,000, reset roughly daily: a
# 403 "Token quota is exceeded. Try again in 16 hours"). The bot and
# eval_dev.py use the same key, so one run stops after this many tokens.
TOKEN_BUDGET = 700_000
# gpt-5-mini thinks before it answers, and the thinking is billed. Lower
# effort means fewer tokens per image; check it still reads small text.
# Measured on lathe3.jpg (2026-10-18): default 3,114 tokens, "low" 1,772
# with the same counts and text. Adding detail "low" (1,133) lost the
# 1440D model number and counted 1 handwheel instead of 3, and counts are
# what the vision model is for here, so it stays off.
REASONING_EFFORT = "low"
EFFORTS = ("minimal", "low", "medium", "high")
# Two more ways to spend fewer tokens per image, both off by default:
# IMAGE_DETAIL "low" sends a small fixed-size version of the picture
# (cheap, but small print is lost; OCR already has the text), and
# VISION_MAX_SIDE shrinks the picture before sending it.
IMAGE_DETAIL = "auto"           # "auto", "low" or "high"
VISION_MAX_SIDE = None          # e.g. 768; None = send as downloaded

# Mock Lv3 questions are almost all about posters ("According to the
# Nocodile AI poster..."). With a quota too small for every image, these
# go first, then the first image of every page, then the second...
PRIORITY = re.compile(r"poster|winner|result|award|champion|schedule|floor.?plan"
                      r"|rundown|programme|flyer|leaflet", re.I)
# Workshop 0: prelim Lv3 questions come from "posters, event descriptions,
# images of the different rooms". OCR covers posters; room and machine
# photos need the vision model (counts, what is where), so they go next.
ROOM_PAGES = re.compile(
    r"/(makerspace|machineshop|brainstorming|vr-room|wet-?lab|safety_wetlab|equipment"
    r"|digitalprojectstudio|electronic-workbench|laser|waterjet|3dprinting|slaprinting"
    r"|venue|ledwall|photowall|event-?hall|innowing1/(equipment|counter|special|contact)"
    r"|innowing2/venue|labtour[^/]*|facilit[^/]*)/", re.I)
# Camera and gallery file names (DSC01234.jpg, IMG_2041.jpg,
# 20221130_InnoShow_S-175.jpg). Crowd and event photos rarely hold an
# answer, so with the quota they go to the back of the queue.
CAMERA = re.compile(r"^(dsc|img|_?mg|photo|p\d|r_?\d|\d{6,}|whatsapp|0o2a|c\d{4}|screenshot)"
                    r"|innoshow[-_ ]?\d|[-_ ]\d{2,4}(-\d+)?(-\d+x\d+)?\.\w+$", re.I)

# Text chunks of a page are numbered 0, 1, 2... Images start here so the
# two never share a (url, position) pair.
IMAGE_POSITION = 100_000


def pick_images(images: list[dict]) -> list[dict]:
    """Reduce the scraper's records to one entry per picture worth describing.

    Drops other hosts, icon formats, logos repeated across the site and
    the challenge's own pages (see EXCLUDE_URLS in build/index.py), then
    keeps the largest of each picture's resized copies.
    """
    groups = {}
    for im in images:
        path = im["src"].split("?")[0]
        if urlparse(path).netloc not in SITES:
            continue
        if Path(path).suffix.lower() in SKIP_EXT:
            continue
        if any(u in im["page"] for u in EXCLUDE_URLS):
            continue
        groups.setdefault(SIZE.sub("", path), []).append(im)

    def side(im: dict) -> int:
        m = SIZE.search(im["src"].split("?")[0])
        return max(int(m.group(1)), int(m.group(2))) if m else NO_SIZE

    picked = []
    for original, group in groups.items():
        if len({im["page"] for im in group}) >= MAX_PAGES:
            continue
        best = max(group, key=side)
        picked.append({
            "src":      best["src"],
            # Small text on a poster is lost in a resized copy, so the
            # original is tried first whenever the page only has copies.
            "original": original if side(best) < NO_SIZE else "",
            "alt":      next((im["alt"] for im in group if im.get("alt")), ""),
            "caption":  next((im["caption"] for im in group if im.get("caption")), ""),
            "page":     group[0]["page"],
        })
    return picked


def download(src: str) -> Path:
    """Save one image under data/_tmp and return its path.

    The file keeps its extension (.png, .jpg) because the image type sent
    to the model is worked out from the file name.
    """
    r = requests.get(src, headers=HEADERS, timeout=30)
    r.raise_for_status()
    suffix = Path(src.split("?")[0]).suffix.lower() or ".jpg"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TMP_DIR / (hashlib.md5(src.encode()).hexdigest() + suffix)
    tmp.write_bytes(r.content)
    return tmp


def by_priority(images: list[dict]) -> list[dict]:
    """Order of the vision queue when the quota cannot cover every image.

    1. poster / award / schedule file names
    2. photos on room and machine pages (ROOM_PAGES)
    3. other named images before camera and gallery file names (CAMERA)
    4. Inno Show pages before other pages
    5. first image of every page, then the second, and so on
    """
    rank_on_page, seen = {}, {}
    for im in images:
        rank_on_page[im["src"]] = seen[im["page"]] = seen.get(im["page"], -1) + 1

    def key(im: dict):
        name = im["src"].rsplit("/", 1)[-1] + " " + im.get("alt", "")
        return (0 if PRIORITY.search(name) else 1,
                0 if ROOM_PAGES.search(im["page"]) else 1,
                1 if CAMERA.search(im["src"].rsplit("/", 1)[-1]) else 0,
                0 if re.search(r"/innoshow\d*/", im["page"]) else 1,
                rank_on_page[im["src"]])
    return sorted(images, key=key)


def longest_side(path: Path) -> int:
    from PIL import Image
    try:
        with Image.open(path) as image:
            return max(image.size)
    except Exception:
        return 10 ** 6      # unknown: let the vision model try


def fetch(im: dict) -> Path:
    """Download the original if there is one and it is not huge, else the copy."""
    if im["original"]:
        try:
            path = download(im["original"])
            if path.stat().st_size <= MAX_BYTES:
                return path
            path.unlink()
        except requests.RequestException:
            pass
    return download(im["src"])


def describe(path: Path, effort: str = None, detail: str = None,
             max_side: int = None) -> tuple[str, int | None, dict]:
    """Send one image to the vision model.

    Returns (description, tokens used, the reply's usage details). Same
    request as describe_image in bot/llm.py, but reads the raw reply,
    because the gateway's remaining_quota_tokens field is not part of
    the normal response object.
    """
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data_bytes = path.read_bytes()
    max_side = max_side or VISION_MAX_SIDE
    if max_side:
        import io
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(path).convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        data_bytes, mime = buffer.getvalue(), "image/jpeg"
    b64 = base64.b64encode(data_bytes).decode()
    effort = effort or REASONING_EFFORT
    raw = _client(VISION_URL).chat.completions.with_raw_response.create(
        model=VISION_DEPLOYMENT,
        **({"reasoning_effort": effort} if effort else {}),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": DESCRIPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}",
                                                "detail": detail or IMAGE_DETAIL}},
        ]}],
    )
    data = json.loads(raw.text)
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    return text, usage.get("total_tokens"), usage


def describe_one(src: str, effort: str = None, detail: str = None,
                 max_side: int = None) -> str:
    """Describe a single image and print it. Nothing is cached or indexed.

    For trying a prompt: read the output and check the question can be
    answered from it before spending an hour on every image.
    """
    start = time.time()
    path = download(src)
    text, used, usage = describe(path, effort, detail, max_side)
    path.unlink()
    out = usage.get("completion_tokens") or 0
    thinking = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    print(f"{src}\n{'-' * 60}\n{text or '(EMPTY REPLY)'}\n{'-' * 60}")
    print(f"{len(text)} chars   {time.time() - start:.0f}s   "
          f"effort: {effort or REASONING_EFFORT or 'default'}   detail: {detail or IMAGE_DETAIL}   "
          f"max side: {max_side or VISION_MAX_SIDE or 'as downloaded'}")
    print(f"tokens used: {used}  =  input (prompt + image) {usage.get('prompt_tokens')}"
          f"  +  thinking {thinking}  +  written answer {out - thinking}")
    return text


def describe_all(images: list[dict], limit: int = None) -> dict:
    """Describe every image once, cache the result, never regenerate.

    Ingestion is free, but not if you redo it every time you change a
    line downstream. Key the cache by image URL. An icon-sized file is
    cached as "" so it is not downloaded again.
    """
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    todo = [im for im in images if im["src"] not in cache]
    if OCR_PATH.exists():
        ocr = json.loads(OCR_PATH.read_text(encoding="utf-8"))
        texty = [im for im in todo if len(ocr.get(im["src"], "")) >= OCR_ENOUGH]
        todo = [im for im in todo if len(ocr.get(im["src"], "")) < OCR_ENOUGH]
        print(f"{len(texty)} images left to OCR alone (text-heavy), "
              f"{len(todo)} need the vision model")
    else:
        print("No data/ocr.json yet: every image goes to the vision model. "
              "Run python -m build.ocr first to save quota.")
    todo = by_priority(todo)[:limit]
    print(f"{len(images)} images, {len(cache)} cached, describing {len(todo)}")
    if not todo:
        return cache

    lock, stop = threading.Lock(), threading.Event()
    state = {"done": 0, "tokens": 0, "why": ""}
    _client(VISION_URL)     # build the client once, before the threads start

    def save():
        CACHE_PATH.parent.mkdir(exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        tmp.replace(CACHE_PATH)

    def work(im: dict):
        if stop.is_set():
            return
        src, path = im["src"], None
        try:
            path = fetch(im)
            if path.stat().st_size < MIN_BYTES or longest_side(path) < MIN_PIXELS:
                text, used, left = "", 0, None
            else:
                text, used, _ = describe(path)
                if not text:
                    raise ValueError("empty reply")
        except Exception as exc:
            print("failed:", src[-70:], str(exc)[:120])
            message = str(exc).lower()
            # Every gateway error body carries a "remaining_quota_tokens"
            # field, so match the 403 wording, not the word "quota".
            if "quota is exceeded" in message:
                state["why"] = "the key's token quota is used up"
                stop.set()
            elif "content safety" in message or "content_filter" in message:
                # Refused by the safety filter (a medical photo, say). It
                # will be refused again, so cache it as empty.
                with lock:
                    cache[src] = ""
            elif "429" in str(exc):
                time.sleep(30)      # rate limited: let the gateway recover
            return
        finally:
            if path and path.exists():
                path.unlink()

        with lock:
            cache[src] = text
            state["done"] += 1
            state["tokens"] += used or 0
            if state["tokens"] >= TOKEN_BUDGET and not stop.is_set():
                state["why"] = f"this run used its {TOKEN_BUDGET:,} token budget"
                stop.set()
            if state["done"] % SAVE_EVERY == 0:
                save()
                print(f"{state['done']}/{len(todo)}   tokens used: {state['tokens']:,}")

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for future in as_completed([pool.submit(work, im) for im in todo]):
            future.result()
    except KeyboardInterrupt:
        print("stopping, saving what is done...")
        stop.set()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        with lock:
            save()

    if state["why"]:
        print(f"STOPPED: {state['why']}. Run the same command again later to continue.")
    print(f"{len(cache)} descriptions cached   tokens used this run: {state['tokens']:,}")
    return cache


# Words in a file name that say nothing about the picture.
FILE_NOISE = {"img", "dsc", "scaled", "resized", "copy", "final", "png", "jpg", "jpeg",
              "landscape", "portrait", "web", "new", "edit", "edited", "photo", "image", "pic"}
_vocab = None


def vocabulary() -> set:
    """Every word of 3+ letters in the scraped pages, for splitting file names."""
    global _vocab
    if _vocab is None:
        pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
        _vocab = {w for p in pages for w in re.findall(r"[a-z]{3,}", p.get("text", "").lower())}
    return _vocab


def split_glued(word: str) -> list[str]:
    """"trainedmember" -> ["trained", "member"]; a word it cannot split stays whole.

    Fewest pieces wins, each piece a word from the pages, so "selfservice"
    is "self service" and not "sel f service".
    """
    vocab = vocabulary()
    if len(word) < 7 or word in vocab:
        return [word]
    best = {0: []}
    for end in range(1, len(word) + 1):
        for start in range(max(0, end - 20), end - 2):
            if start in best and word[start:end] in vocab:
                pieces = best[start] + [word[start:end]]
                if end not in best or len(pieces) < len(best[end]):
                    best[end] = pieces
    return best.get(len(word), [word])


def file_words(src: str) -> str:
    """Words in an image's file name: the only place a site icon says what it is.

    The Wet Lab's "Trained Members Only" icon is trainedmember-150x150.png;
    its description says "white person on a blue square" and nothing a
    question about trained members would match.
    """
    name = Path(SIZE.sub("", src.split("?")[0])).stem
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)     # RoboMaster -> Robo Master
    words = []
    for token in re.split(r"[^A-Za-z0-9]+", name):
        low = token.lower()
        if (len(low) < 2 or low in FILE_NOISE or re.fullmatch(r"\d+x\d+", low)
                or (low.isdigit() and not re.fullmatch(r"20\d\d", low))):
            continue
        words += split_glued(low) if low.isalpha() else [low]
    if not any(len(w) >= 3 and w.isalpha() for w in words):
        return ""       # camera names like IMG_5770 say nothing
    return " ".join(words)


def index_descriptions(descriptions: dict, images: list[dict], fresh: bool = False) -> None:
    """Add image chunks to the same store as the scraped text.

    One chunk per image that has a vision description, OCR text from
    build/ocr.py, colours from build/colours.py, or any of these. Only chunks that are new or whose text changed
    are embedded, because embedding thousands of chunks again on every
    run spends the key's token quota for nothing. Each chunk carries the
    same metadata fields as a text chunk of its page.
    """
    pages = {p["url"]: p for p in
             json.loads(Path("data/pages.json").read_text(encoding="utf-8"))}
    ocr = json.loads(OCR_PATH.read_text(encoding="utf-8")) if OCR_PATH.exists() else {}
    colours = (json.loads(COLOURS_PATH.read_text(encoding="utf-8"))
               if COLOURS_PATH.exists() else {})
    local = (json.loads(LOCAL_PATH.read_text(encoding="utf-8"))
             if LOCAL_PATH.exists() else {})
    # A reply missing a section ran off into repetition; leave it out.
    local = {k: v for k, v in local.items() if all(label in v for label in SECTIONS)}
    # Every page a picture appears on, at any size: answer.py reads the
    # images of the pages it retrieved, and a picture is indexed only once.
    shown_on = {}
    for r in json.loads(Path("data/images.json").read_text(encoding="utf-8")):
        shown_on.setdefault(SIZE.sub("", r["src"].split("?")[0]), set()).add(r["page"])
    chunks, per_page = {}, {}
    for im in images:
        text, read = descriptions.get(im["src"]), ocr.get(im["src"], "")[:OCR_MAX_CHARS]
        if not text and local.get(im["src"]):
            text = LOCAL_NOTE + "\n" + local[im["src"]]
        if not text and not read and not colours.get(im["src"]):
            continue
        page = pages.get(im["page"], {"url": im["page"], "title": "", "text": ""})
        n = per_page[im["page"]] = per_page.get(im["page"], -1) + 1
        # Alt text and captions are already text and cost nothing to index.
        # The page title says which event or project the image belongs to,
        # which the image itself often does not. OCR copies small print
        # exactly, which the vision model sometimes marks [unreadable].
        parts = [page.get("title", ""),
                 "Image on this page. " + " ".join(filter(None, [im.get("alt"), im.get("caption")]))]
        if file_words(im["src"]):
            parts.append("File name: " + file_words(im["src"]))
        if read:
            parts.append("Text in the image: " + read)
        if text:
            parts.append(text)
        # OCR-only posters have no colours; build/colours.py adds them.
        if colours.get(im["src"]):
            parts.append("COLOURS: " + colours[im["src"]])
        chunks["img_" + hashlib.md5(im["src"].encode()).hexdigest()] = ("\n".join(parts), {
            "url":       im["page"],
            "title":     page.get("title", ""),
            "position":  IMAGE_POSITION + n,
            "kind":      "image",
            "year":      guess_year(page),
            "page_type": page.get("page_type", ""),
            "site":      page.get("site", ""),
            "image":     im["src"],
            "pages":     " ".join(sorted(shown_on.get(SIZE.sub("", im["src"].split("?")[0]),
                                                  {im["page"]}))),
        })

    store = get_store(reset=False)
    old = store.get(where={"kind": "image"}, include=["documents", "metadatas", "embeddings"])
    old_ids = set(old["ids"])
    unchanged = {i for i, doc, meta in zip(old["ids"], old["documents"], old["metadatas"])
                 if i in chunks and chunks[i] == (doc, meta)}
    stale = [i for i in old["ids"] if i not in chunks]
    for i in range(0, len(stale), 1000):
        store.delete(ids=stale[i:i + 1000])
    # A chunk whose text grew a line (a file name, colours) keeps its old
    # embedding unless fresh=True: the vector barely moves, BM25 sees the
    # new words at once, and 4,000 re-embeddings would cost quota.
    vectors = dict(zip(old["ids"], old["embeddings"]))
    changed = [i for i in chunks if i in old_ids and i not in unchanged]
    if fresh:
        store.delete(ids=changed)
    else:
        for i in range(0, len(changed), 500):
            batch = changed[i:i + 500]
            store.update(ids=batch, documents=[chunks[j][0] for j in batch],
                         metadatas=[chunks[j][1] for j in batch],
                         embeddings=[vectors[j] for j in batch])
    new = [i for i in chunks if i not in old_ids or (fresh and i in changed)]
    add_to_store(store, [chunks[i][0] for i in new], [chunks[i][1] for i in new], ids=new)
    print(f"image chunks: {len(chunks)} total, {len(new)} embedded now, "
          f"{0 if fresh else len(changed)} text updated, {len(unchanged)} unchanged, "
          f"{len(stale)} removed")


def merge(other_path: str, images: list[dict]) -> None:
    """Add a teammate's descriptions.json to ours, without overwriting ours.

    Their keys may be a different size of the same picture
    (photo-1024x683.jpg vs photo.jpg), so both are matched by the name
    with the size removed.
    """
    ours = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    theirs = json.loads(Path(other_path).read_text(encoding="utf-8"))
    by_name = {SIZE.sub("", im["src"].split("?")[0]): im["src"] for im in images}
    added, unknown = 0, 0
    for src, text in theirs.items():
        key = by_name.get(SIZE.sub("", src.split("?")[0]))
        if key is None:
            unknown += 1
        elif text and not ours.get(key):
            ours[key] = text
            added += 1
    CACHE_PATH.write_text(json.dumps(ours, indent=1), encoding="utf-8")
    print(f"added {added}, already had {len(theirs) - added - unknown}, "
          f"not in our image list {unknown}. Now {len(ours)} cached.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "test":
        # python -m build.images test [url] [minimal|low|medium|high]
        #                             [detail-low|detail-high] [max side, e.g. 768]
        effort = next((a for a in args[1:] if a in EFFORTS), None)
        detail = next((a[7:] for a in args[1:] if a.startswith("detail-")), None)
        side = next((int(a) for a in args[1:] if a.isdigit()), None)
        url = next((a for a in args[1:] if a.startswith("http")), TEST_IMAGE)
        describe_one(url, effort, detail, side)
        sys.exit()

    images = pick_images(json.loads(Path("data/images.json").read_text(encoding="utf-8")))

    if args and args[0] == "merge":
        # python -m build.images merge path/to/their/descriptions.json
        merge(args[1], images)
        sys.exit()

    if args and args[0] == "index":
        # python -m build.images index [fresh]   fresh: re-embed changed chunks too
        descriptions = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        index_descriptions(descriptions, images, fresh="fresh" in args)
        sys.exit()

    limit = int(args[0]) if args and args[0].isdigit() else None
    word = next((a for a in args if not a.isdigit()), "")
    subset = [im for im in images if word in im["page"]]
    descriptions = describe_all(subset, limit)
    index_descriptions(descriptions, images)
