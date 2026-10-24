"""Photos taken inside the Innovation Wing. Lv5.

Lv5 questions ask what you would only know from being there: how many
chairs in the brainstorming area, what a sign in the open event area
says, the colour of a bench. The websites do not show this, so each
photo is described once and indexed as its own chunk.

Two ways in:

  data/lv5_raw/<zone folder>/<spot folder>/IMG_1234.JPG, with a notes.txt
      in each folder (the team's survey of 24 Oct). The zone comes from
      the top folder (ZONE_FOLDERS). The spot's notes go into the prompt
      so the model knows where it is standing, and every notes.txt is
      also indexed as its own chunk: they hold counts made by hand
      ("53 stools"), which beat any count a model makes from a photo.

  data/lv5_photos/zone__anything.jpg (and ../Labs/images): the zone comes
      from the file name ("brainstorming__wall.jpg").

Every room photo is described twice over: once whole (objects, counts,
colours, layout) and once as four overlapping quarters, enlarged, for
the text on signs. A sign that is a few pixels high in the whole photo
is lost when the model shrinks it, and is readable in a quarter. A
close-up of a sign or a poster is readable whole, so it gets one call:
the poster folders, a spot folder with "sign" in its name, or a file
named with sign, poster, label, close, notice, board, screen, plate or
text after the "__".

Run from submission_repo on HKU Wi-Fi or HKUVPN:

    python -m build.lv5            describe new photos, then index all
    python -m build.lv5 index      index the cache and notes only

Chunks are kind "photo", not "image": `python -m build.images index`
removes every "image" chunk it did not make, and would delete these.
"""
import base64
import hashlib
import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bot.llm import VISION_DEPLOYMENT, VISION_URL, _client
from bot.store import add_to_store, get_store
from build.images import REASONING_EFFORT, TMP_DIR

PHOTO_DIRS = [Path("data/lv5_photos"), Path("../Labs/images")]
RAW_DIR = Path("data/lv5_raw")
CACHE_PATH = Path("data/lv5_descriptions.json")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
WHOLE_SIDE = 2048       # longest side sent for the whole photo
TILE_SIDE = 1536        # longest side sent for each quarter
OVERLAP = 0.15          # quarters overlap so text on a cut line is whole in one
PHOTO_POSITION = 200_000    # images.py uses 100_000+, text chunks 0+
NOTES_POSITION = 300_000
# Extra passes over the same photos by the local model machine
# (local_vlm/LV5_BRIEF.md): exact OCR text at full resolution, and one
# description per poster on a wall of posters. Both optional.
LOCAL_OCR = Path("local_vlm/lv5_ocr_local.json")
LOCAL_POSTERS = Path("local_vlm/lv5_posters_local.json")
OCR_MAX = 2500          # characters of OCR text kept per photo

# Filename keyword -> the zone's name as the organisers write it.
ZONES = {
    "brainstorm":       "Brainstorming area",
    "makerspace":       "Makerspace A",
    "open event":       "Open event area",
    "open_event":       "Open event area",
    "openevent":        "Open event area",
    "digital learning": "Digital Learning Studio",
    "digital_learning": "Digital Learning Studio",
    "dls":              "Digital Learning Studio",
    "gallery":          "Photo gallery near the entrance",
    "entrance":         "Photo gallery near the entrance",
}
# Top folder under data/lv5_raw -> zone.
ZONE_FOLDERS = {
    "makerspace_a":               "Makerspace A",
    "brainstorming":              "Brainstorming area",
    "digital_learning_studio":    "Digital Learning Studio",
    "open_event_area":            "Open event area",
    "lg_corridor_posters":        "LG/F corridors",
    "staircase_lg_to_gf_posters": "staircase between LG/F and G/F",
}
POSTER_FOLDERS = {"lg_corridor_posters", "staircase_lg_to_gf_posters"}
# The same four photos of the open event area were also put in Makerspace
# A's folder (the "OPEN EVENT AREA" stand is in them). Indexed there, they
# would give Makerspace A V-shaped tables and grey chairs.
MISFILED = {f"makerspace_a/01_inside_wide/IMG_{n}.JPG" for n in (1389, 1390, 1391, 1392)}

WHOLE_PROMPT = """This photo was taken inside the Tam Wing Fan Innovation Wing at HKU, in the {zone}. Someone will later answer questions about this place from your notes alone, without the photo. Questions are like: how many chairs are there, what colour is the bench, what does the sign say, what is on the wall at the back, what machines are there.
{notes}
Write plain text in these sections:
TEXT: every piece of visible text exactly as written (signs, wall lettering, posters, labels, screens, machine brands). Write [unreadable] for any part you cannot read. Never guess.
OBJECTS: every kind of furniture, machine, screen and fitting, each with an exact count and its colour, e.g. "12 orange cube stools". Count carefully, item by item. If some are hidden, say "at least".
COLOURS: walls, floor, ceiling, furniture, signs, and the main colours of any poster.
LAYOUT: what is where (back wall, left, right, centre, front), what is next to what.

Only what is visible. No summary, no preamble."""

NOTES_HINT = """The team that took the photo wrote this about the spot, to tell you where you are (describe only what you can see in the photo):
{text}
"""

TILE_PROMPT = """This is one quarter of a photo taken in the {zone} of the HKU Innovation Wing. Copy every piece of visible text exactly as written: signs, notices, wall lettering, posters, labels, screens, brand names. For each, say in a few words what it is on (e.g. "white notice on the glass door: ..."). Write [unreadable] for any part you cannot read. Never guess. If there is no text, write "none"."""

CLOSE_UP = re.compile(r"sign|poster|label|close|notice|board|screen|plate|text", re.I)


def zone_of(key: str) -> str:
    """"makerspace_a/01_inside_wide/IMG_1.JPG" or "makerspace__door.jpg" -> "Makerspace A"."""
    if "/" in key:
        return ZONE_FOLDERS.get(key.split("/")[0], "Innovation Wing")
    name = Path(key).stem.lower()
    for word, zone in ZONES.items():
        if word in name:
            return zone
    if "__" in name:
        return name.split("__")[0].replace("_", " ").replace("-", " ").title()
    return "Innovation Wing"


def spot_of(key: str) -> str:
    """"brainstorming/back_left_stair_step_hku_sign/IMG_1.JPG" -> "back left stair step hku sign"."""
    parts = key.split("/")
    if len(parts) < 3:
        return ""
    return re.sub(r"^\d+_", "", parts[1]).replace("_", " ").replace("-", " ")


def photos() -> dict[str, Path]:
    """Cache key -> file. Old flat photos keep their bare file name as key."""
    found = {}
    for folder in PHOTO_DIRS:
        if folder.exists():
            found.update({p.name: p for p in sorted(folder.iterdir())
                          if p.suffix.lower() in EXTENSIONS})
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.rglob("*")):
            key = p.relative_to(RAW_DIR).as_posix()
            if p.suffix.lower() in EXTENSIONS and key not in MISFILED:
                found[key] = p
    return found


def notes_files() -> list[Path]:
    """Every notes.txt under data/lv5_raw (not the *_original_* drafts)."""
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.rglob("notes.txt") if p.read_text(encoding="utf-8").strip())


def needs_quarters(key: str) -> bool:
    if "/" in key:
        parts = key.split("/")
        return parts[0] not in POSTER_FOLDERS and "sign" not in parts[1].lower()
    return not CLOSE_UP.search(Path(key).stem.split("__")[-1])


def quarters(path: Path) -> list[Path]:
    """Four overlapping quarters of the photo, saved under data/_tmp."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = image.size
    cw, ch = int(w * (0.5 + OVERLAP / 2)), int(h * (0.5 + OVERLAP / 2))
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (x, y) in enumerate([(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)]):
        tile = image.crop((x, y, x + cw, y + ch))
        tile.thumbnail((TILE_SIDE, TILE_SIDE))
        p = TMP_DIR / f"{hashlib.md5(str(path).encode()).hexdigest()}_q{i}.jpg"
        tile.save(p, "JPEG", quality=90)
        out.append(p)
    return out


def ask(path: Path, prompt: str, side: int) -> str:
    """One vision call with this file's prompt, the photo shrunk to `side`.

    Same request as describe() in images.py, which always sends its own
    DESCRIPTION_PROMPT.
    """
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    r = _client(VISION_URL).chat.completions.create(
        model=VISION_DEPLOYMENT, reasoning_effort=REASONING_EFFORT,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    print(f"  {path.name}: {r.usage.total_tokens} tokens")
    return (r.choices[0].message.content or "").strip()


def describe_photo(key: str, path: Path) -> str:
    zone = zone_of(key)
    notes_path = path.parent / "notes.txt"
    notes = notes_path.read_text(encoding="utf-8").strip() if "/" in key and notes_path.exists() else ""
    prompt = WHOLE_PROMPT.format(zone=zone, notes=NOTES_HINT.format(text=notes) if notes else "")
    tiles = quarters(path) if needs_quarters(key) else []
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            whole = pool.submit(ask, path, prompt, WHOLE_SIDE)
            parts = [pool.submit(ask, t, TILE_PROMPT.format(zone=zone), TILE_SIDE) for t in tiles]
            whole, parts = whole.result(), [p.result() for p in parts]
    finally:
        for t in tiles:
            t.unlink(missing_ok=True)
    signs = "\n".join(p for p in parts if p and p.lower() != "none")
    return f"{whole}\nTEXT IN CLOSE-UPS:\n{signs}" if signs else whole


PHOTOS_AT_ONCE = 3      # x 5 calls each = 15 requests in flight


def describe_all() -> dict:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    todo = {k: p for k, p in photos().items() if k not in cache}
    print(f"{len(cache)} photos cached, describing {len(todo)} "
          f"({sum(needs_quarters(k) for k in todo)} with quarters)")
    lock, stop = threading.Lock(), threading.Event()

    def work(item):
        key, path = item
        if stop.is_set():
            return
        try:
            text = describe_photo(key, path)
        except Exception as exc:
            # A quota stop or a dropped VPN: keep what is done, rerun later.
            print(f"failed: {key}: {str(exc)[:150]}")
            if "quota is exceeded" in str(exc).lower():
                stop.set()
            return
        with lock:
            cache[key] = text
            CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
            print(f"done {key} ({zone_of(key)})  [{len(cache)} cached]")

    with ThreadPoolExecutor(max_workers=PHOTOS_AT_ONCE) as pool:
        list(pool.map(work, todo.items()))
    if stop.is_set():
        print("STOPPED: the key's token quota is used up. Run again after it resets.")
    return cache


def index_photos(cache: dict) -> None:
    """Replace every photo chunk with one chunk per cached photo and per notes.txt."""
    texts, metas, ids = [], [], []

    def add(text: str, zone: str, title: str, position: int, name: str, id_: str):
        texts.append(text)
        metas.append({
            "url":       "innowing-photo://" + zone.lower().replace(" ", "-").replace("/", ""),
            "title":     title,
            "position":  position,
            "kind":      "photo",
            "year":      "",
            "page_type": "photo",
            "site":      "innowing",
            "image":     name,
            "zone":      zone,
        })
        ids.append(id_)

    keys = set(photos())
    ocr = json.loads(LOCAL_OCR.read_text(encoding="utf-8")) if LOCAL_OCR.exists() else {}
    posters = (json.loads(LOCAL_POSTERS.read_text(encoding="utf-8"))
               if LOCAL_POSTERS.exists() else {})
    for n, (key, text) in enumerate(sorted(cache.items())):
        if key in MISFILED or ("/" in key and key not in keys):
            continue
        zone, spot = zone_of(key), spot_of(key)
        where = f"in the {zone}" + (f", {spot}" if spot else "")
        if posters.get(key):
            text += ("\nPOSTER BY POSTER (rough notes by a small local model):\n"
                     + "\n".join(posters[key]))
        if ocr.get(key):
            text += "\nTEXT READ BY OCR (exact letters, in no particular order):\n" + ocr[key][:OCR_MAX]
        add(f"Photo taken inside the Tam Wing Fan Innovation Wing, {where}.\n{text}",
            zone, f"{zone} (photo)", PHOTO_POSITION + n, key,
            "photo_" + hashlib.md5(key.encode()).hexdigest())
    for n, path in enumerate(notes_files()):
        key = path.relative_to(RAW_DIR).as_posix()
        zone, spot = ZONE_FOLDERS.get(key.split("/")[0], "Innovation Wing"), spot_of(key)
        where = f"the {zone}" + (f", {spot}" if spot else "")
        add(f"Notes written on site by our team, counted by hand: {where}.\n"
            + path.read_text(encoding="utf-8").strip(),
            zone, f"{zone} (notes counted on site)", NOTES_POSITION + n, key,
            "notes_" + hashlib.md5(key.encode()).hexdigest())

    store = get_store(reset=False)
    old = store.get(where={"kind": "photo"})["ids"]
    if old:
        store.delete(ids=old)
    add_to_store(store, texts, metas, ids=ids)
    print(f"photo chunks: {len(texts)} indexed ({len(notes_files())} notes), "
          f"{len(old)} replaced. Store now {store.count()}.")


if __name__ == "__main__":
    if sys.argv[1:] == ["index"]:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    else:
        cache = describe_all()
    index_photos(cache)
