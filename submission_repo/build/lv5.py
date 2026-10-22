"""Photos taken inside the Innovation Wing. Lv5.

Lv5 questions ask what you would only know from being there: how many
chairs in the brainstorming area, what a sign in the open event area
says, the colour of a bench. The websites do not show this, so each
photo is described once and indexed as its own chunk.

Name each photo after its zone, then anything: "brainstorming__wall.jpg",
"makerspace__printers.jpg". The zone goes into the chunk text, because a
question names the zone ("in the open event area") and the photo itself
rarely says where it is.

Every photo is described twice over: once whole (objects, counts,
colours, layout) and once as four overlapping quarters, enlarged, for
the text on signs. A sign that is a few pixels high in the whole photo
is lost when the model shrinks it, and is readable in a quarter.

Run from submission_repo on HKU Wi-Fi or HKUVPN:

    python -m build.lv5            describe new photos, then index all
    python -m build.lv5 index      index the cache only

Chunks are kind "photo", not "image": `python -m build.images index`
removes every "image" chunk it did not make, and would delete these.
"""
import base64
import hashlib
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bot.llm import VISION_DEPLOYMENT, VISION_URL, _client
from bot.store import add_to_store, get_store
from build.images import REASONING_EFFORT, TMP_DIR

PHOTO_DIRS = [Path("data/lv5_photos"), Path("../Labs/images")]
CACHE_PATH = Path("data/lv5_descriptions.json")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
WHOLE_SIDE = 2048       # longest side sent for the whole photo
TILE_SIDE = 1536        # longest side sent for each quarter
OVERLAP = 0.15          # quarters overlap so text on a cut line is whole in one
PHOTO_POSITION = 200_000    # images.py uses 100_000+, text chunks 0+

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

WHOLE_PROMPT = """This photo was taken inside the Tam Wing Fan Innovation Wing at HKU, in the {zone}. Someone will later answer questions about this place from your notes alone, without the photo. Questions are like: how many chairs are there, what colour is the bench, what does the sign say, what is on the wall at the back, what machines are there.

Write plain text in these sections:
TEXT: every piece of visible text exactly as written (signs, wall lettering, posters, labels, screens, machine brands). Write [unreadable] for any part you cannot read. Never guess.
OBJECTS: every kind of furniture, machine, screen and fitting, each with an exact count and its colour, e.g. "12 orange cube stools". Count carefully, item by item. If some are hidden, say "at least".
COLOURS: walls, floor, ceiling, furniture, signs.
LAYOUT: what is where (back wall, left, right, centre, front), what is next to what.

Only what is visible. No summary, no preamble."""

TILE_PROMPT = """This is one quarter of a photo taken in the {zone} of the HKU Innovation Wing. Copy every piece of visible text exactly as written: signs, notices, wall lettering, posters, labels, screens, brand names. For each, say in a few words what it is on (e.g. "white notice on the glass door: ..."). Write [unreadable] for any part you cannot read. Never guess. If there is no text, write "none"."""


def zone_of(path: Path) -> str:
    """"makerspace__door.jpg" -> "Makerspace A"; "wet_lab__sign.jpg" -> "Wet Lab"."""
    name = path.stem.lower()
    for key, zone in ZONES.items():
        if key in name:
            return zone
    if "__" in name:
        return name.split("__")[0].replace("_", " ").replace("-", " ").title()
    return "Innovation Wing"


def photos() -> list[Path]:
    found = []
    for folder in PHOTO_DIRS:
        if folder.exists():
            found += sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    return found


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


def describe_photo(path: Path) -> str:
    zone = zone_of(path)
    tiles = quarters(path)
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            whole = pool.submit(ask, path, WHOLE_PROMPT.format(zone=zone), WHOLE_SIDE)
            parts = [pool.submit(ask, t, TILE_PROMPT.format(zone=zone), TILE_SIDE) for t in tiles]
            whole, parts = whole.result(), [p.result() for p in parts]
    finally:
        for t in tiles:
            t.unlink(missing_ok=True)
    signs = "\n".join(p for p in parts if p and p.lower() != "none")
    return f"{whole}\nTEXT IN CLOSE-UPS:\n{signs}"


PHOTOS_AT_ONCE = 3      # x 5 calls each = 15 requests in flight


def describe_all() -> dict:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    todo = [p for p in photos() if p.name not in cache]
    print(f"{len(cache)} photos cached, describing {len(todo)}")
    lock = threading.Lock()

    def work(path: Path):
        try:
            text = describe_photo(path)
        except Exception as exc:
            # A quota stop or a dropped VPN: keep what is done, rerun later.
            print(f"failed: {path.name}: {str(exc)[:150]}")
            return
        with lock:
            cache[path.name] = text
            CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
            print(f"done {path.name} ({zone_of(path)})")

    with ThreadPoolExecutor(max_workers=PHOTOS_AT_ONCE) as pool:
        list(pool.map(work, todo))
    return cache


def index_photos(cache: dict) -> None:
    """Replace every photo chunk with one chunk per cached photo."""
    texts, metas, ids = [], [], []
    for n, (name, text) in enumerate(sorted(cache.items())):
        zone = zone_of(Path(name))
        texts.append(f"Photo taken inside the Tam Wing Fan Innovation Wing, in the {zone}.\n{text}")
        metas.append({
            "url":       "innowing-photo://" + zone.lower().replace(" ", "-"),
            "title":     f"{zone} (photo)",
            "position":  PHOTO_POSITION + n,
            "kind":      "photo",
            "year":      "",
            "page_type": "photo",
            "site":      "innowing",
            "image":     name,
            "zone":      zone,
        })
        ids.append("photo_" + hashlib.md5(name.encode()).hexdigest())
    store = get_store(reset=False)
    old = store.get(where={"kind": "photo"})["ids"]
    if old:
        store.delete(ids=old)
    add_to_store(store, texts, metas, ids=ids)
    print(f"photo chunks: {len(texts)} indexed, {len(old)} replaced. Store now {store.count()}.")


if __name__ == "__main__":
    if sys.argv[1:] == ["index"]:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        cache = describe_all()
    index_photos(cache)
