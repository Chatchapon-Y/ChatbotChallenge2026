"""The colours of every poster and sign, for "what colour is..." questions.

A poster whose OCR text was long enough skipped the vision model (see
OCR_ENOUGH in build/images.py), so its chunk has every word but no
colour: the Robot Arm workshop poster could not answer "what colour
scheme was used". Colour is the whole picture, not small print, so the
model gets a 512 px copy at detail "low" and minimal reasoning: a few
hundred tokens an image instead of ~2,000 for a full description.

Run from submission_repo on HKU Wi-Fi or HKUVPN:

    python -m build.colours 5        colour 5 images and print them (check the wording)
    python -m build.colours          colour every image left, within TOKEN_BUDGET
    python -m build.images index     then put the colours into the chunks

Results are cached in data/colours.json and never asked again.
"""
import base64
import io
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bot.llm import VISION_DEPLOYMENT, VISION_URL, _client
from build.images import (CACHE_PATH, COLOURS_PATH, OCR_ENOUGH, OCR_PATH, ROOM_PAGES,
                          by_priority, download, pick_images)

PROMPT = """Describe the colours of this image for a search index. Reply with one line in exactly this form:
background: <colours>; title/main text: <colours>; other main colours: <colours>; overall: <1-3 colour names>
Use plain colour names (blue, dark blue, light blue, navy, teal, green, yellow, orange, red, pink, purple, brown, beige, black, grey, white). If the image is a sign, icon or logo, give the colour of the sign itself. Nothing else."""

SIDE = 512              # detail "low" sees 512 px anyway; a smaller upload is faster
WORKERS = 8
TOKEN_BUDGET = 700_000  # the key's whole day is ~1.2M and the bot needs some
SAVE_EVERY = 25


def targets(images: list[dict]) -> list[dict]:
    """Images whose chunk says nothing about colour, posters first.

    1. posters OCR read fully (no description at all)
    2. described posters, slides and logos whose description has no colours
    3. other OCR-only images
    4. icons with no text at all on room and facility pages
       (the Wet Lab's "Trained Members Only" sign is one)
    """
    desc = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    ocr = json.loads(OCR_PATH.read_text(encoding="utf-8")) if OCR_PATH.exists() else {}
    tiers = [[], [], [], []]
    for im in images:
        d, read = desc.get(im["src"]), ocr.get(im["src"], "")
        if not d and len(read) >= OCR_ENOUGH:
            tiers[0].append(im)
        elif d and "COLOUR" not in d.upper() and re.search(
                r"TYPE:\s*(poster|slide|certificate|logo)", d, re.I):
            tiers[1].append(im)
        elif not d and read:
            tiers[2].append(im)
        elif not d and ROOM_PAGES.search(im["page"]):
            tiers[3].append(im)
    return [im for tier in tiers for im in by_priority(tier)]


def colour(path: Path) -> tuple[str, int]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    image.thumbnail((SIDE, SIDE))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    r = _client(VISION_URL).chat.completions.create(
        model=VISION_DEPLOYMENT, reasoning_effort="minimal",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                                "detail": "low"}},
        ]}],
    )
    return (r.choices[0].message.content or "").strip(), r.usage.total_tokens


def colour_all(todo: list[dict], show: bool = False) -> None:
    cache = json.loads(COLOURS_PATH.read_text(encoding="utf-8")) if COLOURS_PATH.exists() else {}
    todo = [im for im in todo if im["src"] not in cache]
    print(f"{len(cache)} cached, colouring {len(todo)}")
    lock, stop = threading.Lock(), threading.Event()
    state = {"done": 0, "tokens": 0, "why": ""}
    _client(VISION_URL)

    def save():
        tmp = COLOURS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(COLOURS_PATH)

    def work(im: dict):
        if stop.is_set():
            return
        path = None
        try:
            path = download(im["src"])
            text, used = colour(path)
        except Exception as exc:
            print("failed:", im["src"][-60:], str(exc)[:120])
            if "quota is exceeded" in str(exc).lower():
                state["why"] = "the key's token quota is used up"
                stop.set()
            return
        finally:
            if path and path.exists():
                path.unlink()
        with lock:
            cache[im["src"]] = text
            state["done"] += 1
            state["tokens"] += used or 0
            if show:
                print(f"{im['src'].rsplit('/', 1)[-1][:50]}  ({used} tokens)\n  {text}")
            if state["tokens"] >= TOKEN_BUDGET and not stop.is_set():
                state["why"] = f"this run used its {TOKEN_BUDGET:,} token budget"
                stop.set()
            if state["done"] % SAVE_EVERY == 0:
                save()
                print(f"{state['done']}/{len(todo)}   tokens used: {state['tokens']:,}   "
                      f"~{state['tokens'] // state['done']} per image")

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
    print(f"{len(cache)} colours cached   tokens used this run: {state['tokens']:,}")


if __name__ == "__main__":
    images = pick_images(json.loads(Path("data/images.json").read_text(encoding="utf-8")))
    todo = targets(images)
    args = sys.argv[1:]
    if args and args[0].isdigit():
        colour_all(todo[:int(args[0])], show=True)
    else:
        colour_all(todo)
    print("Now run: python -m build.images index")
