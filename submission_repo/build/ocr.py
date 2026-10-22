"""Read the text in every image with the Mac's built-in OCR. No API, no quota.

The vision model (build/images.py) costs about 4,300 tokens per image and
the key's quota allows a few hundred images a day. OCR is free, takes
about a second per image, and copies small print exactly, which is what
most Lv3 questions ask about ("According to the Nocodile AI poster, how
many students...?"). It cannot count objects or say what is next to
what; that is still the vision model's job.

Needs macOS and the ocrmac package (pip install ocrmac). Build-time only:
the bot never imports this file, so the grader does not need ocrmac.

Run from submission_repo (any network, no VPN needed):

    python -m build.ocr                 read every image, then index
    python -m build.ocr 50              read up to 50 new images, then index

Indexing embeds the new chunks, which does need HKU Wi-Fi or HKUVPN. If
you are not connected, the OCR results are still saved; run
`python -m build.images index` later.
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from build.images import (CACHE_PATH, MIN_BYTES, OCR_PATH, by_priority, download,
                          index_descriptions, pick_images)

WORKERS = 8         # mostly waiting for downloads
SAVE_EVERY = 100
MIN_CONFIDENCE = 0.3    # below this a "line" is usually a smudge read as letters
# The sites have some Chinese text (award news, sup_mat). Listing English
# first keeps it the main language.
LANGUAGES = ["en-US", "zh-Hant"]


def read_text(path: Path) -> str:
    """All text lines in one image, top to bottom, one per line."""
    from ocrmac import ocrmac     # imported here so images.py works without it
    from PIL import Image
    # Print-ready posters are often CMYK JPEGs, which ocrmac cannot
    # convert on its own ("cannot write mode CMYK as PNG").
    # Some A1 posters were uploaded at print resolution (243 megapixels,
    # 117 MB). Pillow refuses those by default; shrinking them to 4000 px
    # keeps the text readable and the OCR fast.
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path).convert("RGB")
    image.thumbnail((4000, 4000))
    lines = ocrmac.OCR(image, recognition_level="accurate",
                       language_preference=LANGUAGES).recognize()
    return "\n".join(text for text, confidence, _ in lines if confidence >= MIN_CONFIDENCE)


def ocr_all(images: list[dict], limit: int = None) -> dict:
    """OCR every image once and cache the result in data/ocr.json, keyed by URL.

    An image with no text, or too small to be more than an icon, is
    cached as "" so it is not downloaded again.
    """
    cache = json.loads(OCR_PATH.read_text(encoding="utf-8")) if OCR_PATH.exists() else {}
    todo = by_priority([im for im in images if im["src"] not in cache])[:limit]
    print(f"{len(images)} images, {len(cache)} already read, reading {len(todo)}")
    lock, done = threading.Lock(), [0]

    def save():
        tmp = OCR_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OCR_PATH)

    def work(im: dict):
        path = None
        try:
            # The resized copy the page shows, not the original: originals
            # downloaded at about 2 KB/s (a 2.8 MB poster took 10 minutes),
            # copies at about 110 KB/s, and OCR read every test answer on
            # the copies too (Nocodile 723x1024, Pitching-Winner 1024x724).
            path = download(im["src"])
            text = read_text(path) if path.stat().st_size >= MIN_BYTES else ""
        except Exception as exc:
            print("failed:", im["src"][-70:], str(exc)[:120])
            return
        finally:
            if path and path.exists():
                path.unlink()
        with lock:
            cache[im["src"]] = text
            done[0] += 1
            if done[0] % SAVE_EVERY == 0:
                save()
                print(f"{done[0]}/{len(todo)}")

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for future in as_completed([pool.submit(work, im) for im in todo]):
            future.result()
    except KeyboardInterrupt:
        print("stopping, saving what is done...")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        with lock:
            save()

    print(f"{len(cache)} images read, {sum(1 for t in cache.values() if t)} with text")
    return cache


if __name__ == "__main__":
    args = sys.argv[1:]
    images = pick_images(json.loads(Path("data/images.json").read_text(encoding="utf-8")))
    ocr_all(images, int(args[0]) if args and args[0].isdigit() else None)

    descriptions = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    try:
        index_descriptions(descriptions, images)
    except Exception as exc:
        print(f"OCR saved, but indexing failed ({str(exc)[:100]}).\n"
              f"Connect to HKU Wi-Fi or HKUVPN and run: python -m build.images index")
