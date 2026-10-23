"""Describe website images with a local Ollama vision model.

Usage:
  python describe_local.py                     # full run over todo.json (resumes)
  python describe_local.py --test URL [URL..]  # quality check, prints replies,
                                               # writes descriptions_test.json
  python describe_local.py --redo-bad          # after the full run: retry replies missing a section
Options: --model qwen2.5vl:7b  --size 1024  --limit N  --dry-run (with --redo-bad)
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TODO = os.path.join(HERE, "todo.json")
PROMPT = open(os.path.join(HERE, "prompt.txt"), encoding="utf-8").read()
OUT = os.path.join(HERE, "descriptions_local.json")
SKIPPED = os.path.join(HERE, "skipped_local.json")  # icons/failed downloads, so a resume doesn't retry them
OLLAMA = "http://localhost:11434/api/chat"
# prompt.txt stays untouched; without this the 7b model often drops the section labels.
SYSTEM = ("Reply with exactly five labelled sections, each starting on a new line with its label: "
          "TEXT:, TYPE:, PLACE:, OBJECTS:, LAYOUT:.")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
MAX_ORIGINAL = 4 * 1024 * 1024
MIN_BYTES = 5000
MIN_SIDE = 250
SAVE_EVERY = 20


def load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def download(url, max_bytes=None):
    """Return bytes, or None if failed or larger than max_bytes. Kept in memory only."""
    try:
        with requests.get(url, headers=UA, timeout=30, stream=True) as r:
            r.raise_for_status()
            size = r.headers.get("Content-Length")
            if max_bytes and size and int(size) > max_bytes:
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if max_bytes and buf.tell() > max_bytes:
                    return None
            return buf.getvalue()
    except requests.RequestException as e:
        print(f"    download failed: {url} ({e})")
        return None


def fetch(entry):
    data = None
    if entry.get("original"):
        data = download(entry["original"], MAX_ORIGINAL)
    if data is None:
        data = download(entry["src"])
    return data


def prepare(data, size):
    """Return base64 JPEG/PNG resized to `size` longest side, or a skip reason string."""
    if len(data) < MIN_BYTES:
        return None, f"too small ({len(data)} B)"
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        return None, f"not an image ({e})"
    if max(img.size) < MIN_SIDE:
        return None, f"icon ({img.size[0]}x{img.size[1]})"
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img.thumbnail((size, size), Image.LANCZOS)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=92)
    return base64.b64encode(out.getvalue()).decode("ascii"), None


def describe(b64, model, extra):
    # num_predict/repeat_penalty: at temperature 0 the 7b model can loop on
    # "[unreadable]" until the context is full and never reach TYPE..LAYOUT.
    options = {"temperature": 0, "num_ctx": 4096, "num_predict": 1200, "repeat_penalty": 1.1}
    options.update(extra)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT, "images": [b64]}],
        "stream": False,
        "options": options,
    }
    r = requests.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


LABELS = ("TEXT:", "TYPE:", "PLACE:", "OBJECTS:", "LAYOUT:")
# At temperature 0 a retry gives the same reply, so retries sample with fixed seeds.
RETRY_OPTS = [{"temperature": 0.3, "seed": 1}, {"temperature": 0.3, "seed": 2}, {"temperature": 0.6, "seed": 3}]


def score(reply):
    """Number of section labels present; 0 if the reply echoes the prompt."""
    if "Copy every piece of visible text" in reply:
        return 0
    return sum(label in reply for label in LABELS)


def redo_bad(args):
    """Retry replies that miss a section. Keeps the best-scoring reply per image."""
    with open(TODO, encoding="utf-8") as f:
        entries = json.load(f)
    done = load(OUT)
    bad = [e for e in entries if e["src"] in done and score(done[e["src"]]) < len(LABELS)]
    if args.limit:
        bad = bad[:args.limit]
    print(f"{len(bad)} malformed replies to retry | dry_run={args.dry_run}")
    fixed = 0
    try:
        for i, e in enumerate(bad, 1):
            t0 = time.time()
            data = fetch(e)
            b64 = prepare(data, args.size)[0] if data else None
            if b64 is None:
                print(f"[{i}/{len(bad)}] SKIP download/prepare failed  {e['src']}")
                continue
            best = done[e["src"]]
            for extra in RETRY_OPTS:
                try:
                    reply = describe(b64, args.model, extra)
                except Exception as ex:
                    print(f"    ERROR ollama: {ex}")
                    continue
                if score(reply) > score(best):
                    best = reply
                if score(best) == len(LABELS):
                    break
            if score(best) > score(done[e["src"]]):
                fixed += score(best) == len(LABELS)
                if args.dry_run:
                    print("-" * 70 + "\n" + best + "\n" + "-" * 70)
                else:
                    done[e["src"]] = best
            print(f"[{i}/{len(bad)}] {time.time() - t0:5.1f}s  score {score(best)}/5  {e['src']}")
            if not args.dry_run and i % SAVE_EVERY == 0:
                save(OUT, done)
    except KeyboardInterrupt:
        print("interrupted, saving...")
    finally:
        if not args.dry_run:
            save(OUT, done)
        left = sum(score(v) < len(LABELS) for v in done.values())
        print(f"fully fixed {fixed}; {left} of {len(done)} replies still miss a section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo-bad", action="store_true", help="retry replies that miss a section")
    ap.add_argument("--dry-run", action="store_true", help="with --redo-bad: print, don't save")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new descriptions")
    ap.add_argument("--test", nargs="+", metavar="URL")
    ap.add_argument("--opts", default="{}", help='extra Ollama options as JSON, e.g. \'{"repeat_penalty": 1.2}\'')
    args = ap.parse_args()

    if args.redo_bad:
        return redo_bad(args)
    if args.test:
        entries = [{"src": u, "original": ""} for u in args.test]
        out_path, skipped = os.path.join(HERE, "descriptions_test.json"), {}
        done = {}
    else:
        with open(TODO, encoding="utf-8") as f:
            entries = json.load(f)
        out_path = OUT
        done, skipped = load(OUT), load(SKIPPED)

    todo = [e for e in entries if e["src"] not in done and e["src"] not in skipped]
    print(f"{len(entries)} entries, {len(done)} done, {len(skipped)} skipped, {len(todo)} to go "
          f"| model={args.model} size={args.size}")

    new, unsaved, t_start = 0, 0, time.time()
    try:
        for i, e in enumerate(todo, 1):
            t0 = time.time()
            data = fetch(e)
            if data is None:
                skipped[e["src"]] = "download failed"
                print(f"[{i}/{len(todo)}] SKIP download failed  {e['src']}")
                continue
            b64, reason = prepare(data, args.size)
            del data
            if b64 is None:
                skipped[e["src"]] = reason
                print(f"[{i}/{len(todo)}] SKIP {reason}  {e['src']}")
                continue
            try:
                reply = describe(b64, args.model, json.loads(args.opts))
            except Exception as ex:
                print(f"[{i}/{len(todo)}] ERROR ollama: {ex}  {e['src']}")
                continue
            done[e["src"]] = reply
            new += 1
            unsaved += 1
            dt = time.time() - t0
            avg = (time.time() - t_start) / i
            eta_h = avg * (len(todo) - i) / 3600
            print(f"[{i}/{len(todo)}] {dt:5.1f}s  tier={e.get('tier', '-')}  eta {eta_h:.1f}h  {e['src']}")
            if args.test:
                print("-" * 70 + "\n" + reply + "\n" + "-" * 70)
            if unsaved >= SAVE_EVERY:
                save(out_path, done)
                if not args.test:
                    save(SKIPPED, skipped)
                unsaved = 0
            if args.limit and new >= args.limit:
                break
    except KeyboardInterrupt:
        print("interrupted, saving...")
    finally:
        save(out_path, done)
        if not args.test:
            save(SKIPPED, skipped)
        print(f"saved {len(done)} descriptions to {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
