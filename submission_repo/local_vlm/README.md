# Brief: describe website images with a local vision model

Written for: Claude Code on Pong's second computer, plus Pong. You do not
need the rest of the project. This folder is everything required.

## Why
Team 1 is in the final of the HKU Chatbot Challenge 2026 (28 Oct 2026). The bot
answers questions about the HKU Innovation Wing websites. Many of those answers
live only inside images: posters, room photos, equipment photos.

- Images are turned into text descriptions that get indexed.
- The organiser's API key (gpt-5-mini) has a quota of about 1.2M tokens a day.
  That quota is reserved for the team's own Lv5 photos.
- Website images that still have no description go to a **local** vision model.
  The rules allow this (W0 slide 5: "Models you deploy locally are fine").
- Never use any other cloud API.

## Input (this folder)
- **`todo.json`:** 3,469 images, already in priority order. Do them top to bottom.
  - Each entry: `{"src", "original", "page", "tier"}`.
  - `tier 1–2`: 1,094 images with no text in them, mostly photos. Most valuable,
    because a description is the only source for counts and objects.
  - `tier 3`: 2,375 images whose text is already captured by OCR. A description
    adds objects and layout. Lower value; do them only if time allows.
- **`prompt.txt`:** the exact prompt to send with each image. Its output format
  (TEXT / TYPE / PLACE / OBJECTS / LAYOUT) must stay the same, because the
  indexer reads it.

## Output (what to bring back)
- **`descriptions_local.json`:** one JSON object, `{ "<src>": "<model reply text>", ... }`
  - The key is the exact `src` string from `todo.json`.
  - Write UTF-8 with `ensure_ascii=False`.
  - Save every ~20 images, so a crash loses little and an interrupted run resumes.
  - Skip an image (no entry) when its download fails, or when it is smaller than
    5,000 bytes or under 250 px on its longest side. Those are icons.

Back on the main computer it is merged with:
`python -m build.images merge local_vlm/descriptions_local.json`
then `python -m build.images index`.

**Deadline:** bring the file back by the evening of 25 Oct, even if only part of
the list is done. A partial file is fine.

## Steps
1. **Check the hardware and pick the model:**

   | Hardware | Model | Expected speed |
   |---|---|---|
   | NVIDIA GPU ≥ 8 GB VRAM, or Apple M-series with ≥ 16 GB | `qwen2.5vl:7b` | ~5–20 s per image |
   | Weaker machines | `qwen2.5vl:3b` | slower |
   | Only ~8 GB RAM and no GPU | none | do not bother; tell Pong |

2. **Install [Ollama](https://ollama.com):** run `ollama pull qwen2.5vl:7b` (or `:3b`), then check that `ollama list` shows the model.
3. **Write `describe_local.py` in this folder:**
   - For each todo entry not already in the output file: download `original`
     (when non-empty and ≤ 4 MB), otherwise `src`. Use a normal `User-Agent` header
     and a 30 s timeout.
   - Resize to 1024 px on the longest side with Pillow.
   - POST to `http://localhost:11434/api/chat` with
     `{"model": ..., "messages": [{"role": "user", "content": PROMPT, "images": [<base64>]}], "stream": false, "options": {"temperature": 0, "num_ctx": 4096}}`.
   - Store `response["message"]["content"]`.
   - Print progress with seconds per image.
   - Run one image at a time. Ollama gains little from parallel requests on one GPU.
   - Delete each downloaded file after use.
4. **Quality check before the long run:** describe these 3 images, show Pong the output,
   and compare it with the cloud version quoted below.
   - `https://innoacademy.engg.hku.hk/wp-content/uploads/2025/03/Pitching-Winner-1024x724.jpg`
     The cloud model read the 3 winners: "SMARTSOCKS: AI-BASED WEARABLES FOR FEET PRONATION SENSING...", "DIGITOE", "GINGTROLLEY - REVOLUTIONIZING IN-FLIGHT WASTE MANAGEMENT...".
   - The first 2 entries of `todo.json`.
   - Things to check:
     - People and objects counted with numbers, not "some".
     - No invented text.
     - All five sections present.

   If the names come out wrong or garbled, try the 7b model, or a size of 1280 px.
5. **Run the whole list** (overnight is fine). If a run is cut short, start it again; it resumes from the output file.

## Do not
- Use any cloud API key, or put a key in this folder. None is needed here.
- Change `prompt.txt`, or the keys of the output file.
- Commit this folder's images or output to git.
