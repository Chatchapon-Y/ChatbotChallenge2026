# Brief 3: extra passes over the Lv5 photos (in-building)

Written for: Claude Code on Pong's second computer (RTX 5060, Ollama with
`qwen2.5vl:7b`, the setup from briefs 1 and 2).

## Why
Lv5 questions are about things inside the Innovation Wing. They are about 40% of
the final on 28 Oct. The team took 181 photos.

gpt-5-mini has described 165 of them on the main computer. Close-up signs come out
word-perfect: names and phone numbers on safety placards are read exactly. Three
gaps remain, and two of them can be filled here without the cloud quota:

1. **Walls with many posters.** One photo of a corridor wall holds 10–20 posters.
   In the cloud description, most of their text is "[unreadable]", and their
   colours are lumped together ("posters: white, green, blue").
   - Questions like "what colour is the HKU Racing poster" or "what date is on the
     Inno Show poster in the corridor" fail.
   - Cutting each wall photo into tiles fixes this: describe it poster by poster,
     and run OCR at full resolution.
2. **Small print in wide room photos:** labels on lockers, machine brands, notices
   on far walls. OCR on full-resolution tiles reads these exactly.
3. **133 small icons and signs** were skipped by the colour pass (brief 2) under the
   size rule. `STAFF-ONLY-150x150-1.png` is one of them. Their colour is still
   unknown.

## Input
- **Photos:** `InnoWing Pics.zip` (1.4 GB). Pong copies it here; on the main
  computer it is `D:\HKU\compete\ChatbotChallenge2026\data_innowing_pics_lv5\InnoWing Pics.zip`.
  - Extract it to `local_vlm/lv5_raw/`, so you get
    `local_vlm/lv5_raw/makerspace_a/01_inside_wide/IMG_1416.JPG` and so on.
  - Skip `.DS_Store` and `__MACOSX`.
  - Never commit the photos.
- **Keys:** every output is keyed by the photo's path relative to `lv5_raw/`, with
  forward slashes and the original case, for example
  `lg_corridor_posters/IMG_1311.JPG`. The main computer matches on that exact
  string.
- **Skip these four.** They are misfiled copies of open event area photos:
  `makerspace_a/01_inside_wide/IMG_1389.JPG`, `…IMG_1390.JPG`, `…IMG_1391.JPG`,
  `…IMG_1392.JPG`.
- **Notes:** each folder has a `notes.txt`, the team's description of the spot.
  Read them for context, but do not copy them into the outputs.

## Task A: OCR at full resolution (highest value; no model needed)
1. Install the OCR engine: `python -m pip install rapidocr-onnxruntime`. It runs
   on the CPU and handles English well.
2. For each photo:
   - Apply the EXIF rotation (`ImageOps.exif_transpose`).
   - Run OCR on a 3×3 grid of tiles, each overlapping its neighbours by about 15%,
     at full resolution. The photos are 8064×6048, so each tile is about 3000 px.
   - Also run OCR on the whole photo, scaled to 3000 px on its longest side.
3. Keep lines with confidence ≥ 0.6. Drop exact duplicate lines, which come from
   the tile overlaps. Order lines top to bottom, then left to right.
4. **Output** `lv5_ocr_local.json`: `{ "<key>": "line\nline\n..." }`.
   - Save every 10 photos, and make the run resumable.
   - Photos with no text get `""`.

## Task B: poster by poster (qwen2.5vl:7b)
For every photo in `lg_corridor_posters/` and `staircase_lg_to_gf_posters/`, and in
every folder whose name contains `sign`:
1. Cut a 2×2 grid of overlapping tiles (about 15% overlap), resized to 1280 px. For
   a wide wall photo (aspect ratio > 1.2), use a 3×2 grid instead.
2. Ask the model about each tile with this prompt, unchanged:
   ```
   List every poster, sign or notice you can see in this part of a photo of a wall inside a university building, one per line, in this form:
   TITLE (exact words as printed) | colours: background ...; title text ... | key text: names, dates, times, places, prices, numbers exactly as printed
   Skip anything cut off so badly that you cannot read its title. Answer in English. If there is none, write NONE.
   ```
   Use options `{"temperature": 0, "num_ctx": 4096, "num_predict": 600, "repeat_penalty": 1.1}`.
3. Merge a photo's tile replies:
   - Drop NONE lines.
   - When two lines share the same TITLE, which happens through the overlaps,
     keep the longer one.
4. **Output** `lv5_posters_local.json`: `{ "<key>": ["line", "line", ...] }`.
   - Make it resumable.
   - Expect about 4 calls per photo; about 100 photos in total.

## Task C: colours of the skipped icons
Re-run brief 2's colour pass on the entries in `colours_skipped.json`, without the
size rule. Upscale anything smaller than 256 px to 256 px first.

Add the results to `colours_local.json`, using the same format and the same keys.

## Check before the long runs
Show Pong the outputs for these photos:
- Task A: `lg_corridor_posters/IMG_1311.JPG` (a wall of about 15 posters) and
  `makerspace_a/04_closeup_signs_equipment/IMG_1436.JPG` (a fire escape plan).
  The OCR should contain "Fire Assembly Point".
- Task B: `lg_corridor_posters/IMG_1311.JPG`. It should list separate lines for
  HKU DBF TEAM, LEAPS, HKU RACING, the purple "Digital Learning Studio" sign, and so
  on, each with its own colours.

## Monday (26 Oct)
The team takes more photos on Monday. They arrive in the same folder layout.

Extract them over `lv5_raw/` and run A and B again. Both resume, so only the new
photos are done.

## Bring back
Bring back `lv5_ocr_local.json`, `lv5_posters_local.json` and the updated
`colours_local.json`, plus a short `LV5_REPORT.md` covering:
- counts
- 3 sample outputs
- anything that looked wrong

Deadline: **26 Oct, 21:00**. A partial output is fine; bring what is done.

## Do not
- Use a cloud API, or put a key in this folder.
- Rename keys, or change the prompt wording.
- Commit the photos.
