# Report: Lv5 photo passes (brief 3)

Written for Claude Code on the main computer, replying to `LV5_BRIEF.md`.
All three tasks finished on 25 Oct 2026 (about 13:00). The script is `lv5_local.py`.

## Counts
| Task | Output | Result |
|---|---|---|
| A: OCR | `lv5_ocr_local.json` | **261 / 261 photos** after Monday (see the end). First batch: 181 / 181, 7,782 lines in total (median 35 per photo). 3 photos have `""`: `brainstorming/left_iw-lg14_dream_chamber/IMG_1276.JPG`, `digital_learning_studio/01_inside_wide/IMG_1306.JPG`, `makerspace_a/04_closeup_signs_equipment/IMG_1430.JPG`. |
| B: posters | `lv5_posters_local.json` | **115 / 115 photos**, 662 poster/sign lines. 3 photos have `[]` (`makerspace_a/04_closeup_signs_equipment/IMG_1428–1430`, tool-box shelves with no posters). |
| C: icon colours | `colours_local.json` | **133 / 133** skipped icons coloured, 0 failed. The file now covers all 3,572 entries of `colours_todo.json`. |

- Keys are paths relative to `lv5_raw/`, with forward slashes and the original case.
- The four misfiled `makerspace_a/01_inside_wide/IMG_1389–1392.JPG` are not in the outputs.
- **Photos in task B:** the 79 in `lg_corridor_posters/` and `staircase_lg_to_gf_posters/`, plus 36 from folders whose name contains `sign`.

## Changes from the brief (please read)
1. **Task A cleanup is stricter than "drop exact duplicates".** On the test wall
   photo the tile overlaps produced many fragments ("ent Hall A/B", "LOCA",
   "OJECTSERIES") and copies that differed only in spacing ("Electronicworkbenches"
   vs "Electronic workbenches"). After the exact-duplicate pass I also drop:
   - any line contained in a longer line, ignoring case and spaces;
   - single-character lines.

   Where two copies differ only in spacing, the one with spaces is kept. No text
   is lost, because the longer line stays. On the fire-plan photo this cut 59
   lines to 39.
2. **Task B merge strips " (exact words as printed)" from lines.** The model copied
   the prompt's phrase after the title in 125 of 668 lines. That stopped the merge
   from seeing "X" and "X (exact words as printed)" as the same TITLE. Stripping it
   merged 6 more duplicates.
3. **Settings:** RapidOCR runs with `max_side_len=3200`. Its default of 2000 would
   have shrunk the ~3000 px tiles, which defeats "full resolution".

Everything else follows the brief: the 3×3 / 15%-overlap tiles plus the whole
photo at 3,000 px, confidence ≥ 0.6, the poster prompt and options unchanged,
2×2 or 3×2 tiles at 1,280 px, the colour prompt with 256 px upscaling for icons.

## 3 sample outputs
**A: OCR, `makerspace_a/04_closeup_signs_equipment/IMG_1436.JPG` (fire escape plan), excerpt:**
```
Fire Escape RoutePlan
Makerspace A
Machine shop
Electronic workbenches
Lasercuttingroom
3D Printing Area
Makerspace B
Fire Assembly Point: Top of Suen Yat Sun Steps
```
It contains "Fire Assembly Point", as the brief expects. It also keeps some OCR
garbage ("DKINISRATIO", "COERENG", "TR4T").

**B: posters, `lg_corridor_posters/IMG_1311.JPG` (the wall of ~15 posters), 19 lines, excerpt:**
```
LEAPS "LOCAL ENVIRONMENT ALLIANCE" PROJECT SERIES | colours: background green; title text white | key text: LEAPS, LOCAL ENVIRONMENT ALLIANCE, PROJECT SERIES
HKU RACING | colours: background white; title text black | key text: HKUR, Design, Build and Drive, Formula Student team from The University of Hong Kong, #FSUK, MADE IN HONG KONG
Digital Learning Studio | colours: background purple; title text white | key text: NONE
Event Hall A/B | colours: background white; title text blue | key text: Event Hall A/B
AI Chatbot Challenge Inno Wing Trivia | colours: black; title text: AI Chatbot Challenge Inno Wing Trivia |
HKU Robomaster | colours: white; title text: HKU Robomaster | key text: Competition & Goal, Technical Management, Algorithm development, Mechanical & Hardware designs
```
It also lists HKU DBF TEAM, HAYKOZI, Nstar, LG/F of Innovation Wing,
Brainstorming Area Open Event Area, PITCH NEW TECH IDEAS WINNERS, Funding Scheme,
Self-Balancing Bicycle, Social Innovation Challenge Series Bank U, HKUR, LFP POWER
and Project Progress.

**C: colour, `STAFF-ONLY-150x150-1.png`:**
`background: red; title/main text: white; other main colours: none; overall: red, white`

## What looked wrong (checked by eye on `IMG_1311`)
- **Task B misses some posters.** On `IMG_1311` it missed **INNO SHOW**, the
  example in the brief, and the "NOW READY TO DOMINATE THE SEA" poster. The Inno
  Show poster is in the B output for 8 other corridor photos. Task A's OCR does
  read "InnO SHOW" on `IMG_1311`, so combine A and B when answering.
- **Task B attaches text to the wrong poster.** On `IMG_1311`,
  "October 28-31 & November 13-14" is printed on the Bank on U poster, but the
  model put it under HKU UNMANNED AERIAL SYSTEMS TEAM. Treat B's `key text` as
  hints and prefer task A's OCR for exact dates.
- **Task B colours are sometimes wrong or malformed.**
  - The HKU RACING title is white on black, but B says black on white.
  - In 80 lines the "title text" field holds the title words instead of a colour.
  - 8 lines have no colours field.
- **Task B on close-up signs** sometimes splits one sign across tiles into
  fragments. `open_event_area/02_closeup_signs/IMG_1386.JPG` gives "Open",
  "Open Event", "ent", "Area". 9 lines echo the prompt template as their key text
  ("names, dates, times, places, …").
- **Some B lines contain Chinese copied from the sign itself.** 6 lines, for
  example `ENTRY RESTRICTION 進入須特別指示`. This is real printed text, not a
  language error.
- **Task A** still has misread words on angled or small text, for example
  "Toal Box mastee serles" next to the correct "Tool Box master series".

## Files
- **To merge:** `lv5_ocr_local.json`, `lv5_posters_local.json`, `colours_local.json`.
- **Not to merge:**
  - `lv5_posters_tiles.json`: raw tile replies, kept so the merge can be redone.
  - `lv5_*_log.txt`: run logs.
- **Not committed:** the photos in `lv5_raw/`.

## Monday's photos (added 27 Oct)
- **New photos:** 80, all in a new folder, `photo_gallery/`. It is the faculty
  history gallery: framed historic photos with printed captions, plaques, the
  Tam Wing Fan Innovation Wing sign, the purple "Turn Right… Approx. 40.00m" sign,
  and the "How to Become an InnoWing Member?" boards. This folder has no
  `notes.txt`. All the earlier folders are unchanged.
- **Task A:** done for all 80. `lv5_ocr_local.json` now has **261 / 261 photos**.
  - The gallery adds 1,286 lines.
  - Only `photo_gallery/IMG_1501.JPG` has `""`.
  - Captions are read exactly, apart from some missing spaces. Examples:
    "Thirty-tonTesting Machine,circa 1918", "Cement Testing Machine, circa 1932",
    "Site Visit to Tam Wing Fan Innovation Wing,2020",
    "Opening Ceremony for Tam Wing Fan Innovation Wing Two,2021",
    "Access to InnoWing One will be provided within two business days."
  - `IMG_1458` is noisy. A plaque is reflected in the sign's glass, so it gives
    many word fragments.
- **Task B:** nothing new. `photo_gallery` is neither a poster folder nor has
  "sign" in its name, so the brief's rule leaves it out. `lv5_posters_local.json`
  is still 115 photos.
- **Near-duplicates:** `IMG_E1533.JPG` and `IMG_E1535.JPG` are iPhone edited
  copies of `IMG_1533`/`IMG_1535`. Both have their own key. Expect duplicate hits.
