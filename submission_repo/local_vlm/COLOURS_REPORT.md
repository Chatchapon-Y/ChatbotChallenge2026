# Report: colour pass (brief 2)

Written for Claude Code on the main computer, replying to `COLOURS_BRIEF.md`.
The run finished early on 24 Oct 2026 (01:37 to about 06:00).

## Status: complete
| | count |
|---|---|
| Entries in `colours_todo.json` | 3,572 |
| Coloured (in `colours_local.json`) | **3,439** |
| Skipped (in `colours_skipped.json`) | 133: 128 too small (< 5,000 bytes), 5 icons (< 250 px) |
| Not processed | 0 |

- Every key is an exact `src` from `colours_todo.json`.
- Every reply is one line and contains all four fields: `background:`,
  `title/main text:`, `other main colours:`, `overall:`.
- No reply is in Chinese. The "Answer in English only." system message was never
  needed, and no image needed a temperature-0.3 retry.

## Done as the brief says
The script is `colour_local.py`. It uses `colours_prompt.txt` unchanged, with no
system message. Images are resized to 512 px. Options are
`{"temperature": 0, "num_ctx": 2048, "num_predict": 120, "repeat_penalty": 1.1}`.
Each reply is validated as the brief asks: one line containing `background:` and
`overall:`. Speed was about 3.5–4.5 s per image, slower than the brief's estimate
of 1–3 s, mostly because of download time.

## Quality (checked by eye on the first 6 posters)
- `background` and `overall` were right on 6 of 6.
- `title/main text` was right on 4 of 6. It said blue or light blue for two
  white titles (Smart Soccer Robot, HKU Racing). Treat this field as less
  reliable than background and overall.

## Notes for the indexer
- `overall` mostly uses the prompt's colour list. The most common names are
  white (2,252), blue, black, green, dark blue, red and grey. 386 values in
  `overall` are outside the list. The main ones are "black and white" (131, one
  item covering two colours), "silver" (100), "gold" (62), "dark gray" or
  "dark grey" (16; the US spelling "gray" also appears) and "maroon"/"cream".
  To match a question, normalise gray to grey and split "X and Y".
- **Skipped icons:** the brief says to use run 1's skip rules, so 128 small
  files (< 5,000 bytes) have no colour line. Many are icons, QR codes or small
  signs, for example `STAFF-ONLY-150x150-1.png`, which the prompt does cover.
  If colours for icons matter, those 133 entries can be re-run without the
  size rule; that takes about 10 minutes.

## Files
- `colours_local.json`: **the output to merge.**
- `colours_skipped.json`: skip log (`{src: reason}`). Don't merge.
- `colours_log.txt`: run log. `colour_local.py`: the script (resumable).
