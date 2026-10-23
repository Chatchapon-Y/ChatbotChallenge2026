# Report: local VLM descriptions (from Pong's second computer)

Written for: Claude Code on the main computer. Reply to the brief in `README.md`.
Date written: 23 Oct 2026.

## Status: complete (finished 23 Oct 2026, 23:55)
The whole of `todo.json` has been processed. Every entry is either described or skipped.

| | tier 1–2 | tier 3 | total |
|---|---|---|---|
| Described (in `descriptions_local.json`) | 1,027 | 2,367 | **3,394** |
| Skipped (no entry) | 67 | 8 | 75 |
| **All five sections present** | | | **3,359 (99.0%)** |
| One or more sections missing | | | 35 |

- Every key in the output is an exact `src` from `todo.json`.
- No reply echoes the prompt back.
- The 35 incomplete replies: 23 have 4 of the 5 labels (usually TYPE: is the
  one missing), 10 have 3, and 2 have only 1. The indexer should parse each
  section separately and treat a missing one as "none".
- The screen turned off (Modern Standby, 16:54–19:41) during the run. Speed
  stayed the same, so the run did not stop and nothing was lost.

## Setup used
- Hardware: RTX 5060 Laptop GPU 8 GB, 16 GB RAM, Windows 11.
- Model: `qwen2.5vl:7b` via Ollama, 100% on GPU, 1024 px longest side.
- Speed: about 4–7 s per image. That is about 4–5 h for the whole list, or about
  1.5 h for tier 1–2.

## Changes from the brief (please read)
`prompt.txt` and the output keys are unchanged. The request differs in three ways:

1. **Options are `{"temperature": 0, "num_ctx": 4096, "num_predict": 1200, "repeat_penalty": 1.1}`.**
   With only `temperature` and `num_ctx`, the model printed `[unreadable]` on
   hundreds of lines until the context was full. That took 98 s, and the reply
   never got to TYPE through LAYOUT.
2. **A system message is added before the user prompt:**
   `Reply with exactly five labelled sections, each starting on a new line with its label: TEXT:, TYPE:, PLACE:, OBJECTS:, LAYOUT:.`
   Without it the 7b model often left out the section labels. Prefilling the
   assistant reply with `TEXT:` was tried and failed: the model answered in Chinese.

3. **Second pass for malformed replies (`describe_local.py --redo-bad`).**
   After the main run, 245 replies (7%) were missing a label, and a few were
   unlabelled or echoed the prompt. At temperature 0 a retry returns the same
   reply, so each of these was retried up to 3 times with
   `temperature 0.3/0.3/0.6` and `seed 1/2/3`. The reply with the most sections
   was kept. That fixed 230; 35 are still incomplete.
   Side effect: with sampling, spelling can drift. In one test a name came back
   as "Chueyng" when the image says "Chueying". So for about 230 entries, the
   TEXT section is slightly less trustworthy than in the temperature-0 replies.

## Quality: what to trust
I compared the replies with the real images:

| Section | Reliability |
|---|---|
| TEXT | Good. The three Pitching-Winner team names match the cloud version exactly ("SMARTSOCKS: AI-BASED WEARABLES FOR FEET PRONATION SENSING AND CUSTOMISED CALF MASSAGE", "DIGITOE", "GINGTROLLEY - REVOLUTIONIZING IN-FLIGHT WASTE MANAGEMENT WITH SMART SUSTAINABILITY"). Some small text such as "March 26" and the WINNER signs was missed. |
| TYPE / PLACE | Good in the tests. |
| OBJECTS | **Not reliable.** The Pitching-Winner poster shows 11 people and 3 WINNER signs; the model said "4 people holding certificates" and invented "1 device (SMARTSOCKS)". The Lumino event photo has about 8 people; the model said 5. |
| LAYOUT | Rough, sometimes invented (the "devices held by individuals" above). |

Suggestion: rank local descriptions below the cloud (gpt-5-mini) ones. Don't let
the bot quote counts from local descriptions as exact numbers.

1280 px gave the same result as 1024 px, because the source images are already
that size or smaller, so they are never upscaled.

## Skipped images
- 75 skipped in total: 35 download failed (mostly 404), 20 icons (< 250 px),
  16 too small (< 5,000 bytes), 4 not an image. 67 of them are tier 1–2, almost
  all at the top of the list, which seems to hold the old images the site has
  deleted.
- The first 2 entries the brief named for the quality check both return 404.
  The next 2 that still download were used instead (`kenneth-wong.jpg`,
  `Lumino_HKUEAA-b-768x345.jpg`).
- Skipped images have no entry in `descriptions_local.json`, as the brief asks.
  The reason for each is logged in `skipped_local.json` (`{src: reason}`):
  "download failed", "too small (N B)", "icon (WxH)" or "not an image". This file
  is for information only. Don't merge it. It may help to drop dead images from
  the index.

## Files in this folder
- `descriptions_local.json`: **the output to merge.**
  `python -m build.images merge local_vlm/descriptions_local.json`, then
  `python -m build.images index`.
- `skipped_local.json`: skip log (see above).
- `descriptions_test.json`: quality-check replies only. Don't merge.
- `describe_local.py`: the script. Run it with no arguments to resume, or with
  `--redo-bad` to retry malformed replies.
- `redo_log.txt`: log of the final resume and the `--redo-bad` pass. Don't merge.

## Checking the file
```python
import json
d = json.load(open("local_vlm/descriptions_local.json", encoding="utf-8"))
labels = ("TEXT:", "TYPE:", "PLACE:", "OBJECTS:", "LAYOUT:")
bad = [k for k, v in d.items() if not all(l in v for l in labels)]
print(len(d), "descriptions,", len(bad), "missing a section")
```
