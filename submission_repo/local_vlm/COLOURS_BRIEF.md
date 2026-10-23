# Brief 2: colour pass with the local model

Written for: Claude Code on Pong's second computer (the RTX 5060 machine).
This follows on from `README.md` and `REPORT_FOR_MAIN.md`. The setup is the same:
Ollama with `qwen2.5vl:7b`, and `describe_local.py` already works.

## Why
- The final asks "what colour…" questions about posters and signs. Two of
  the three preliminary questions the organisers published were colour questions.
- gpt-5-mini coloured 1,296 images. **3,572 indexed images still have no colour
  information at all.**
- Colour is a whole-picture property, so the 7b model should be reliable here.
  It is not like counting.
- Thank you for run 1. Its descriptions are indexed. Their TEXT turned out
  useful; their counts are labelled as rough.

## Input
- **`colours_todo.json`:** 3,572 entries `{"src", "original", "page"}`, posters and
  icons first. Do them top to bottom.
- **`colours_prompt.txt`:** the exact prompt. It asks for one line in a fixed form:
  `background: …; title/main text: …; other main colours: …; overall: …`

## Output
- **`colours_local.json`:** `{ "<src>": "<the one-line reply>", ... }`
  - The key is the exact `src` string.
  - Write UTF-8 with `ensure_ascii=False`.
  - Save every ~25 images. Make the run resumable, like run 1.
  - Skip downloads that fail, and icons or tiny files, with the same rules as
    run 1. Log them to `colours_skipped.json`.

Bring back `colours_local.json`, even a partial one, by **25 Oct, 18:00**.

## How
Reuse `describe_local.py`. Add a mode, or copy it to `colour_local.py`, with
these changes:
- **Prompt:** `colours_prompt.txt`. Remove the five-sections system message
  added in run 1.
- **Image size:** resize to **512 px** on the longest side. Colour does not
  need more.
- **Options:** `{"temperature": 0, "num_ctx": 2048, "num_predict": 120, "repeat_penalty": 1.1}`.
- **Validation:** keep a reply only if it is a single line containing
  `background:` and `overall:`. Otherwise retry up to 2 times with temperature
  0.3, then skip it.
- **Colour names:** they must be plain English (blue, dark blue, white, …).
  If the model answers in Chinese, add a system message: "Answer in English only."

Check 5 replies against the real images before the long run. Expect about
1–3 s per image, so roughly 2–3 hours in total.

## Do not
- Use a cloud API, or put a key in this folder.
- Change the prompt wording, or the output keys.
