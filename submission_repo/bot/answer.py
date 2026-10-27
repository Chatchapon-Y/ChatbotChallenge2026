"""Your chatbot. STUB. This is the file you actually write.

Two functions must exist with these exact names and signatures. Every
other line in this file, and every file under build/, is yours to
rewrite.
"""
import json
import re
from pathlib import Path

from bot.bm25 import get_bm25, tokenize
from bot.llm import VISION_DEPLOYMENT, VISION_URL, _client, chat
from bot.store import get_store, query

# --------------------------------------------------------------------
# The prompt. Workshop 1 block 2 covers what each part is doing.
# --------------------------------------------------------------------
SYSTEM_PROMPT = """You answer trivia questions about the Tam Wing Fan Innovation Wing
at HKU, and general knowledge questions about engineering and making.

You are given some context retrieved from the Innovation Wing websites.
If the context contains the answer, use it; where it disagrees with what
you think you know, the context is correct.
If the context is empty or unrelated to the question, answer from your
own general knowledge. Do not mention the context in your answer.

Reply with the answer only. No explanation, no preamble, no hedging.
Do not end the answer with a full stop.
If the question asks how many, reply with a number.
Give a price with its currency and a size with its unit, as the context
writes them (HK$300, 2,400 square metres).
If the answer is an abbreviation, give the full name too.
If the answer is a web address, give the whole address, not only the site name.
If the question asks for a colour or colour scheme, name the main colours in
plain words, e.g. "Blue and white".
If the question asks for names (teams, people, projects), give every name in
full as written, separated by semicolons. On a poster a team name is often
followed by a line saying what the project is ("AquaSense", then "Smart Water
Bottle for Hikers"): give that line with the name.
If the question names a year, use only context about that year.
A passage marked "Rough description by a small local model" may have wrong
counts: for a number, prefer any other passage that gives one.
A passage marked "Notes written on site by our team, counted by hand" is the
most reliable source about things inside the building: for a count, colour or
position there, prefer it over a photo description. Several photos often show
the same room from different sides: never add their counts together; without
notes, use the largest count seen in one photo.
If the question asks about the first or earliest of a series (first workshop,
first edition), compare the dates in the context and use the earliest one.

Always give a specific answer, even when the context does not contain
it: make your best guess from the context and your own knowledge. A wrong
guess scores the same as no answer, so always guess. Never say that the
information is not provided, not specified, not mentioned or unknown."""

CONFIG = {
    "k": 10,             # 8 missed the floor area, CAD computer and FDM printer pages
    "search": "hybrid",  # "vector", "bm25", or "hybrid" (both, merged)
    "candidates": 40,    # how many each search returns before merging (20: page found 50/55, 40: 54/55)
    "neighbors": 1,      # also read this many following chunks of the same page, 0 = off
    "page_images": 4,    # also read this many image descriptions from the retrieved pages, 0 = off
    "photos": 8,         # in-building photos and on-site notes, on top of the k web chunks
}

# Reciprocal Rank Fusion. A chunk scores 1/(RRF_K + rank) in each list
# it appears in, and the scores add. Only the rank counts, so a vector
# distance and a BM25 score never need to be on the same scale. 60 is
# the usual value from the original paper.
RRF_K = 60

# The model still refuses sometimes despite the prompt ("...is not
# provided in the context"). A refusal always scores 0, so ask again.
REFUSAL = re.compile(
    r"not (provided|specified|mentioned|available|given|stated|included)"
    r"|does not (contain|provide|mention|specify|include)"
    r"|no (information|mention|details)"
    r"|cannot (be determined|determine|find)"
    r"|\bunknown\b|\bunreadable\b|i don.?t know|i do not know|not sure",
    re.I)
# gpt-4o-mini sometimes restates the question despite the prompt: "The
# winning teams in Pitch New Tech Ideas in 2025 were SmartSocks, ...".
# Workshop 0: points are deducted when the output format is wrong. A
# reply that repeats several question words around a verb is a sentence,
# so ask once more for the answer alone.
SENTENCE_VERB = re.compile(r"\b(is|are|was|were|has|have|had|does|did)\b", re.I)
SHORTEN = ("Rewrite your reply as the answer only: just the name, number, date "
           "or short phrase, not a sentence. Keep every item if there are several. "
           "No full stop.")
GUESS_AGAIN = ("Do not say the information is missing. Give your single "
               "best guess. Reply with the answer only.")

# gpt-4o-mini finds the right numbers and then gets the sum wrong: "15
# workbenches with 6 seats each" -> 66, "66 vs 55 square metres, how
# much larger" -> 55. Questions worded like a calculation go to
# gpt-5-mini, which reasons before it answers. Everything else stays on
# gpt-4o-mini, which is faster.
ARITHMETIC = re.compile(
    r"\bhow (many|much) (more|fewer|less|larger|bigger|smaller|longer|shorter|higher|lower)\b"
    r"|\bdifference\b|\bcombined\b|\bin total\b|\baltogether\b|\bsum of\b"
    r"|\btotal (number|amount|cost|area|prize)\b|\bdays (were there )?between\b",
    re.I)
CALCULATE = ("This question needs arithmetic. Find every number it needs in the "
             "context, check that each one belongs to the item the question means, "
             "then calculate. Reply with the final answer only: the number and its "
             "unit, no working.")
# The other trap is a passage that looks like the answer but is about
# something next to it. "Winning teams in Pitch New Tech Ideas 2025" was
# answered from a poster for a sharing session *before* the 2025 final,
# by the 2024 winners, whose text says "Pitch New Tech Ideas 2025". "The
# first Robot Arm workshop" was answered from a poster reading "WORKSHOP
# 1" from a later season. gpt-5-mini checks candidates against the
# question when told to; gpt-4o-mini takes the first good-looking one.
CAREFUL = re.compile(
    r"\b(win|wins|won|winner|winners|winning|champions?|first|earliest|latest"
    r"|most recent|oldest|newest)\b", re.I)
CHECK = ("Several passages may look like the answer. Check each candidate against "
         "every detail of the question: the exact event and edition, the year, and "
         "whether the passage is the result itself or something else about it (a "
         "sharing session by earlier winners, a promotion poster, a later season). "
         "If the question asks for the first, earliest or latest, compare the dates "
         "and choose by them. Reply with the final answer only.")
REASON_TIMEOUT = 15     # seconds; on timeout gpt-4o-mini answers instead (startup ~6 s + search ~3 s + this < 30 s)

# Questions about the building go to gpt-5-mini too. With photos of one
# room from several sides, gpt-4o-mini mixed them up: the colour of the
# letters for the colour of the background behind ENGINEERING, the
# CHECK POINT sign on IW-LG13's door for IW-LG14's.
INSIDE_CHECK = ("Some passages are photos taken inside the building and notes our "
                "team wrote on site. Use only the ones about exactly the place and "
                "object the question names (check the zone and spot each photo was "
                "taken at). For a colour, keep the colour of letters apart from the "
                "colour of the background. Prefer the on-site notes for counts. "
                "Reply with the final answer only.")

# A calculation names two things, and one embedding of the whole question
# lands between them (Workshop 2, block 4): "How much larger is the open
# event area than the Digital Learning Studio?" found the studio's 55 m²
# but not the open event area's 66 m² on the same page. Each part is
# searched on its own as well.
SPLIT = ("List the separate facts needed to answer this question, one short "
         "search query per line, at most 3 lines, nothing else.")
PART_K = 4              # chunks added per part


def parts_of(question: str) -> list[str]:
    try:
        reply = chat([{"role": "system", "content": SPLIT},
                      {"role": "user", "content": question}], max_tokens=80)
    except Exception:
        return []
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in reply.splitlines()]
    return [line for line in lines if len(line) > 3][:3]


def reason(messages: list[dict]) -> str:
    """One gpt-5-mini call. Returns "" on any failure so the caller can fall back."""
    try:
        r = _client(VISION_URL).chat.completions.create(
            model=VISION_DEPLOYMENT, messages=messages, reasoning_effort="low",
            max_completion_tokens=4000, timeout=REASON_TIMEOUT,
        )
        return r.choices[0].message.content or ""
    except Exception:
        return ""


# Workshop posters per year, counted from /workshop/ by build/counts.py.
# An LLM reading chunks cannot count them: the page text does not say
# which poster sits under which "Year 20xx" heading.
POSTER_COUNTS = Path("data/poster_counts.json")
_poster_counts = None


def counted_answer(question: str) -> str | None:
    """'How many workshop posters ... for 2025?' -> "31", else None.

    Only a question naming workshop posters, "how many" and exactly one
    year in the table is answered here; everything else goes to the
    normal search, so a near miss costs nothing.
    """
    global _poster_counts
    q = question.lower()
    if "how many" not in q or "poster" not in q or "workshop" not in q:
        return None
    years = re.findall(r"\b(20\d\d)\b", q)
    if len(years) != 1:
        return None
    if _poster_counts is None:
        try:
            _poster_counts = json.loads(POSTER_COUNTS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _poster_counts = {}
    count = _poster_counts.get(years[0])
    return None if count is None else str(count)


# A question about one year must not be answered from another year's
# page: the Pitch New Tech Ideas 2026 question ranked three 2025 winner
# posters above the 2026 one, and the model named the 2025 winners.
YEAR = re.compile(r"\b(20\d\d)\b")


def other_year(question_years: list[str], chunk: dict) -> bool:
    """True if the chunk's page, title or image file is about a different year.

    Only the address, title and file name count, not the text: the 2026
    page mentions 2025 too. The /uploads/2025/ folder is when a file was
    uploaded, not what it is about, so only the file's own name is read.
    """
    if len(question_years) != 1:
        return False
    m = chunk["metadata"]
    label = " ".join([m.get("url", ""), m.get("title", ""),
                      str(m.get("image", "")).rsplit("/", 1)[-1]])
    years = set(YEAR.findall(label))
    return bool(years) and question_years[0] not in years


# build/images.py marks descriptions from the local model (good text,
# unreliable counts). 3,394 short chunks flooded the results: "What is
# PLA short for" got 9 of them in 12. They now only fill a slot when no
# other candidate is left, and come last among a page's images.
LOCAL = "Rough description by a small local model"


def is_local(chunk: dict) -> bool:
    return LOCAL in chunk["text"]


# A website image description puts its text first and its layout last;
# past this many characters it is mostly layout, and every question pays
# for it in tokens. Building photos are not cut: their OCR comes last.
IMAGE_CHARS = 3000


def clip(chunk: dict) -> str:
    text = chunk["text"]
    return text[:IMAGE_CHARS] if chunk["metadata"].get("kind") == "image" else text


def is_photo(chunk: dict) -> bool:
    """A photo taken inside the building, or the team's on-site notes (build/lv5.py)."""
    return chunk["metadata"].get("kind") == "photo"


def retrieve(question: str, k: int = None, where: dict = None) -> list[dict]:
    """Return the k chunks most relevant to the question.

    Kept separate from rag_answer on purpose: it lets you check whether
    an answer was ever fetched at all, which is the only way to tell a
    retrieval failure from a prompt failure. Do not delete it even if
    you rewrite everything else.
    """
    k = k or CONFIG["k"]
    store = get_store()
    chunks = search(store, question, k, where)
    chunks = add_neighbors(store, chunks, CONFIG["neighbors"])
    return add_page_images(store, question, chunks, CONFIG["page_images"])


def add_page_images(store, question: str, chunks: list[dict], n: int) -> list[dict]:
    """Append up to n image descriptions from the pages already retrieved.

    A question is worded like the page's text ("winning teams in Pitch
    New Tech Ideas 2025"), so the search finds the /pitching_2025/ text.
    The answer is on a poster on that page, whose description says
    "WINNERS ... SMARTSOCKS" and ranks outside the top results on its own.
    """
    if n <= 0:
        return chunks
    taken = {(c["metadata"].get("url"), c["metadata"].get("position")) for c in chunks}
    urls = {c["metadata"].get("url") for c in chunks}
    years = list(set(YEAR.findall(question)))
    images = [im for im in get_bm25(store).images_on(question, urls, n + len(chunks))
              if (im["metadata"]["url"], im["metadata"]["position"]) not in taken
              and not other_year(years, im)]
    quoted = quoted_words(question)
    images.sort(key=lambda im: (not names_quote(im, quoted), is_local(im)))
    return chunks + images[:n]


QUOTED = re.compile(r"[\"“”']([^\"“”']{3,60})[\"“”']")


def quoted_words(question: str) -> set:
    """Words of any text the question quotes: 'the sign that says "Trained Members Only"'."""
    return {w for q in QUOTED.findall(question) for w in tokenize(q) if len(w) >= 4}


def names_quote(chunk: dict, words: set) -> bool:
    """True if the image's file name has a quoted word ("trainedmember.png").

    An icon has no text of its own to match, only its file name, and
    "members" must match "member", so either word may start the other.
    """
    line = next((l for l in chunk["text"].splitlines() if l.startswith("File name:")), "")
    names = [w for w in tokenize(line[10:]) if len(w) >= 4]
    return any(a.startswith(b) or b.startswith(a) for a in words for b in names)


def search(store, question: str, k: int, where: dict = None) -> list[dict]:
    """k chunks from the websites, plus CONFIG["photos"] from the building.

    Photos of the building and the team's on-site notes are searched
    apart: in one shared search they filled the candidate lists for
    "How many computers are in the Digital Learning Studio?" and pushed
    the /venue/ page out, and a Lv5 question still needs them.
    """
    web = ranked_search(store, question, k, CONFIG["candidates"], where, photos=False)
    n = (0 if ONLINE.search(question)
         else CONFIG["photos"] if INSIDE.search(question) else min(CONFIG["photos"], 2))
    if where is not None or n <= 0:
        return web
    return web + ranked_search(store, question, n, PHOTO_CANDIDATES,
                               {"kind": "photo"}, photos=True)


# "...shown as icons on the Wet Lab page" asks about the website: a photo
# of a sign in the building answered it with the wrong icons.
ONLINE = re.compile(r"\b(web ?page|page|website|web ?site|online)\b", re.I)
PHOTO_CANDIDATES = 30   # 15 missed the stair-step notes behind "ENGINEERING"
# A question about the building itself names a place in it or a thing you
# see there. Others get only 2 photo chunks: each is 2-5k characters.
INSIDE = re.compile(
    r"makerspace|brainstorm|open event|digital learning|studio|corridor|stair|lobby"
    r"|entrance|gallery|room|area|wall|floor|door|sign|poster|board|locker|shelf|shelves"
    r"|chair|stool|seat|bench|table|desk|printer|screen|monitor|projector|computer"
    r"|colou?r|inside|located|next to|behind|in front of|LG/?F|G/F|IW-", re.I)


def ranked_search(store, question: str, k: int, n: int, where: dict | None,
                  photos: bool) -> list[dict]:
    years = list(set(YEAR.findall(question)))
    vector_where = where if photos or where else {"kind": {"$ne": "photo"}}
    lists = []
    if CONFIG["search"] in ("vector", "hybrid"):
        lists.append(query(store, question, k=n, where=vector_where))
    if CONFIG["search"] in ("bm25", "hybrid"):
        # BM25 filters on plain equality only, so photos are dropped here.
        hits = get_bm25(store).search(question, k=n + (0 if photos else 300), where=where)
        lists.append([c for c in hits if is_photo(c) == photos][:n])
    scores, chunks = {}, {}
    for ranked in lists:
        for rank, c in enumerate(ranked):
            key = (c["metadata"].get("url"), c["metadata"].get("position"))
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            chunks.setdefault(key, c)
    # Chunks about another year go to the back, behind every other
    # candidate; local-model descriptions just before them.
    best = sorted(scores, key=lambda key: (other_year(years, chunks[key]),
                                           is_local(chunks[key]), -scores[key]))[:k]
    return [{**chunks[key], "rrf": scores[key]} for key in best]


def add_neighbors(store, chunks: list[dict], n: int) -> list[dict]:
    """Append the next n chunks of the same page to each retrieved chunk.

    Fixed-size chunking can cut a heading from the fact under it: on
    /funding-scheme/ "Second Round 2025-26" ends chunk 2 and the deadline
    "February 27 2026" starts chunk 3. The search finds chunk 2, so
    reading one chunk further brings the answer in.

    A neighbour that is itself in the results is skipped, so no text is
    sent twice.
    """
    if n <= 0:
        return chunks
    bm25 = get_bm25(store)
    taken = {(c["metadata"].get("url"), c["metadata"].get("position")) for c in chunks}
    out = []
    for c in chunks:
        url, pos = c["metadata"].get("url"), c["metadata"].get("position")
        text = c["text"]
        # An image or photo description is complete on its own; the "next
        # chunk" would be a different image of the same page or zone.
        steps = 0 if c["metadata"].get("kind") in ("image", "photo") else n
        for step in range(1, steps + 1):
            key = (url, pos + step)
            if key in taken:
                break
            nxt = bm25.text_at(*key)
            if nxt is None:
                break
            taken.add(key)
            # Drop the title line that index.py puts on every chunk.
            text += "\n" + nxt.split("\n", 1)[-1]
        out.append({**c, "text": text})
    return out


def rag_answer(question: str) -> str:
    """One question in, one answer out.

    Runs once per question with a 30 second budget. Anything expensive
    belongs in build/, not here.
    """
    counted = counted_answer(question)
    if counted is not None:
        return counted

    chunks = retrieve(question)
    if ARITHMETIC.search(question):
        store = get_store()
        taken = {(c["metadata"].get("url"), c["metadata"].get("position")) for c in chunks}
        for part in parts_of(question):
            for c in add_neighbors(store, search(store, part, PART_K), CONFIG["neighbors"]):
                key = (c["metadata"].get("url"), c["metadata"].get("position"))
                if key not in taken:
                    taken.add(key)
                    chunks.append(c)

    context = "\n\n".join(
        f"[{c['metadata'].get('url', '?')}]\n{clip(c)}" for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    reply = ""
    note = (CALCULATE if ARITHMETIC.search(question)
            else CHECK if CAREFUL.search(question)
            else INSIDE_CHECK if INSIDE.search(question) else None)
    if note:
        reply = reason([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\n{note}"},
        ])
    if not reply.strip():
        reply = chat(messages)

    if REFUSAL.search(reply):
        reply = chat(messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": GUESS_AGAIN},
        ])

    if looks_like_sentence(question, reply):
        reply = chat(messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": SHORTEN},
        ])
    return reply.strip().rstrip(".")


def looks_like_sentence(question: str, reply: str) -> bool:
    """True for "The X of Y was Z", False for "Z" or a long proper name."""
    if len(reply.split()) < 6 or not SENTENCE_VERB.search(reply):
        return False
    return len(set(tokenize(question)) & set(tokenize(reply))) >= 3


def rag_answer_batch(questions: list[str]) -> list[str]:
    """Many questions in, the same number of answers out, in order.

    Ships as a loop, which is correct and is all most teams will need.
    Replace it if you can share work across questions: one embedding
    call for every query rather than one per query, the store opened
    once, sub-queries running concurrently.

    Whatever you do, answers[i] must be the answer to questions[i].
    """
    return [rag_answer(q) for q in questions]
