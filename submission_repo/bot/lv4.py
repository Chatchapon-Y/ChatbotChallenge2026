"""Level 4: questions whose answer is not written down anywhere.

W2 slides 18-25. Four kinds, told apart by their wording (no model call,
so Lv1-3 questions pay nothing):

  arithmetic  "How much larger is the open event area than the DLS?"
  multi       "When was the 2026 contest held, and how many were shortlisted?"
  compare     "Which year has the most workshop posters?"
  count       "How many 3D printers does the Centre run in total?"

What each kind gets that a normal question does not:

- Fact sheets (data/facts.json, build/facts.py): counts and totals
  worked out at build time. The 7 printers are on two sites; no chunk
  says 7.
- Decomposition (arithmetic, multi, compare): one gpt-4o-mini call
  splits the question into a search per subject, run at the same time.
  One embedding for "UK tour vs Singapore tour" sits between the two
  pages and finds neither well (slide 19).
- Wide retrieval and the whole top page (count): k sets a ceiling on any
  count (slide 20), so a count reads 20 chunks and every chunk of the
  best page, where lists like "Shortlisted Projects" live.
- Exact-phrase filter (count): every page naming "Terence Leung", not
  the 10 most similar (slide 21, search vs filter).
- List, then answer (slide 25): the model writes the items it found and
  ends with "ANSWER: ...", which final_answer() keeps.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

from bot.bm25 import BM25, get_bm25
from bot.llm import CHAT_DEPLOYMENT, CHAT_URL, _client
from bot.store import get_store

# --------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------
# gpt-4o-mini finds the right numbers and then gets the sum wrong: "15
# workbenches with 6 seats each" -> 66, "66 vs 55 square metres, how
# much larger" -> 55. These go to gpt-5-mini, which reasons first.
ARITHMETIC = re.compile(
    r"\bhow (many|much) (more|fewer|less|larger|bigger|smaller|longer|shorter|higher|lower)\b"
    r"|\bdifference\b|\bcombined\b|\baltogether\b|\bsum of\b|\bdays (were there )?between\b"
    r"|\bhow (many|much)\b.*\b(than|between|per (week|day|month|year))\b",
    re.I)
# "What is the total cash prize ...": a sum, unless it asks how many
# ("How many 3D printers ... in total" is a count).
TOTAL = re.compile(r"\btotal\b", re.I)
# A second question inside the first: ", and how many ...", two question
# marks, or a link between two lists ("winners who later spoke").
MULTI = re.compile(
    r"\?\s*\S.*\?"
    r"|,?\s+\band\s+(what|which|who|whom|when|where|why|how)\b"
    r"|\b(later|both|as well as)\b",
    re.I)
# "at most 10 people" and "the most recent" are not comparisons; "most
# recent", "latest" and "first" stay with answer.CAREFUL.
COMPARE = re.compile(
    r"(?<!at )\b(more|most|fewer|fewest|least|larger|largest|bigger|biggest"
    r"|smaller|smallest|longer|longest|shorter|shortest|higher|highest|lowest"
    r"|cheaper|cheapest|older|newer)\b(?!\s+recent)"
    r"|\bcompared?\b|\bversus\b|\bvs\.?\b",
    re.I)
COUNT = re.compile(r"\bhow many\b|\b(the|total) number of\b|\bcount\b", re.I)
# "According to the Receiptify poster, how many trees ...": the number is
# printed on one poster. Reading it is Lv3, not a count.
LOOKUP = re.compile(r"^\s*according to\b", re.I)
QUOTED = re.compile(r"[\"“][^\"”]*[\"”]")


def route(question: str) -> str | None:
    """The Lv4 kind of a question, or None for an ordinary one."""
    # A quoted title can hold its own "?" ("AI + IoT + Construction = ?").
    text = QUOTED.sub('"title"', question)
    if LOOKUP.search(text):
        return None
    for kind, pattern in (("arithmetic", ARITHMETIC), ("multi", MULTI),
                          ("compare", COMPARE), ("count", COUNT),
                          ("arithmetic", TOTAL)):
        if pattern.search(text):
            return kind
    return None


# Appended to the question for gpt-5-mini. Each ends in an ANSWER line,
# so the list the model writes first never reaches the judges.
FINAL = ("End with a last line 'ANSWER: ' followed by the answer only: no "
         "working, no full stop.")
NOTES = {
    "count": (
        "This question asks how many. If a passage states the number for exactly "
        "this (a Fact sheet, or a line like '6 proposals are shortlisted'), use it. "
        "Otherwise the number is not written anywhere: find every distinct item "
        "in the context that matches every detail of the question (the right "
        "event, year, place and kind of thing) and count them. The same item seen "
        "twice, on both websites or in two passages, counts once. A passage marked "
        "'Rough description by a small local model' may have wrong counts. Write "
        "the items one per line starting with '- ', then the count. " + FINAL),
    "arithmetic": (
        "This question needs arithmetic. Write each number it needs, one per line "
        "starting with '- ', with what it belongs to, and check that each one is "
        "about the item, place, year or edition the question means. Then "
        "calculate. Give the result with its unit or currency (HK$4,500, 11 "
        "square metres, 12 days). " + FINAL),
    "compare": (
        "This question compares things. Write the value for each thing being "
        "compared, one per line starting with '- ', checked against the question "
        "(right event, year, place, edition). Then compare them and answer what "
        "the question asks: the name or year that wins, or the difference with "
        "its unit. " + FINAL),
    "multi": (
        "This question asks more than one thing. The passages are grouped by the "
        "search they came from. Answer each part from the passages; if a part "
        "links two lists (who appears in both), compare the lists name by name. "
        "One team or project is often written differently in two places: a short "
        "form ('Project X' or 'Team X' for 'X: a long title'), another spelling or "
        "capitalisation. Count those as the same name. "
        "Write each part's answer on its own line starting with '- '. " + FINAL
        + " For several parts, give every part in the order asked, separated by "
        "'; '."),
}

ANSWER_LINE = re.compile(r"^\W*answer\W*:\s*(.+?)\s*$", re.I | re.M)


def final_answer(reply: str) -> str:
    """The text after the last 'ANSWER:', or the reply if there is none."""
    found = ANSWER_LINE.findall(reply or "")
    return found[-1].strip().strip("*").strip() if found else (reply or "")


# --------------------------------------------------------------------
# Fact sheets
# --------------------------------------------------------------------
FACT_FILES = [Path("data/facts.json"), Path("data/facts_lv5.json")]
# BM25 score below which a fact is not about the question. Low, because
# "Makerspace A" scores on "makerspace" alone ("a" is a stopword): its
# chairs note scored 3.8. The half-the-best rule in top_facts does most
# of the filtering.
FACT_SCORE = 3.0
_facts = None


def facts_index() -> BM25 | None:
    global _facts
    if _facts is None:
        docs, metas = [], []
        for path in FACT_FILES:
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for i, f in enumerate(rows):
                docs.append(f"Fact sheet: {f['title']}\n{f['text']}")
                metas.append({"url": (f["sources"] or [path.name])[0],
                              "position": -1 - len(metas), "kind": "fact",
                              "title": f["title"]})
        _facts = BM25(docs, metas) if docs else False
    return _facts or None


def top_facts(question: str, n: int = 4) -> list[dict]:
    """Up to n fact sheets about the question; at least half the best score."""
    index = facts_index()
    if index is None:
        return []
    found = index.search(question, k=n)
    floor = max(FACT_SCORE, found[0]["score"] / 2) if found else 0
    return [f for f in found if f["score"] >= floor]


# --------------------------------------------------------------------
# Exact-phrase filter and whole pages
# --------------------------------------------------------------------
PHRASE_PAGES = 25   # a phrase on more pages than this is too common to filter on
QUESTION_WORDS = {"How", "What", "Which", "Who", "Whom", "When", "Where", "Why",
                  "If", "During", "On", "In", "According", "Besides", "The", "A"}
TITLE_RUN = re.compile(r"(?:[A-Z][\w&'.-]*\s+){1,5}[A-Z][\w&'-]*")


def phrases(question: str) -> list[str]:
    """Quoted text and runs of capitalised words: 'Terence Leung', 'Event Hall A'."""
    found = re.findall(r"['\"“‘]([^'\"”’]{3,60})['\"”’]", question)
    for run in TITLE_RUN.findall(question):
        words = run.split()
        while words and words[0] in QUESTION_WORDS:
            words = words[1:]
        while words and words[0].rstrip(".") in ("Prof", "Dr", "Mr", "Ms", "Mrs"):
            words = words[1:]
        if len(words) >= 2:
            found.append(" ".join(words))
    return found


def phrase_chunks(question: str) -> list[dict]:
    """One chunk from every page that contains a rare phrase of the question.

    Ranked search returns the k most similar chunks and never promises it
    found them all; a count needs all of them. The rarest phrase that
    appears on 2 to PHRASE_PAGES pages wins.
    """
    bm25 = get_bm25(get_store())
    best = None
    for phrase in phrases(question):
        needle = phrase.lower()
        hits = {}
        for i, doc in enumerate(bm25.docs):
            if needle in doc.lower():
                hits.setdefault(bm25.metas[i].get("url"), i)
        if 2 <= len(hits) <= PHRASE_PAGES and (best is None or len(hits) < len(best)):
            best = hits
    if not best:
        return []
    return [{"text": bm25.docs[i], "metadata": bm25.metas[i]} for i in best.values()]


def whole_page(url: str, limit: int = 12) -> list[dict]:
    """The page's text chunks in order, up to limit."""
    bm25 = get_bm25(get_store())
    out = []
    for pos in range(limit):
        text = bm25.text_at(url, pos)
        if text is None:
            break
        out.append({"text": text, "metadata": {**bm25.metas[bm25.by_position[(url, pos)]]}})
    return out


# --------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------
SPLIT_PROMPT = """Break the question into the separate searches needed to answer it.
Each search is a short question that names its subject in full: repeat the
event names, years, editions and places from the original question.
If the question compares or combines two or more things, write one search
per thing. If it asks two things, write one search per thing.
If one search is enough, write the question unchanged.
One search per line, at most 3 lines, nothing else."""
SPLIT_TIMEOUT = 8       # seconds; on timeout the question is searched whole


def split(question: str) -> list[str]:
    try:
        r = _client(CHAT_URL).chat.completions.create(
            model=CHAT_DEPLOYMENT, temperature=0, max_tokens=200, timeout=SPLIT_TIMEOUT,
            messages=[{"role": "system", "content": SPLIT_PROMPT},
                      {"role": "user", "content": question}])
        lines = (r.choices[0].message.content or "").splitlines()
    except Exception:
        return [question]
    subs = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in lines]
    subs = [s for s in subs if len(s) > 8][:3]
    return subs or [question]


# --------------------------------------------------------------------
# Gathering the context
# --------------------------------------------------------------------
WIDE_K = 20             # count: chunks from the search (normal questions: 10)
SUB_K = 8               # each sub-question of a split one
MAX_CHARS = 32_000      # ~8k tokens: keeps gpt-5-mini well inside 30 seconds


def labelled(chunks: list[dict], label: str) -> list[dict]:
    return [{**c, "text": f"(found for: {label})\n{c['text']}"} for c in chunks]


def gather(question: str, kind: str, retrieve) -> list[dict]:
    """The context for a Lv4 question. retrieve is answer.retrieve."""
    facts = top_facts(question)

    if kind == "count":
        found = retrieve(question, k=WIDE_K)
        text_urls = [c["metadata"].get("url") for c in found
                     if c["metadata"].get("kind") == "text"]
        page = whole_page(text_urls[0]) if text_urls else []
        groups = [found, phrase_chunks(question), page]
    else:
        subs = split(question)
        if len(subs) > 1:
            # Open Chroma and build BM25 before the threads: two threads
            # opening the store at once fail with "Could not connect to
            # tenant default_tenant" when this is the first question.
            get_bm25(get_store())
            with ThreadPoolExecutor(len(subs) + 1) as pool:
                results = list(pool.map(lambda q: retrieve(q, k=SUB_K), subs + [question]))
            groups = [labelled(r, s) for r, s in zip(results, subs + ["the whole question"])]
            # "chairs in the open event area and Makerspace A": each place
            # has its own notes, and one search ranks only one of them high.
            facts += [f for s in subs for f in top_facts(s, 3)]
        else:
            groups = [retrieve(question, k=WIDE_K // 2 + 5)]

    # Take the groups in turn, one chunk each, so the budget is shared:
    # a building photo is 2-5k characters, and the first part's eight
    # photos alone would fill it.
    turns = [c for row in zip_longest(*groups) for c in row if c is not None]
    out, seen, size = [], set(), 0
    for c in facts + turns:
        key = (c["metadata"].get("url"), c["metadata"].get("position"))
        if key in seen:
            continue
        if size + len(c["text"]) > MAX_CHARS:
            continue
        seen.add(key)
        size += len(c["text"])
        out.append(c)
    return out
