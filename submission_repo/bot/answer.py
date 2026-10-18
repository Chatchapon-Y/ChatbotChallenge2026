"""Your chatbot. STUB. This is the file you actually write.

Two functions must exist with these exact names and signatures. Every
other line in this file, and every file under build/, is yours to
rewrite.
"""
import re

from bot.bm25 import get_bm25, tokenize
from bot.llm import chat
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
If the answer is an abbreviation, give the full name too.

Always give a specific answer, even when the context does not contain
it: make your best guess from the context and your own knowledge. A wrong
guess scores the same as no answer, so always guess. Never say that the
information is not provided, not specified, not mentioned or unknown."""

CONFIG = {
    "k": 8,              # 5 often found the right page but the wrong chunk of it
    "search": "hybrid",  # "vector", "bm25", or "hybrid" (both, merged)
    "candidates": 20,    # how many each search returns before merging
    "neighbors": 1,      # also read this many following chunks of the same page, 0 = off
    "page_images": 4,    # also read this many image descriptions from the retrieved pages, 0 = off
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
    r"|\bunknown\b|i don.?t know|i do not know|not sure",
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
    images = [im for im in get_bm25(store).images_on(question, urls, n + len(chunks))
              if (im["metadata"]["url"], im["metadata"]["position"]) not in taken]
    return chunks + images[:n]


def search(store, question: str, k: int, where: dict = None) -> list[dict]:
    if CONFIG["search"] == "vector":
        return query(store, question, k=k, where=where)
    if CONFIG["search"] == "bm25":
        return get_bm25(store).search(question, k=k, where=where)

    n = CONFIG["candidates"]
    ranked_lists = [
        query(store, question, k=n, where=where),
        get_bm25(store).search(question, k=n, where=where),
    ]
    scores, chunks = {}, {}
    for ranked in ranked_lists:
        for rank, c in enumerate(ranked):
            key = (c["metadata"].get("url"), c["metadata"].get("position"))
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            chunks.setdefault(key, c)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
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
        # An image description is complete on its own; the "next chunk"
        # would be a different image of the same page.
        steps = 0 if c["metadata"].get("kind") == "image" else n
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
    chunks = retrieve(question)

    # TODO [W2 b4] once this works: decompose compound questions,
    # TODO retrieve wide then filter, or filter by metadata before
    # TODO searching.

    context = "\n\n".join(
        f"[{c['metadata'].get('url', '?')}]\n{c['text']}" for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
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
