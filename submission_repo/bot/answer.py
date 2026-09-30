"""Your chatbot. STUB. This is the file you actually write.

Two functions must exist with these exact names and signatures. Every
other line in this file, and every file under build/, is yours to
rewrite.
"""
import re

from bot.bm25 import get_bm25
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

    # TODO check the format of what came back: if you asked for a
    # TODO number, make sure you got one, and strip any stray prose.
    return reply.strip()


def rag_answer_batch(questions: list[str]) -> list[str]:
    """Many questions in, the same number of answers out, in order.

    Ships as a loop, which is correct and is all most teams will need.
    Replace it if you can share work across questions: one embedding
    call for every query rather than one per query, the store opened
    once, sub-queries running concurrently.

    Whatever you do, answers[i] must be the answer to questions[i].
    """
    return [rag_answer(q) for q in questions]
