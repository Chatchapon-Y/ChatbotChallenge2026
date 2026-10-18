"""Run a question set, time each answer, and show whether retrieval found it.

Run from submission_repo:
    python eval_dev.py          # dev_set.json from the organisers (9 questions)
    python eval_dev.py lv2      # dev_set_lv2.json, our own level 2 questions
    python eval_dev.py lv2 bm25 # same, with one search mode: vector, bm25 or hybrid
    python eval_dev.py lv2 noneighbors   # without reading the next chunk
    python eval_dev.py mock_sets/mock1.json   # any question file, here a mock preliminary

Mark each answer yourself (1 / 0.5 / 0). Both checks below are crude,
a stricter version of the lab 2 test: the whole expected answer, or 80%
of its longer words and all its numbers, somewhere in the text. They
miss "7" vs "seven" and can pass when the words appear in an unrelated
sentence.

  retrieval FOUND  the answer is in the retrieved chunks
  bot MATCH        the answer is in the bot's reply
"""
import json
import re
import sys
import time

from bot.answer import CONFIG, rag_answer, retrieve


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def answer_present(expected: str, text: str) -> bool:
    hay = normalise(text)
    need = " ".join(normalise(expected).split())
    # Whole words only, so "7" does not match "2017".
    if need and re.search(rf"\b{re.escape(need)}\b", hay):
        return True
    # Numbers always count as terms, so "3rd Prize" does not pass on
    # "Prize" alone. A term may start a word: "27" matches "27th".
    terms = [t for t in need.split() if len(t) > 3 or any(ch.isdigit() for ch in t)]
    hits = sum(bool(re.search(rf"\b{re.escape(t)}", hay)) for t in terms)
    return bool(terms) and hits / len(terms) >= 0.8


def short_url(url: str) -> str:
    return url.split(".hk")[-1]


def run_eval(path: str) -> None:
    dev = json.load(open(path, encoding="utf-8"))
    print(f"{path}   k={CONFIG['k']}   search={CONFIG['search']}   "
          f"neighbors={CONFIG['neighbors']}\n")

    found_count, match_count, slowest = 0, 0, 0.0
    for d in dev:
        start = time.time()
        reply = rag_answer(d["question"])
        seconds = time.time() - start
        slowest = max(slowest, seconds)

        chunks = retrieve(d["question"])
        found = answer_present(d["answer"], " ".join(c["text"] for c in chunks))
        match = answer_present(d["answer"], reply)
        found_count += found
        match_count += match

        print(f"Lv{d['level']} | {seconds:4.1f}s | retrieval {'FOUND  ' if found else 'missing'} "
              f"| bot {'MATCH' if match else 'wrong'} | {d['question']}")
        print(f"    bot:      {reply}")
        print(f"    expected: {d['answer']}")
        urls = sorted({short_url(c['metadata'].get('url', '?')) for c in chunks})
        print(f"    top urls: {', '.join(urls)}")
        if "source" in d:
            hit = "yes" if short_url(d["source"]) in urls else "NO"
            print(f"    source:   {short_url(d['source'])}   in top urls: {hit}")
        print()

    print(f"retrieval FOUND {found_count}/{len(dev)}   bot MATCH {match_count}/{len(dev)}   "
          f"slowest {slowest:.1f}s")
    if slowest > 30:
        print("WARNING: at least one question is over the 30 second limit.")


def eval_dev() -> None:
    """The organisers' 9 questions, all levels."""
    run_eval("dev_set.json")


def eval_lv2() -> None:
    """Our own level 2 questions, each with the page its answer is on."""
    run_eval("dev_set_lv2.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    for mode in ("vector", "bm25", "hybrid"):
        if mode in args:
            CONFIG["search"] = mode
    if "noneighbors" in args:
        CONFIG["neighbors"] = 0
    paths = [a for a in args if a.endswith(".json")]
    if paths:
        for path in paths:
            run_eval(path)
    elif "lv2" in args:
        eval_lv2()
    else:
        eval_dev()