"""Final round: type a question, get the answer. Not used by the grader.

main.py opens the store and builds the BM25 index again for every
question, a few seconds each time. This opens them once and then keeps
asking, so an answer comes back in the time the models take.

    python ask.py            check the key and network, then ask questions
    python ask.py --check    only check the key and network (two tiny calls)

An empty line or Ctrl+C quits. Every question and answer is appended to
runs/final_session.jsonl.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from bot.answer import rag_answer
from bot.bm25 import get_bm25
from bot.llm import chat, embed
from bot.store import get_store

LOG = Path("runs/final_session.jsonl")


def check() -> bool:
    """One embedding and one chat call: is the key valid and the gateway reachable?"""
    try:
        start = time.time()
        embed(["ping"])
        chat([{"role": "user", "content": "Reply with OK"}], max_tokens=5)
        print(f"API OK ({time.time() - start:.1f}s)")
        return True
    except Exception as exc:
        print(f"API FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        print("Check Wi-Fi (HKU network or HKUVPN), and that .env is in this folder.")
        return False


def main() -> None:
    # Windows consoles default to a code page that cannot print every answer.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = check()
    if "--check" in sys.argv:
        sys.exit(0 if ok else 1)

    start = time.time()
    store = get_store()
    get_bm25(store)
    print(f"store ready: {store.count()} chunks ({time.time() - start:.1f}s)\n")

    LOG.parent.mkdir(exist_ok=True)
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        start = time.time()
        try:
            answer = rag_answer(question)
        except Exception as exc:        # the network can drop mid-event
            answer = ""
            print(f"[error] {type(exc).__name__}: {str(exc)[:200]}")
        seconds = time.time() - start
        print(f"\n  {answer}\n  ({seconds:.1f}s)\n")
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                "question": question, "answer": answer,
                                "seconds": round(seconds, 1)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
