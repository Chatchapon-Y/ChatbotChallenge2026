"""Embed every dev question once, so retrieval experiments cost no quota.

    python exp_embed_cache.py      -> data_exp/question_embeddings.json

Only questions not already in the cache are sent. Not part of the bot.
"""
import json
from pathlib import Path

from bot.llm import embed

OUT = Path("data_exp/question_embeddings.json")
cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
questions = []
for f in ("my_dev.json", "dev_set.json", "final_dev.json", "lv5_dev.json"):
    if Path(f).exists():
        questions += [d["question"] for d in json.loads(Path(f).read_text(encoding="utf-8"))]
todo = sorted(set(questions) - set(cache))
if todo:
    cache.update(zip(todo, embed(todo)))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(cache), encoding="utf-8")
print(f"embedded {len(todo)} new questions, {len(cache)} cached in {OUT}")
