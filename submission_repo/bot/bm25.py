"""BM25 keyword search in plain Python, so the grader needs no extra package.

Vector search matches meaning and is weak on exact tokens: it treats
"RoboMaster 2019" and "RoboMaster 2020" as nearly the same thing. BM25
scores exact word matches, and rare words ("2019", "Timothy") count for
more than common ones ("the", "team"). answer.py runs both and merges
the two rankings.

The index is built from the chunks already in Chroma, the first time a
question needs it, and cached for the rest of the process.
"""
import math
import re
from collections import Counter, defaultdict

K1 = 1.5    # how quickly repeating a word stops adding score
B = 0.75    # how much long chunks are penalised

STOPWORDS = set("""
a an and are as at be by did do does for from had has have how in is it its
of on or that the their this to was were what when where which who whom why
will with you your
""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], metas: list[dict]):
        self.metas = metas
        self.docs = docs
        self.postings = defaultdict(list)       # word -> [(doc index, count)]
        self.lengths = []
        for i, doc in enumerate(docs):
            words = tokenize(doc)
            self.lengths.append(len(words))
            for word, count in Counter(words).items():
                self.postings[word].append((i, count))
        n = len(docs)
        self.avg_len = sum(self.lengths) / max(n, 1)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in self.postings.items()}

    def search(self, question: str, k: int = 20, where: dict = None) -> list[dict]:
        scores = defaultdict(float)
        for word in set(tokenize(question)):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for i, count in self.postings[word]:
                norm = count + K1 * (1 - B + B * self.lengths[i] / self.avg_len)
                scores[i] += idf * count * (K1 + 1) / norm

        if where:
            # Plain equality only, e.g. {"year": "2019"}. That covers the
            # filters this project uses; Chroma's $and/$or are not handled.
            scores = {i: s for i, s in scores.items()
                      if all(self.metas[i].get(key) == value for key, value in where.items())}

        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [{"text": self.docs[i], "metadata": self.metas[i], "score": scores[i]}
                for i in best]


_index = None


def get_bm25(store) -> BM25:
    global _index
    if _index is None:
        everything = store.get(include=["documents", "metadatas"])
        _index = BM25(everything["documents"], everything["metadatas"])
    return _index
