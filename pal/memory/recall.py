"""Finding earlier events, turns and episodes again: keyword and vector search, combined.

Every entry is one line of text (the same wording the memory updaters see) with
its time. Keyword search (BM25) catches exact names and numbers ("pallet seven",
"Mark"); vector search catches the same thing said differently ("damaged
crate" for "crushed corner"). The two rankings are merged with reciprocal rank
fusion. Embedding happens in the background as entries come in, so a search
only has to embed the query.

Similarity alone can't tell a real match from noise: with a small embedding
model, unrelated text still scores 0.6-0.8 against a query. What separates them
is how far the best matches stand out from the typical score, so a vector hit
only counts when it clears the median by MIN_MARGIN. That way a query with no
answer returns nothing, which matters for automatic recall.
"""
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import numpy as np

from pal.llm.embedding_client import EmbeddingClient

STOPWORDS = set("""a about all an and any are as at be but by can could did do does for from get got had has have
he her here his how i if in is it its just me my no not now of on or our she should so some than that the their them
then there they this to up was we were what when where which who why will with would you your buddy""".split())
RRF_K = 60        # reciprocal rank fusion constant; higher flattens the difference between ranks
MIN_MARGIN = 0.08  # a vector hit must beat the median similarity by this much (real answers: 0.10-0.19)


@dataclass(frozen=True)
class Entry:
    id: UUID
    at: datetime | None  # None for reference material, which has no time
    kind: str            # "event", "turn", "episode" or "knowledge"
    text: str


@dataclass(frozen=True)
class Hit:
    entry: Entry
    score: float       # fused rank score, higher is better
    similarity: float  # cosine similarity to the query
    keyword: float     # BM25 score, 0 if no keyword matched


class RecallIndex:
    def __init__(self, embedder: EmbeddingClient | None):
        self.embedder = embedder
        self._entries: list[Entry] = []
        self._position: dict[UUID, int] = {}
        self._tokens: list[list[str]] = []
        self._vectors: list[np.ndarray | None] = []
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._entries)

    def add(self, entries: list[Entry], vectors: np.ndarray | None = None):
        """Add new entries; an entry whose text changed (an extended episode) is replaced.
        `vectors`, one row per entry, skips embedding them (e.g. from a cache)."""
        with self._lock:
            for n, e in enumerate(entries):
                i = self._position.get(e.id)
                if i is not None and self._entries[i].text == e.text:
                    continue
                vector = vectors[n] if vectors is not None else None
                if i is None:
                    self._position[e.id] = len(self._entries)
                    self._entries.append(e)
                    self._tokens.append(_tokenize(e.text))
                    self._vectors.append(vector)
                else:
                    self._entries[i], self._tokens[i], self._vectors[i] = e, _tokenize(e.text), vector

    def embed_pending(self):
        """Embed entries that don't have a vector yet. Meant to run in the background."""
        if self.embedder is None:
            return
        with self._lock:
            pending = [(i, self._entries[i].text) for i, v in enumerate(self._vectors) if v is None]
        if not pending:
            return
        vectors = self.embedder.vectorise([text for _, text in pending])
        with self._lock:
            for (i, text), v in zip(pending, vectors):
                if self._entries[i].text == text:  # not replaced in the meantime
                    self._vectors[i] = v

    def search(self, query: str, k: int = 5, exclude: set = frozenset()) -> list[Hit]:
        with self._lock:
            entries, tokens, vectors = list(self._entries), list(self._tokens), list(self._vectors)
        candidates = [i for i, e in enumerate(entries) if e.id not in exclude]
        if not candidates:
            return []

        keyword = _bm25(_tokenize(query), [tokens[i] for i in candidates])
        similarity = np.zeros(len(candidates), dtype=np.float32)
        if self.embedder is not None:
            q = self.embedder.vectorise([query])[0]
            similarity = np.array([float(vectors[i] @ q) if vectors[i] is not None else 0.0 for i in candidates])

        def ranks(scores, keep):
            order = [j for j in np.argsort(-np.asarray(scores)) if keep(j)]
            return {j: r for r, j in enumerate(order, 1)}
        by_keyword = ranks(keyword, lambda j: keyword[j] > 0)
        typical = float(np.median(similarity)) if len(similarity) else 0.0
        by_vector = ranks(similarity, lambda j: similarity[j] - typical >= MIN_MARGIN)

        fused = {j: sum(1 / (RRF_K + r[j]) for r in (by_keyword, by_vector) if j in r)
                 for j in set(by_keyword) | set(by_vector)}
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(entries[candidates[j]], fused[j], float(similarity[j]), float(keyword[j])) for j in best]


def format_hits(hits: list[Hit]) -> str:
    """One line per hit, oldest first, as the response model and the recall tool see them."""
    if not hits:
        return "Nothing relevant found."
    return "\n".join(f"[{h.entry.at.astimezone():%H:%M}] {h.entry.text}" for h in sorted(hits, key=lambda h: h.entry.at))


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def _stem(word: str) -> str:
    """Light suffix stripping, so "drivers" matches "driver" and "checking" matches "check"."""
    for suffix, keep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x"),
                         ("ing", ""), ("ed", ""), ("s", "")):
        stem_length = len(word) - len(suffix) + len(keep)
        if word.endswith(suffix) and stem_length >= 3 and not (suffix == "s" and word.endswith("ss")):
            return word[:-len(suffix)] + keep
    return word


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not query or not docs:
        return [0.0] * len(docs)
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(w for d in docs for w in set(d))
    idf = {w: math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1) for w in set(query)}
    scores = []
    for d in docs:
        tf = Counter(d)
        scores.append(sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg_len))
                          for w in idf if tf[w]))
    return scores
