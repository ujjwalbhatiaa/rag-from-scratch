"""BM25 retrieval over text chunks, implemented from scratch.

Okapi BM25 with the standard k1=1.5, b=0.75 defaults. Tokenization is a
simple lowercase alphanumeric regex — no dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25:
    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75
    _doc_terms: list[dict[str, int]] = field(default_factory=list,
                                            repr=False)
    _doc_freq: dict[str, int] = field(default_factory=dict, repr=False)
    _avgdl: float = 0.0

    def __post_init__(self) -> None:
        total_len = 0
        for chunk in self.chunks:
            terms: dict[str, int] = {}
            for tok in tokenize(chunk.text):
                terms[tok] = terms.get(tok, 0) + 1
            self._doc_terms.append(terms)
            total_len += sum(terms.values())
            for tok in terms:
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1
        self._avgdl = total_len / max(len(self.chunks), 1)

    def _idf(self, term: str) -> float:
        # Robertson/Sparck-Jones IDF with +1 smoothing floor at 0
        n = len(self.chunks)
        df = self._doc_freq.get(term, 0)
        return max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 0.0)

    def scores(self, query: str) -> list[float]:
        qterms = tokenize(query)
        out = []
        for terms in self._doc_terms:
            dl = sum(terms.values())
            s = 0.0
            for t in qterms:
                tf = terms.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += self._idf(t) * tf * (self.k1 + 1) / denom
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, score) pairs for the query."""
        ranked = sorted(zip(self.chunks, self.scores(query)),
                        key=lambda cs: cs[1], reverse=True)
        return [(c, s) for c, s in ranked[:k] if s > 0]
