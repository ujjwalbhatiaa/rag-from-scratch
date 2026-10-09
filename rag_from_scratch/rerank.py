"""Lightweight reranking: BM25 fused with bigram-overlap.

A cheap cross-check before generation — chunks that share distinctive
two-word phrases with the query get boosted. No model required.
"""

from __future__ import annotations

from .chunking import Chunk
from .retrieval import tokenize


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def bigram_overlap(query: str, text: str) -> float:
    """Fraction of query bigrams appearing in the text (0..1)."""
    q = _bigrams(tokenize(query))
    if not q:
        return 0.0
    t = _bigrams(tokenize(text))
    return len(q & t) / len(q)


def rerank(query: str, candidates: list[tuple[Chunk, float]],
           k: int = 3, alpha: float = 0.7) -> list[tuple[Chunk, float]]:
    """Fuse normalized BM25 (weight alpha) with bigram overlap (1 - alpha)."""
    if not candidates:
        return []
    top = max(s for _, s in candidates) or 1.0
    fused = []
    for chunk, score in candidates:
        norm = score / top
        boost = bigram_overlap(query, chunk.text)
        fused.append((chunk, alpha * norm + (1 - alpha) * boost))
    fused.sort(key=lambda cs: cs[1], reverse=True)
    return fused[:k]
