"""Lightweight reranking: BM25 fused with bigram-overlap.

A cheap cross-check before generation — chunks that share distinctive
two-word phrases with the query get boosted. No model required.
"""

from __future__ import annotations

import math

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


def _term_vector(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity of two token-count vectors (0..1)."""
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())
                     * sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def mmr_rerank(query: str, candidates: list[tuple[Chunk, float]],
               k: int = 3, lambda_: float = 0.7) -> list[tuple[Chunk, float]]:
    """Maximal Marginal Relevance: relevance plus diversity.

    Greedily selects k chunks maximizing
        lambda_ * relevance - (1 - lambda_) * max_similarity_to_selected,
    where relevance is the candidate's retrieval score normalized to [0, 1]
    and similarity is cosine similarity over token-count vectors between
    chunks. lambda_=1.0 is pure relevance; lambda_=0.0 is pure diversity.

    `query` is accepted for drop-in compatibility with rerank(); relevance
    comes from the candidate scores. Returns (chunk, mmr_score) pairs in
    selection order.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if not candidates or k <= 0:
        return []
    top = max(s for _, s in candidates) or 1.0
    rel = [score / top for _, score in candidates]
    vectors = [_term_vector(tokenize(chunk.text)) for chunk, _ in candidates]

    picked: list[tuple[int, float]] = []  # (candidate index, mmr score)
    remaining = set(range(len(candidates)))
    while remaining and len(picked) < k:
        best_i, best_score = -1, float("-inf")
        for i in sorted(remaining):  # sorted = deterministic tie-breaks
            penalty = max((_cosine(vectors[i], vectors[j]) for j, _ in picked),
                          default=0.0)
            score = lambda_ * rel[i] - (1.0 - lambda_) * penalty
            if score > best_score:
                best_i, best_score = i, score
        picked.append((best_i, best_score))
        remaining.discard(best_i)
    return [(candidates[i][0], s) for i, s in picked]
