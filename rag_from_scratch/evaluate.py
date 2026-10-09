"""Evaluation helpers for the RAG pipeline.

- recall@k: did retrieval surface the known-relevant chunks?
- citation_coverage: what fraction of the provided chunks does the generated
  answer actually cite? (a cheap faithfulness proxy)
"""

from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[(\d+)\]")


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids appearing in the top-k ranked ids."""
    if not relevant_ids:
        return 1.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def mean_recall_at_k(results: list[tuple[list[str], set[str]]],
                     k: int) -> float:
    """Average recall@k over (ranked_ids, relevant_ids) pairs."""
    if not results:
        return 0.0
    return sum(recall_at_k(r, rel, k) for r, rel in results) / len(results)


def cited_indices(answer: str) -> set[int]:
    """Chunk indices cited in the answer as [1], [2], ... (1-based)."""
    return {int(m.group(1)) for m in _CITE_RE.finditer(answer)}


def citation_coverage(answer: str, n_chunks: int) -> float:
    """Fraction of the n provided chunks cited at least once."""
    if n_chunks == 0:
        return 0.0
    return len(cited_indices(answer)) / n_chunks
