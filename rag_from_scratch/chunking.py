"""Recursive character chunking with overlap.

Long documents are split on paragraph, line, sentence, then word boundaries
so chunks stay coherent and under the target size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str        # e.g. "returns.txt#3"
    text: str
    source: str    # file the chunk came from
    index: int     # chunk number within the source


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _merge(pieces: list[str], sep: str, chunk_size: int,
           chunk_overlap: int) -> list[str]:
    """Greedily merge pieces into chunks no longer than chunk_size."""
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        cand = p if not cur else (cur + sep + p if sep else cur + p)
        if len(cand) > chunk_size and cur.strip():
            chunks.append(cur)
            tail = cur[-chunk_overlap:] if chunk_overlap else ""
            cur = (tail + sep + p) if tail and sep else p
        else:
            cur = cand
    if cur.strip():
        chunks.append(cur)
    return chunks


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50,
               source: str = "", _depth: int = 0) -> list[Chunk]:
    """Split text into overlapping chunks, recursing on finer separators."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(id=f"{source}#0", text=text, source=source, index=0)]

    if _depth >= len(_SEPARATORS):  # last resort: hard cut with overlap
        out, i, n = [], 0, 0
        while i < len(text):
            out.append(Chunk(id=f"{source}#{n}", text=text[i:i + chunk_size],
                             source=source, index=n))
            i += chunk_size - chunk_overlap
            n += 1
        return out

    sep = _SEPARATORS[_depth]
    done: list[str] = []       # finished chunk texts, in order
    pending: list[str] = []    # raw pieces awaiting merge

    def flush() -> None:
        for t in _merge(pending, sep, chunk_size, chunk_overlap):
            done.append(t)
        pending.clear()

    for p in text.split(sep):
        if len(p) > chunk_size:
            # oversized piece: flush what we have, then splice in the
            # recursively-split chunks (already sized; never re-merge them)
            flush()
            done.extend(c.text for c in split_text(
                p, chunk_size, chunk_overlap, source, _depth + 1))
        elif p.strip():
            pending.append(p)
    flush()

    return [Chunk(id=f"{source}#{i}", text=t, source=source, index=i)
            for i, t in enumerate(done)]


def split_documents(docs: dict[str, str], chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list[Chunk]:
    """Split a {source_name: text} mapping into chunks."""
    chunks: list[Chunk] = []
    for source, text in docs.items():
        chunks.extend(split_text(text, chunk_size, chunk_overlap, source))
    # re-id globally so ids are unique across sources
    for i, c in enumerate(chunks):
        c.id = f"{c.source}#{c.index}"
    return chunks
