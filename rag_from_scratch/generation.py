"""Grounded answer generation.

Builds a citation-enforcing prompt from retrieved chunks and calls any
OpenAI-compatible chat-completions API via urllib (standard library only).
Without an API key it falls back to an extractive answer: the top chunks
themselves, clearly labeled.
"""

from __future__ import annotations

import json
import urllib.request

from .chunking import Chunk

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context chunks. "
    "Every factual claim in your answer must cite the chunk it came from "
    "with [1], [2], etc. matching the chunk numbers. If the context does "
    "not contain the answer, say so plainly instead of guessing."
)


def build_prompt(query: str, chunks: list[Chunk]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) with numbered context chunks."""
    context = "\n\n".join(
        f"[{i + 1}] (source: {c.source})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    user = (f"Context:\n{context}\n\nQuestion: {query}\n\n"
            f"Answer with citations like [1], [2].")
    return SYSTEM_PROMPT, user


def _call_api(system: str, user: str, api_key: str,
              model: str, base_url: str, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def extractive_answer(chunks: list[Chunk]) -> str:
    """Fallback when no API key is set: return the top chunks verbatim."""
    parts = [f"[{i + 1}] (source: {c.source})\n{c.text}"
             for i, c in enumerate(chunks)]
    return ("(extractive fallback — set an API key for generated answers)\n\n"
            + "\n\n".join(parts))


def generate(query: str, chunks: list[Chunk], api_key: str | None = None,
             model: str = "gpt-4o-mini",
             base_url: str = "https://api.openai.com/v1") -> str:
    """Answer the query from the chunks, grounded with citations."""
    if not chunks:
        return "I couldn't find anything relevant in the documents."
    if not api_key:
        return extractive_answer(chunks)
    system, user = build_prompt(query, chunks)
    return _call_api(system, user, api_key, model, base_url)
