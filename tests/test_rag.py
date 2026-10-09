"""Tests for rag-from-scratch. Run with: pytest (no network needed)."""

import json
import os
import tempfile

from rag_from_scratch import (
    BM25,
    Chunk,
    build_prompt,
    citation_coverage,
    cited_indices,
    extractive_answer,
    generate,
    mean_recall_at_k,
    recall_at_k,
    rerank,
    split_documents,
    split_text,
)
from rag_from_scratch.rerank import bigram_overlap


def _chunks():
    docs = {
        "returns.txt": "Returns are accepted within 30 days of delivery for a full refund.",
        "shipping.txt": "Standard shipping takes 3 to 5 business days and costs $6.95.",
        "membership.txt": "Northwind Plus costs $9.99 per month with free express shipping.",
    }
    return split_documents(docs, chunk_size=200, chunk_overlap=20)


# --- chunking ---------------------------------------------------------------

def test_split_short_text_single_chunk():
    chunks = split_text("hello world", chunk_size=500, source="a.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_split_long_text_respects_size():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=100, chunk_overlap=10, source="b.txt")
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)  # size + small merge slack
    assert all(c.text.strip() for c in chunks)


def test_split_empty():
    assert split_text("   ", source="c.txt") == []


def test_split_documents_ids_unique():
    chunks = _chunks()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# --- retrieval ----------------------------------------------------------------

def test_bm25_ranks_relevant_chunk_first():
    chunks = _chunks()
    bm25 = BM25(chunks)
    top = bm25.search("how long do returns take", k=3)
    assert top[0][0].source == "returns.txt"


def test_bm25_shipping_query():
    chunks = _chunks()
    bm25 = BM25(chunks)
    top = bm25.search("shipping cost standard", k=3)
    assert top[0][0].source == "shipping.txt"


def test_bm25_no_match_returns_empty():
    chunks = _chunks()
    bm25 = BM25(chunks)
    assert bm25.search("quantum astrophysics", k=3) == []


def test_bm25_scores_nonnegative():
    chunks = _chunks()
    bm25 = BM25(chunks)
    assert all(s >= 0 for s in bm25.scores("refund policy"))


# --- rerank -------------------------------------------------------------------

def test_bigram_overlap_exact():
    assert bigram_overlap("refund policy", "our refund policy allows returns") == 1.0


def test_bigram_overlap_partial():
    assert bigram_overlap("express shipping cost", "express shipping rates") == 0.5


def test_rerank_keeps_k():
    docs = {
        "a.txt": "The membership price is nine ninety-nine per month.",
        "b.txt": "Membership includes free express shipping every month.",
        "c.txt": "Unrelated text about gardening and soil.",
    }
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    bm25 = BM25(chunks)
    cands = bm25.search("membership price per month", k=3)
    assert len(cands) >= 2
    top = rerank("membership price per month", cands, k=2)
    assert len(top) == 2
    assert top[0][0].source == "a.txt"


def test_rerank_empty():
    assert rerank("q", [], k=3) == []


# --- generation ---------------------------------------------------------------

def test_build_prompt_numbers_chunks():
    chunks = _chunks()[:2]
    system, user = build_prompt("What is the policy?", chunks)
    assert "[1]" in user and "[2]" in user
    assert "ONLY" in system  # grounded instruction present


def test_extractive_fallback_labels_chunks():
    ans = extractive_answer(_chunks()[:2])
    assert "[1]" in ans and "[2]" in ans


def test_generate_no_chunks():
    assert "couldn't find" in generate("q", [])


def test_generate_no_key_uses_fallback():
    ans = generate("q", _chunks()[:1], api_key=None)
    assert "extractive fallback" in ans


# --- evaluation ---------------------------------------------------------------

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == 0.5
    assert recall_at_k(["a"], set(), k=1) == 1.0


def test_mean_recall_at_k():
    pairs = [(["a", "b"], {"a"}), (["c", "d"], {"x"})]
    assert mean_recall_at_k(pairs, k=2) == 0.5


def test_cited_indices():
    assert cited_indices("see [1] and [3]") == {1, 3}
    assert cited_indices("no citations") == set()


def test_citation_coverage():
    assert citation_coverage("answer [1] more [2]", 4) == 0.5
    assert citation_coverage("answer", 4) == 0.0


def test_end_to_end_recall_on_sample_docs():
    docs = {
        "returns.txt": open("/tmp/rag-from-scratch/sample_docs/returns.txt").read(),
        "shipping.txt": open("/tmp/rag-from-scratch/sample_docs/shipping.txt").read(),
        "membership.txt": open("/tmp/rag-from-scratch/sample_docs/membership.txt").read(),
    }
    chunks = split_documents(docs, chunk_size=400, chunk_overlap=40)
    bm25 = BM25(chunks)
    queries = [
        ("how do I return an item", "returns.txt"),
        ("how long does standard shipping take", "shipping.txt"),
        ("student discount membership", "membership.txt"),
    ]
    for q, want in queries:
        top = bm25.search(q, k=5)
        assert top, f"no results for {q!r}"
        assert top[0][0].source == want, f"{q!r} -> {top[0][0].source}"
    # and recall@5 still surfaces every relevant chunk for the returns query
    ranked = [c.id for c, _ in bm25.search("how do I return an item", k=5)]
    relevant = {c.id for c in chunks if c.source == "returns.txt"}
    assert recall_at_k(ranked, relevant, k=5) >= 0.5


def test_cli_ingest_and_ask(tmp_path):
    import subprocess
    import sys
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("The refund window is 30 days for unused items.")
    idx = tmp_path / "index.json"
    r = subprocess.run([sys.executable, "-m", "rag_from_scratch", "ingest",
                        str(docs), "--index", str(idx)],
                       capture_output=True, text=True, cwd="/tmp/rag-from-scratch")
    assert r.returncode == 0, r.stderr
    assert json.loads(idx.read_text())["chunks"]
    r = subprocess.run([sys.executable, "-m", "rag_from_scratch", "ask",
                        "refund window", "--index", str(idx)],
                       capture_output=True, text=True, cwd="/tmp/rag-from-scratch")
    assert r.returncode == 0, r.stderr
    assert "30 days" in r.stdout
