# rag-from-scratch

A complete retrieval-augmented generation (RAG) pipeline built from scratch
in pure Python — no LangChain, no framework. Just the standard library
(plus an optional LLM API key for the generation step).

## Pipeline

```text
.txt documents
    → chunking (recursive splitter, overlapping chunks)
    → BM25 retrieval (Okapi BM25, from scratch)
    → reranking (BM25 fused with bigram-overlap, or MMR for diversity)
    → generation (grounded prompt, citations enforced)
    → evaluation (recall@k, citation coverage)
```

## Usage

```bash
# 1. Ingest documents into an index
python -m rag_from_scratch ingest sample_docs/ --index index.json

# 2. Ask questions (extractive fallback when no API key is set)
python -m rag_from_scratch ask "What is the refund policy?" --index index.json

# 3. With an OpenAI-compatible API key for generated answers
python -m rag_from_scratch ask "How much is express shipping?" \
    --index index.json --api-key $OPENAI_API_KEY --show-sources

# 4. Score retrieval + reranking on a labeled eval set (JSONL:
#    one {"query": ..., "relevant_sources": [...]} per line)
python -m rag_from_scratch eval --index index.json --evalset eval.jsonl \
    --threshold 0.8
```

Or use it as a library:

```python
from rag_from_scratch import split_documents, BM25, rerank, mmr_rerank, generate

chunks = split_documents({"guide.txt": open("guide.txt").read()})
retriever = BM25(chunks)
candidates = retriever.search("how do returns work?", k=8)
top = rerank("how do returns work?", candidates, k=3)
# or diversify with Maximal Marginal Relevance:
top = mmr_rerank("how do returns work?", candidates, k=3, lambda_=0.7)
print(generate("how do returns work?", [c for c, _ in top],
               api_key="sk-..."))
```

## Design notes

- **Chunking**: recursive split on paragraph → line → sentence → word
  boundaries, with configurable overlap. Oversized pieces recurse on the
  next-finer separator instead of being hard-cut.
- **Retrieval**: Okapi BM25 (k1=1.5, b=0.75) with Robertson/Sparck-Jones IDF.
- **Reranking**: normalized BM25 fused 70/30 with query–chunk bigram overlap —
  a cheap cross-check that catches phrase-level matches BM25 undervalues.
  `mmr_rerank` offers the classic alternative: Maximal Marginal Relevance,
  greedily trading relevance against cosine-similarity redundancy
  (`lambda_` in [0, 1]); exposed on the CLI as `ask --mmr --mmr-lambda 0.7`.
- **Generation**: the system prompt forces answers to cite `[1]`, `[2]`, …
  and to say "I don't know" when the context lacks the answer. Without an
  API key, it degrades honestly to an extractive answer (top chunks verbatim).
- **Evaluation**: `recall_at_k` over labeled query→doc pairs, plus
  `citation_coverage` — the fraction of provided chunks the answer cites.
  The `eval` CLI command runs a labeled JSONL set through the same
  retrieval+rerank path as `ask` and reports per-query and mean recall@k
  and MRR (mean reciprocal rank), exiting nonzero when mean recall@k falls
  below `--threshold`.

## Tests

```bash
pytest
```

31 tests: chunking (sizes, overlap, empty input), BM25 ranking and edge
cases, reranking (fusion + MMR: near-duplicate demotion, pure-relevance
order, lambda bounds, edge cases), prompt construction, evaluation metrics
(recall@k, reciprocal rank/MRR, citation coverage), an end-to-end recall
check on the sample docs, CLI ingest→ask round-trips, and CLI `eval`
round-trips (labeled JSONL scoring, `--threshold` failure).
