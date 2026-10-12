"""CLI: ingest documents into an index, then ask questions over them.

    python -m rag_from_scratch ingest sample_docs/ --index index.json
    python -m rag_from_scratch ask "What is the refund policy?" --index index.json
    python -m rag_from_scratch ask "..." --index index.json --api-key $OPENAI_API_KEY
    python -m rag_from_scratch eval --index index.json --evalset eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .chunking import Chunk, split_documents
from .retrieval import BM25
from .rerank import rerank, mmr_rerank
from .generation import generate
from .evaluate import recall_at_k, reciprocal_rank


def _load_index(path: str) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [Chunk(id=c["id"], text=c["text"], source=c["source"],
                  index=c["index"]) for c in data["chunks"]]


def cmd_ingest(args: argparse.Namespace) -> int:
    docs: dict[str, str] = {}
    for root, _, files in os.walk(args.docs_dir):
        for name in sorted(files):
            if name.endswith(".txt"):
                full = os.path.join(root, name)
                with open(full, encoding="utf-8") as fh:
                    docs[os.path.relpath(full, args.docs_dir)] = fh.read()
    if not docs:
        print(f"error: no .txt files under {args.docs_dir}", file=sys.stderr)
        return 1
    chunks = split_documents(docs, args.chunk_size, args.overlap)
    with open(args.index, "w", encoding="utf-8") as fh:
        json.dump({"chunks": [c.__dict__ for c in chunks]}, fh)
    print(f"ingested {len(docs)} documents -> {len(chunks)} chunks -> {args.index}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    chunks = _load_index(args.index)
    bm25 = BM25(chunks)
    candidates = bm25.search(args.query, k=args.retrieve_k)
    if args.mmr:
        top = mmr_rerank(args.query, candidates, k=args.final_k,
                         lambda_=args.mmr_lambda)
    else:
        top = rerank(args.query, candidates, k=args.final_k)
    answer = generate(args.query, [c for c, _ in top],
                      api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
    print(answer)
    if args.show_sources:
        print("\nSources:")
        for c, s in top:
            print(f"  [{c.id}] score={s:.3f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score retrieval + reranking on a labeled eval set.

    Eval set is JSONL, one query per line:
        {"query": "...", "relevant_sources": ["returns.txt"]}
    or, for finer control:
        {"query": "...", "relevant_ids": ["returns.txt#0", "returns.txt#1"]}

    Each query runs through the same BM25 + rerank/MMR path as `ask`,
    reporting recall@k and reciprocal rank, then a corpus-level mean.
    Exits 1 when mean recall@k drops below --threshold (handy for CI).
    """
    chunks = _load_index(args.index)
    bm25 = BM25(chunks)

    rows: list[tuple[int, dict]] = []
    with open(args.evalset, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"error: {args.evalset}:{lineno}: bad JSON ({exc})",
                      file=sys.stderr)
                return 2
            if "query" not in item:
                print(f"error: {args.evalset}:{lineno}: missing 'query'",
                      file=sys.stderr)
                return 2
            rows.append((lineno, item))
    if not rows:
        print("error: no queries in eval set", file=sys.stderr)
        return 2

    recalls: list[float] = []
    rrs: list[float] = []
    for lineno, item in rows:
        query = item["query"]
        rel_ids = item.get("relevant_ids")
        relevant = set(rel_ids) if rel_ids else {
            c.id for c in chunks
            if c.source in item.get("relevant_sources", [])
        }
        candidates = bm25.search(query, k=args.retrieve_k)
        if args.mmr:
            top = mmr_rerank(query, candidates, k=args.final_k,
                             lambda_=args.mmr_lambda)
        else:
            top = rerank(query, candidates, k=args.final_k)
        ranked = [c.id for c, _ in top]
        recall = recall_at_k(ranked, relevant, args.final_k)
        rr = reciprocal_rank(ranked, relevant)
        recalls.append(recall)
        rrs.append(rr)
        print(f"[{lineno}] recall@{args.final_k}={recall:.2f} "
              f"rr={rr:.2f}  {query!r}")

    mean_recall = sum(recalls) / len(recalls)
    mean_rr = sum(rrs) / len(rrs)
    print(f"eval: {len(rows)} queries | "
          f"mean recall@{args.final_k}={mean_recall:.2f} | "
          f"MRR={mean_rr:.2f}")
    if mean_recall < args.threshold:
        print(f"error: mean recall@{args.final_k}={mean_recall:.2f} "
              f"below threshold {args.threshold:.2f}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="From-scratch RAG over .txt docs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="chunk docs into an index file")
    p_ing.add_argument("docs_dir")
    p_ing.add_argument("--index", default="index.json")
    p_ing.add_argument("--chunk-size", type=int, default=500)
    p_ing.add_argument("--overlap", type=int, default=50)

    p_ask = sub.add_parser("ask", help="ask a question over the index")
    p_ask.add_argument("query")
    p_ask.add_argument("--index", default="index.json")
    p_ask.add_argument("--retrieve-k", type=int, default=8)
    p_ask.add_argument("--final-k", type=int, default=3)
    p_ask.add_argument("--api-key", default=None)
    p_ask.add_argument("--show-sources", action="store_true")
    p_ask.add_argument("--mmr", action="store_true",
                       help="rerank with MMR (relevance + diversity) "
                            "instead of BM25/bigram fusion")
    p_ask.add_argument("--mmr-lambda", type=float, default=0.7,
                       help="MMR relevance/diversity tradeoff in [0, 1]")

    p_ev = sub.add_parser("eval", help="score retrieval on a labeled eval set")
    p_ev.add_argument("--index", default="index.json")
    p_ev.add_argument("--evalset", required=True,
                      help="JSONL file: one {query, relevant_sources|"
                           "relevant_ids} per line")
    p_ev.add_argument("--retrieve-k", type=int, default=8)
    p_ev.add_argument("--final-k", type=int, default=3)
    p_ev.add_argument("--mmr", action="store_true")
    p_ev.add_argument("--mmr-lambda", type=float, default=0.7)
    p_ev.add_argument("--threshold", type=float, default=0.0,
                      help="fail (exit 1) when mean recall@k is below this")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "eval":
        return cmd_eval(args)
    return cmd_ask(args)


if __name__ == "__main__":
    raise SystemExit(main())
