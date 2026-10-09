"""CLI: ingest documents into an index, then ask questions over them.

    python -m rag_from_scratch ingest sample_docs/ --index index.json
    python -m rag_from_scratch ask "What is the refund policy?" --index index.json
    python -m rag_from_scratch ask "..." --index index.json --api-key $OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .chunking import Chunk, split_documents
from .retrieval import BM25
from .rerank import rerank
from .generation import generate


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
    top = rerank(args.query, candidates, k=args.final_k)
    answer = generate(args.query, [c for c, _ in top],
                      api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
    print(answer)
    if args.show_sources:
        print("\nSources:")
        for c, s in top:
            print(f"  [{c.id}] score={s:.3f}")
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

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return cmd_ingest(args)
    return cmd_ask(args)


if __name__ == "__main__":
    raise SystemExit(main())
