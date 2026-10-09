"""rag-from-scratch: a from-scratch RAG pipeline in pure Python.

Chunking, BM25 retrieval, reranking, grounded generation, and evaluation —
no LangChain, no framework, standard library only (plus an optional LLM
API key for the generation step).
"""

from .chunking import Chunk, split_text, split_documents
from .retrieval import BM25
from .rerank import rerank
from .generation import build_prompt, generate, extractive_answer
from .evaluate import (mean_recall_at_k, recall_at_k, citation_coverage,
                         cited_indices)

__version__ = "0.1.0"
__all__ = ["Chunk", "split_text", "split_documents", "BM25", "rerank",
           "build_prompt", "generate", "extractive_answer",
           "mean_recall_at_k", "recall_at_k", "citation_coverage",
           "cited_indices"]
