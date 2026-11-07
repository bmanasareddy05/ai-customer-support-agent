
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.rag.vector_store import search_faiss


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(
    query: str,
    chunks: list[str],
    top_k: int = 3
) -> list[dict]:
    """Retrieve chunks using keyword-based BM25 search."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not chunks:
        raise ValueError("Chunks cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    tokenized_query = tokenize(query)

    if not tokenized_query:
        raise ValueError("Query contains no searchable terms.")

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": idx,
            "content": chunks[idx],
            "bm25_score": float(scores[idx])
        }
        for idx in ranked_indices
    ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60
) -> list[dict]:
    """Combine dense and sparse rankings using RRF."""

    if k <= 0:
        raise ValueError("RRF constant must be greater than zero.")

    combined = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]

        combined.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "content": result["content"],
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }
        )

        combined[chunk_id]["dense_score"] = result[
            "similarity_score"
        ]

        combined[chunk_id]["rrf_score"] += 1 / (k + rank)

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["chunk_id"]

        combined.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "content": result["content"],
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }
        )

        combined[chunk_id]["bm25_score"] = result["bm25_score"]

        combined[chunk_id]["rrf_score"] += 1 / (k + rank)

    return sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )


def hybrid_search(
    query: str,
    index,
    chunks: list[str],
    top_k: int = 3
) -> list[dict]:
    """Retrieve and fuse results from FAISS and BM25."""

    dense_results = search_faiss(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    sparse_results = search_bm25(
        query=query,
        chunks=chunks,
        top_k=top_k
    )

    return reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )[:top_k]
