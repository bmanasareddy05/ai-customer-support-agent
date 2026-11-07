
import json
from pathlib import Path

import faiss
import numpy as np

from app.rag.embedding_model import (
    generate_embeddings,
    generate_query_embedding
)


def create_faiss_index(chunks: list[str]):
    """Create a FAISS index from document chunks."""

    if not chunks:
        raise ValueError("Cannot create an index from empty chunks.")

    embeddings = generate_embeddings(chunks)

    dimension = embeddings.shape[1]

    # Normalized vectors + inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype(np.float32))

    return index


def search_faiss(
    query: str,
    index,
    chunks: list[str],
    top_k: int = 3
) -> list[dict]:
    """Retrieve the most relevant chunks for a customer query."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count do not match.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = generate_query_embedding(query)
    query_embedding = query_embedding.astype(np.float32)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "chunk_id": int(idx),
            "content": chunks[idx],
            "similarity_score": float(score)
        })

    return results


def save_vector_store(
    index,
    chunks: list[str],
    storage_dir: str
):
    """Persist the FAISS index and its associated chunks."""

    directory = Path(storage_dir)
    directory.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(directory / "index.faiss")
    )

    with open(directory / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_vector_store(storage_dir: str):
    """Load a previously saved FAISS index and its chunks."""

    directory = Path(storage_dir)

    index_path = directory / "index.faiss"
    chunks_path = directory / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Saved vector store not found.")

    index = faiss.read_index(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("Saved index and chunks are inconsistent.")

    return index, chunks
