
from app.rag.document_loader import load_document
from app.rag.chunker import chunk_text
from app.rag.vector_store import (
    create_faiss_index,
    save_vector_store
)
from app.rag.hybrid_retriever import (
    search_bm25,
    hybrid_search
)

# Load and chunk the company policy
document = load_document("../knowledge_base/return_policy.txt")

chunks = chunk_text(
    document,
    chunk_size=200,
    chunk_overlap=40
)

# Build the FAISS index
index = create_faiss_index(chunks)

save_vector_store(
    index,
    chunks,
    "../knowledge_base/vector_store"
)

query = "How many days do I have to return a product?"

# Test BM25 independently
bm25_results = search_bm25(
    query=query,
    chunks=chunks,
    top_k=3
)

print("\n--- BM25 RESULTS ---")

for result in bm25_results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"BM25 Score: {result['bm25_score']:.4f}")
    print(result["content"])

# Test hybrid retrieval
results = hybrid_search(
    query=query,
    index=index,
    chunks=chunks,
    top_k=3
)

print("\n--- HYBRID RETRIEVAL RESULTS ---")

for result in results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"Dense Score: {result['dense_score']}")
    print(f"BM25 Score: {result['bm25_score']}")
    print(f"RRF Score: {result['rrf_score']:.6f}")
    print(result["content"])

assert len(results) > 0
assert results[0]["rrf_score"] > 0

print("\nHybrid retrieval working successfully!")
