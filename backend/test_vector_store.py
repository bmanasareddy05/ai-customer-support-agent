
from app.rag.document_loader import load_document
from app.rag.chunker import chunk_text
from app.rag.vector_store import (
    create_faiss_index,
    search_faiss,
    save_vector_store,
    load_vector_store
)

# Step 1: Load the company policy
document = load_document("../knowledge_base/return_policy.txt")

# Step 2: Generate chunks
chunks = chunk_text(
    document,
    chunk_size=200,
    chunk_overlap=40
)

# Step 3: Create FAISS index
index = create_faiss_index(chunks)

print("\n--- FAISS INDEX ---")
print("Total vectors:", index.ntotal)
print("Vector dimensions:", index.d)

# Step 4: Search using a customer question
query = "How many days do I have to return a product?"

results = search_faiss(
    query=query,
    index=index,
    chunks=chunks,
    top_k=3
)

print("\n--- SEARCH RESULTS ---")

for result in results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"Content:\n{result['content']}")

# Step 5: Save the index
storage_path = "../knowledge_base/vector_store"

save_vector_store(index, chunks, storage_path)

print("\nVector store saved successfully!")

# Step 6: Load it again to verify persistence
loaded_index, loaded_chunks = load_vector_store(storage_path)

print("Loaded vectors:", loaded_index.ntotal)

# Step 7: Search using the loaded index
loaded_results = search_faiss(
    query="How long does a refund take?",
    index=loaded_index,
    chunks=loaded_chunks,
    top_k=2
)

print("\n--- SEARCH AFTER RELOADING ---")

for result in loaded_results:
    print(f"\nSimilarity: {result['similarity_score']:.4f}")
    print(result["content"])

assert loaded_index.ntotal == len(loaded_chunks)
assert len(results) == 3
assert len(loaded_results) == 2

print("\nFAISS vector search working successfully!")
