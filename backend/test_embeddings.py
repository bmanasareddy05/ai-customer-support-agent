
from app.rag.document_loader import load_document
from app.rag.chunker import chunk_text
from app.rag.embedding_model import (
    generate_embeddings,
    generate_query_embedding
)

# Load the sample company policy
document = load_document("../knowledge_base/return_policy.txt")

# Split the document into chunks
chunks = chunk_text(
    document,
    chunk_size=200,
    chunk_overlap=40
)

# Generate document embeddings
embeddings = generate_embeddings(chunks)

print("\n--- DOCUMENT EMBEDDING RESULTS ---")
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Embedding dimensions:", embeddings.shape[1])

# Generate an embedding for a customer question
query = "How many days do customers have to return a product?"

query_embedding = generate_query_embedding(query)

print("\n--- QUERY EMBEDDING RESULTS ---")
print("Query embedding shape:", query_embedding.shape)

# Verify the output
assert embeddings.shape == (len(chunks), 384)
assert query_embedding.shape == (1, 384)

print("\nEmbedding generation successful!")
