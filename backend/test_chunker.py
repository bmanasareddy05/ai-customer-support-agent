
from app.rag.document_loader import load_document
from app.rag.chunker import chunk_text

# Load our sample company policy
text = load_document("../knowledge_base/return_policy.txt")

# Divide it into overlapping chunks
chunks = chunk_text(
    text,
    chunk_size=200,
    chunk_overlap=40
)

print(f"\nTotal chunks generated: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {index} ---")
    print(chunk)
    print(f"Characters: {len(chunk)}")

print("\nChunking completed successfully!")
