
from app.rag.document_loader import load_document

file_path = "../knowledge_base/return_policy.txt"

content = load_document(file_path)

print("\n--- EXTRACTED DOCUMENT CONTENT ---\n")
print(content)

print("\nDocument loaded successfully!")
