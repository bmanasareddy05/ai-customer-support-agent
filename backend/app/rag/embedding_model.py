
from sentence_transformers import SentenceTransformer
import numpy as np


# Load the model once and reuse it.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into normalized embeddings.

    Returns:
        NumPy array with shape (number_of_texts, 384).
    """

    if not texts:
        raise ValueError("Input text list cannot be empty.")

    if any(not text.strip() for text in texts):
        raise ValueError("Input texts cannot contain empty strings.")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate an embedding for a customer query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embedding
