
def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Extracted document text.
        chunk_size: Maximum target characters per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A list of text chunks.
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to avoid cutting a word in half.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - chunk_overlap

        # Ensure the loop always makes progress.
        start = max(next_start, start + 1)

    return chunks
