"""Recursive text chunking."""
import pandas as pd

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _merge(pieces, sep, chunk_size, overlap):
    """Join small pieces into chunks <= chunk_size, keeping `overlap` chars of context."""
    chunks, current = [], []
    for piece in pieces:
        if current and len(sep.join(current + [piece])) > chunk_size:
            chunks.append(sep.join(current))
            # drop the oldest pieces until the overlap fits and the new piece fits
            while current and (
                len(sep.join(current)) > overlap
                or len(sep.join(current + [piece])) > chunk_size
            ):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(sep.join(current))
    return chunks


def chunk_text(text, chunk_size=800, overlap=100, separators=SEPARATORS):
    """Split text into chunks of at most `chunk_size` characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = next((s for s in separators if s in text), "")
    if sep == "":  # no separator at all: hard cut
        step = chunk_size - overlap
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    rest = separators[separators.index(sep) + 1:]
    pieces = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if len(part) > chunk_size:  # too big: split it with the next separator
            pieces.extend(chunk_text(part, chunk_size, overlap, rest))
        else:
            pieces.append(part)
    return _merge(pieces, sep, chunk_size, overlap)


def chunk_documents(df, chunk_size=800, overlap=100):
    """Turn the corpus DataFrame into one row per chunk, keeping source metadata."""
    rows = []
    for doc in df.itertuples():
        for i, chunk in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            rows.append({
                "chunk_id": f"{doc.id}_{i}",
                "doc_id": doc.id,
                "chunk_index": i,
                "text": chunk,
                "title": doc.title,
                "categories": ", ".join(doc.categories),
                "url": doc.url,
            })
    return pd.DataFrame(rows)