from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.data_loader import Document


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


def build_chunks(docs: Iterable[Document], chunk_size: int = 600, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for doc in docs:
        for piece in chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(chunk_id=idx, source=doc.source, text=piece))
            idx += 1
    return chunks
