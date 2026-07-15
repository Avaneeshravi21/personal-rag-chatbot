"""
Chunking strategies.

For your eval report, run the same doc set through both strategies
and measure recall@k for each — that comparison is a great thing
to show in an interview.
"""
import re
import tiktoken
from dataclasses import dataclass, field
from ingestion.loaders import Document

_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    page: int | None
    chunk_id: str
    metadata: dict = field(default_factory=dict)


def _token_len(text: str) -> int:
    return len(_encoder.encode(text))


def recursive_chunk(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[str]:
    """
    Splits on progressively finer separators (paragraph -> sentence -> word)
    so we don't cut a sentence in half unless we absolutely have to.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split_recursive(text: str, seps: list[str]) -> list[str]:
        if _token_len(text) <= chunk_size:
            return [text]
        if not seps:
            # last resort: hard token slice
            tokens = _encoder.encode(text)
            return [
                _encoder.decode(tokens[i : i + chunk_size])
                for i in range(0, len(tokens), chunk_size - overlap)
            ]

        sep, rest_seps = seps[0], seps[1:]
        parts = [p for p in text.split(sep) if p.strip()]
        chunks, current = [], ""

        for part in parts:
            candidate = current + sep + part if current else part
            if _token_len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if _token_len(part) > chunk_size:
                    chunks.extend(split_recursive(part, rest_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = split_recursive(text, separators)

    # add overlap between consecutive chunks (helps preserve context at boundaries)
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_tail = _encoder.decode(_encoder.encode(raw_chunks[i - 1])[-overlap:])
        overlapped.append(prev_tail + " " + chunk)

    return overlapped


def semantic_chunk(text: str, similarity_threshold: float = 0.75, max_tokens: int = 500):
    """
    Groups consecutive sentences into a chunk while they stay semantically
    similar (via sentence embeddings), splitting only when topic drift is
    detected. Requires sentence-transformers — heavier than recursive_chunk
    but produces more coherent chunks. Good for the "domain_corpus" papers.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")  # small/fast, just for chunking decisions
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return []

    embeddings = model.encode(sentences)
    chunks, current, current_tokens = [], [sentences[0]], _token_len(sentences[0])

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i - 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-8
        )
        sent_tokens = _token_len(sentences[i])
        if sim >= similarity_threshold and current_tokens + sent_tokens <= max_tokens:
            current.append(sentences[i])
            current_tokens += sent_tokens
        else:
            chunks.append(" ".join(current))
            current, current_tokens = [sentences[i]], sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, strategy: str = "recursive", chunk_size: int = 400, overlap: int = 60) -> list[Chunk]:
    if strategy == "recursive":
        raw_chunks = recursive_chunk(doc.text, chunk_size, overlap)
    elif strategy == "semantic":
        raw_chunks = semantic_chunk(doc.text, max_tokens=chunk_size)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunk_id = f"{doc.source}::p{doc.page or 0}::c{i}"
        chunks.append(
            Chunk(
                text=text,
                source=doc.source,
                doc_type=doc.doc_type,
                page=doc.page,
                chunk_id=chunk_id,
                metadata={**doc.metadata, "strategy": strategy},
            )
        )
    return chunks
