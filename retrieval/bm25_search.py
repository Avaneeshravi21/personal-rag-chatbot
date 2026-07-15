"""
BM25 keyword search over the same chunks stored in Qdrant.

Why BM25 alongside dense embeddings: dense vectors are great at "meaning"
(paraphrase, synonyms) but can miss exact terms — model names, acronyms,
numbers, rare proper nouns. BM25 is the classic keyword-frequency algorithm
search engines used before embeddings existed, and still wins on exact-term
queries. Combining both (hybrid search) covers both failure modes.

We rebuild the BM25 index in-memory from Qdrant's stored payload text —
no second database needed, since the text already lives in the payload.
"""
import re
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient


def _tokenize(text: str) -> list[str]:
    # Simple, fast tokenizer: lowercase, strip punctuation, split on whitespace.
    # Good enough for BM25 — no need for a heavier NLP tokenizer here.
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Search:
    def __init__(self, client: QdrantClient, collection: str):
        self.collection = collection
        self.chunk_ids: list[str] = []
        self.payloads: list[dict] = []
        self._load_corpus(client)
        tokenized_corpus = [_tokenize(p["text"]) for p in self.payloads]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_corpus(self, client: QdrantClient, batch_size: int = 256):
        """Scrolls through every point in the collection to build the BM25 corpus."""
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                self.chunk_ids.append(str(pt.id))
                self.payloads.append(pt.payload)
            if next_offset is None:
                break
        print(f"[bm25] indexed {len(self.payloads)} chunks from '{self.collection}'")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "text": self.payloads[i].get("text"),
                "source": self.payloads[i].get("source"),
                "page": self.payloads[i].get("page"),
                "chunk_id": self.payloads[i].get("chunk_id"),
            }
            for i in ranked_idx
        ]
