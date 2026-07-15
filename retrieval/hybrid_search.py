"""
Hybrid search = dense vector search + BM25 keyword search, combined.

Fusion method: Reciprocal Rank Fusion (RRF), not a weighted score average.
Why RRF instead of just weighting the two raw scores together: dense cosine
similarity (0-1 range) and BM25 scores (unbounded, corpus-dependent) live on
totally different scales, so averaging them directly is comparing apples to
oranges unless you carefully re-normalize both every time. RRF sidesteps this
by only looking at *rank position* in each list, not the raw score value —
it's simpler, more robust, and is what most production hybrid search systems
(Elasticsearch, Azure AI Search, Weaviate) use by default.

RRF formula: score(doc) = sum over each ranker of  1 / (k + rank_in_that_list)
k=60 is the standard constant from the original RRF paper.
"""
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore
from retrieval.bm25_search import BM25Search


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, key: str = "chunk_id") -> list[dict]:
    """
    result_lists: list of ranked result lists (e.g. [dense_results, bm25_results]),
    each already sorted best-first. Returns a single fused, re-ranked list.
    """
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc[key]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[doc_id] = doc  # keep one copy of the doc's fields

    ranked_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)
    fused = []
    for doc_id in ranked_ids:
        doc = dict(doc_lookup[doc_id])
        doc["rrf_score"] = fused_scores[doc_id]
        fused.append(doc)
    return fused


class HybridSearcher:
    def __init__(self, collection: str, embedder: Embedder, vector_store: VectorStore):
        self.collection = collection
        self.embedder = embedder
        self.vector_store = vector_store
        # BM25 index is built once at init from what's currently in Qdrant.
        # Rebuild (re-instantiate) this class if you ingest new data afterward.
        self.bm25 = BM25Search(vector_store.client, collection)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(self.collection, query_vector, top_k=top_k)
        bm25_results = self.bm25.search(query, top_k=top_k)
        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        return fused[:top_k]
