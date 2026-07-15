"""
Qdrant vector store wrapper.

Stores chunk text + metadata as payload alongside the dense vector,
so BM25 (sparse) scoring can be done over the same payload text at
query time and fused with dense similarity (hybrid search — Week 2).
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
import uuid


class VectorStore:
    def __init__(self, url: str | None = "http://localhost:6333", api_key: str | None = None, local_path: str | None = None):
        """
        Two modes:
        - local_path set (e.g. "data/qdrant_local"): embedded mode, no Docker,
          no server, runs in-process. Best for 8GB RAM machines and for
          running this same code in Colab/Jupyter.
        - url set (default): connects to a running Qdrant server (Docker or cloud).
        """
        if local_path:
            self.client = QdrantClient(path=local_path)
        else:
            self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection(self, name: str, dim: int = 1024):
        existing = [c.name for c in self.client.get_collections().collections]
        if name in existing:
            print(f"[info] collection '{name}' already exists, skipping create")
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[created] collection '{name}' (dim={dim})")

    def upsert_chunks(self, collection: str, chunks: list, vectors):
        """
        chunks: list[Chunk] (from ingestion.chunker)
        vectors: np.ndarray of shape (len(chunks), dim)
        """
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "doc_type": chunk.doc_type,
                        "page": chunk.page,
                        "chunk_id": chunk.chunk_id,
                        **chunk.metadata,
                    },
                )
            )
        self.client.upsert(collection_name=collection, points=points)
        print(f"[upserted] {len(points)} chunks into '{collection}'")

    def search(self, collection: str, query_vector, top_k: int = 20, query_filter: Filter | None = None):
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            {
                "score": r.score,
                "text": r.payload.get("text"),
                "source": r.payload.get("source"),
                "page": r.payload.get("page"),
                "chunk_id": r.payload.get("chunk_id"),
            }
            for r in results
        ]

    def count(self, collection: str) -> int:
        info = self.client.get_collection(collection)
        return info.points_count
