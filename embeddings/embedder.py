"""
Embedding wrapper around BAAI/bge-large-en-v1.5.

Note: bge models expect a specific instruction prefix for QUERIES
(not for documents) to get best retrieval performance — this is a
detail a lot of tutorial RAG projects miss, and worth mentioning
if asked about retrieval quality tuning.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"):
        # device="cpu" forced by default: a 2GB GPU is too small for bge-large
        # and will OOM. CPU is slower but reliable on modest hardware.
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed chunk texts (no instruction prefix needed for bge docs)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query — uses the bge instruction prefix."""
        return self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
        )
