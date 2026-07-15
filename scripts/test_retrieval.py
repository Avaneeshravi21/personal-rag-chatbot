"""
Quick sanity check after running ingest.py:
ask a question, see what chunks come back, eyeball if they're relevant.

Usage:
    python scripts/test_retrieval.py --collection domain_corpus --query "What is retrieval augmented generation?"
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore


def run(collection: str, query: str, top_k: int = 5):
    embedder = Embedder(config.EMBEDDING_MODEL)
    if config.USE_LOCAL_QDRANT:
        store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
    else:
        store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    query_vector = embedder.embed_query(query)
    results = store.search(collection, query_vector, top_k=top_k)

    print(f"\nQuery: {query!r}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  source={r['source']} (page {r['page']})")
        print(f"    {r['text'][:200].strip()!r}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    run(args.collection, args.query, args.top_k)
