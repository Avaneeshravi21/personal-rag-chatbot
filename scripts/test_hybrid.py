"""
Test the full Week 2 pipeline: hybrid search (dense + BM25 via RRF)
followed by cross-encoder reranking.

Usage:
    python scripts/test_hybrid.py --collection domain_corpus --query "What is retrieval augmented generation?"
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker


def run(collection: str, query: str):
    embedder = Embedder(config.EMBEDDING_MODEL)
    if config.USE_LOCAL_QDRANT:
        store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
    else:
        store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    print(f"\nBuilding hybrid searcher for '{collection}' (this loads the BM25 index once)...")
    searcher = HybridSearcher(collection, embedder, store)

    print(f"\n--- Hybrid search results (before reranking) ---")
    hybrid_results = searcher.search(query, top_k=config.RERANK_TOP_K)
    for i, r in enumerate(hybrid_results[:10], 1):
        print(f"[{i}] rrf={r['rrf_score']:.4f}  source={r['source']} (page {r['page']})")
        print(f"    {r['text'][:150].strip()!r}\n")

    print(f"\n--- After cross-encoder reranking (top {config.FINAL_TOP_K}) ---")
    reranker = Reranker(config.RERANKER_MODEL)
    final_results = reranker.rerank(query, hybrid_results, top_k=config.FINAL_TOP_K)
    for i, r in enumerate(final_results, 1):
        print(f"[{i}] rerank_score={r['rerank_score']:.4f}  source={r['source']} (page {r['page']})")
        print(f"    {r['text'][:200].strip()!r}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    run(args.collection, args.query)
