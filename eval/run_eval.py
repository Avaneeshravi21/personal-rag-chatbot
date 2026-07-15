"""
Runs the eval set against three retrieval configurations and reports
recall@k and MRR (Mean Reciprocal Rank) for each, so you can see the
actual, measurable improvement from hybrid search and reranking.

recall@k = fraction of questions where the expected source appears
           anywhere in the top-k results
MRR      = average of 1/rank_of_first_correct_hit across all questions
           (rewards getting the right answer near rank 1, not just
           somewhere in the top-k)

Usage:
    python eval/run_eval.py --collection domain_corpus
"""
import argparse
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker


def recall_and_mrr(results_per_question: list[list[dict]], expected_sources: list[str], k: int, questions: list[str] = None, verbose: bool = False):
    hits = 0
    reciprocal_ranks = []
    failures = []
    for i, (results, expected) in enumerate(zip(results_per_question, expected_sources)):
        top_k = results[:k]
        found_rank = None
        for rank, r in enumerate(top_k, 1):
            if r["source"] == expected:
                found_rank = rank
                break
        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            if questions:
                got_sources = [r["source"] for r in top_k]
                failures.append((questions[i], expected, got_sources))

    recall = hits / len(expected_sources)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    if verbose and failures:
        print("    Missed questions:")
        for q, expected, got in failures:
            print(f"      Q: {q[:70]}")
            print(f"         expected: {expected}  |  got top-{k}: {got}")

    return recall, mrr


def run(collection: str, k: int = 5):
    with open("eval/eval_set.json") as f:
        eval_data = json.load(f)
    questions = eval_data["questions"]
    print(f"Loaded {len(questions)} eval questions.\n")

    embedder = Embedder(config.EMBEDDING_MODEL)
    if config.USE_LOCAL_QDRANT:
        store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
    else:
        store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    searcher = HybridSearcher(collection, embedder, store)
    reranker = Reranker(config.RERANKER_MODEL)

    dense_results, hybrid_results, reranked_results = [], [], []

    for q in questions:
        query = q["question"]

        # 1. Dense-only (vector search alone, no BM25, no rerank)
        query_vector = embedder.embed_query(query)
        dense = store.search(collection, query_vector, top_k=config.RERANK_TOP_K)
        dense_results.append(dense)

        # 2. Hybrid (dense + BM25 via RRF, no rerank)
        hybrid = searcher.search(query, top_k=config.RERANK_TOP_K)
        hybrid_results.append(hybrid)

        # 3. Hybrid + reranked
        reranked = reranker.rerank(query, hybrid, top_k=config.RERANK_TOP_K)
        reranked_results.append(reranked)

        print(f"  processed: {query[:60]}...")

    expected_sources = [q["expected_source"] for q in questions]

    print(f"\n=== Results @ k={k} ===\n")
    question_texts = [q["question"] for q in questions]
    for name, results in [
        ("Dense-only", dense_results),
        ("Hybrid (dense + BM25)", hybrid_results),
        ("Hybrid + Reranked", reranked_results),
    ]:
        recall, mrr = recall_and_mrr(results, expected_sources, k, questions=question_texts, verbose=True)
        print(f"{name:28s}  recall@{k}={recall:.2%}   MRR={mrr:.3f}")

    print(
        "\nTip: run with --k 1 and --k 3 too, to see how much reranking "
        "specifically helps get the right answer into the very top spot "
        "(that's what MRR and low-k recall capture that recall@5 doesn't)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run(args.collection, args.k)
