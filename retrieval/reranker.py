"""
Cross-encoder reranker.

Why this exists as a separate stage from hybrid search: dense/BM25 retrieval
scores each chunk independently against the query (fast, but approximate).
A cross-encoder instead looks at the query and one candidate chunk *together*
in a single forward pass, so it can judge relevance far more precisely — the
tradeoff is it's too slow to run over your whole corpus, so the standard
pattern is: retrieve a wider candidate set cheaply (hybrid search, top ~20),
then rerank just those with the expensive-but-accurate cross-encoder, then
keep only the true top few (top ~5) for the LLM's context window.
"""
from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = False):
        # use_fp16=False by default: fp16 needs a compatible GPU, and this
        # project defaults to CPU-friendly settings for modest hardware.
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.compute_score(pairs)
        if isinstance(scores, float):  # single-pair edge case
            scores = [scores]

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
