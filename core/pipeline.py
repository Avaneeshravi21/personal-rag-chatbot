"""
The core RAG pipeline, extracted into a reusable class.

Why pull this out of scripts/chat.py into its own module: chat.py was
built for one person typing in a terminal, using one global memory
object for the whole session. A real API needs to serve MULTIPLE users
(or at least multiple browser tabs/sessions) at once, each with their
OWN separate conversation memory -- so the retrieval/generation logic
needs to be reusable and NOT tied to a single global memory instance.

This class is instantiated ONCE (loading all the heavy models/indexes
just once, at startup), then reused across every request. Memory is
kept separately per session_id, not inside this class itself.
"""
import config
from embeddings.embedder import Embedder
from vectorstore.qdrant_store import VectorStore
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker
from llm.client import LLMClient
from llm.prompt_builder import build_system_prompt
from llm.query_rewriter import rewrite_query
from llm.query_decomposer import decompose_query
from memory.conversation_memory import ConversationMemory


class RAGPipeline:
    def __init__(self):
        print("[pipeline] loading embedding model, BM25 indexes, and reranker...")
        self.embedder = Embedder(config.EMBEDDING_MODEL)

        if config.USE_LOCAL_QDRANT:
            self.store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
        else:
            self.store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        self.searchers = {
            "personal_docs": HybridSearcher("personal_docs", self.embedder, self.store),
            "domain_corpus": HybridSearcher("domain_corpus", self.embedder, self.store),
        }
        self.reranker = Reranker(config.RERANKER_MODEL)
        self.llm_client = LLMClient()

        # One ConversationMemory per session_id, so different users/tabs
        # don't share or overwrite each other's conversation history.
        self.sessions: dict[str, ConversationMemory] = {}
        print("[pipeline] ready.")

    def _get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(self.llm_client)
        return self.sessions[session_id]

    def ask(self, session_id: str, user_query: str) -> dict:
        """
        Runs one full turn: rewrite -> decompose -> retrieve -> rerank ->
        generate -> update memory. Returns the answer plus the sources used,
        so the frontend can display citations.
        """
        memory = self._get_memory(session_id)
        recent_turns = memory.get_recent_turns()

        standalone_query = rewrite_query(self.llm_client, user_query, recent_turns)

        sub_queries = decompose_query(self.llm_client, standalone_query)

        per_topic_k = max(2, config.FINAL_TOP_K // len(sub_queries))
        top_chunks = []
        seen_chunk_ids = set()

        for sub_query in sub_queries:
            sub_candidates = []
            for searcher in self.searchers.values():
                sub_candidates.extend(searcher.search(sub_query, top_k=config.RERANK_TOP_K))

            sub_top = self.reranker.rerank(sub_query, sub_candidates, top_k=per_topic_k)
            for chunk in sub_top:
                if chunk["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk["chunk_id"])
                    top_chunks.append(chunk)

        system_prompt = build_system_prompt(top_chunks, memory_summary=memory.get_summary())
        messages = recent_turns + [{"role": "user", "content": user_query}]
        answer = self.llm_client.generate(system_prompt, messages, max_tokens=800)

        memory.add_turn("user", user_query)
        memory.add_turn("assistant", answer)

        sources = sorted(set(
            f"{c['source']} (page {c['page']})" if c.get("page") else c["source"]
            for c in top_chunks
        ))

        return {
            "answer": answer,
            "sources": sources,
            "standalone_query": standalone_query,
            "sub_queries": sub_queries,
        }
