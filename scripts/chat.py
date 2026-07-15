"""
The full assembled chatbot: query rewriting -> multi-source hybrid search
-> reranking -> LLM answer generation -> memory update. This is what
Week 1 (retrieval) and Week 2 (hybrid + rerank) were building toward.

Usage:
    python scripts/chat.py
    (then type questions, Ctrl+C or "exit" to quit)

Why search BOTH collections (personal_docs and domain_corpus) on every
question instead of picking one: we don't know in advance whether a
question is about your resume or about the papers, and asking the user
to pick a mode every time would be annoying. Searching both and letting
the reranker sort ALL candidates together by true relevance means the
right source naturally rises to the top regardless of which collection
it came from -- this is the "source routing" idea from the original plan,
done the simple way (search-then-rerank) rather than a separate
classification step.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    print("Setting up... (this loads the embedding model, BM25 indexes, and reranker)\n")

    embedder = Embedder(config.EMBEDDING_MODEL)
    if config.USE_LOCAL_QDRANT:
        store = VectorStore(local_path=config.QDRANT_LOCAL_PATH)
    else:
        store = VectorStore(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    # One HybridSearcher per collection we want to search across.
    searchers = {
        "personal_docs": HybridSearcher("personal_docs", embedder, store),
        "domain_corpus": HybridSearcher("domain_corpus", embedder, store),
    }
    reranker = Reranker(config.RERANKER_MODEL)
    llm_client = LLMClient()
    memory = ConversationMemory(llm_client)

    print("\nReady. Ask a question (type 'exit' to quit).\n")

    while True:
        try:
            user_query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1. Resolve follow-up references ("it", "that", etc.) using recent history
        recent_turns = memory.get_recent_turns()
        standalone_query = rewrite_query(llm_client, user_query, recent_turns)
        if standalone_query != user_query:
            print(f"  [rewritten query: {standalone_query}]")

        # 2. Split multi-topic questions into separate sub-questions, so each
        #    topic gets its own guaranteed retrieval AND reranking slots,
        #    instead of competing with other topics at either stage.
        sub_queries = decompose_query(llm_client, standalone_query)
        if len(sub_queries) > 1:
            print(f"  [decomposed into: {sub_queries}]")

        # 3 & 4. For EACH sub-question: search all collections, then rerank
        # against THAT sub-question specifically (not the combined question).
        # Reranking against the combined question was the actual bottleneck --
        # a single-topic chunk scores as "less relevant" to a two-topic
        # question even when it's exactly what's needed, so topics were
        # still crowding each other out at the reranking stage even after
        # decomposition fixed the retrieval stage. Giving each sub-topic its
        # own dedicated rerank pass guarantees it a fair slot in the final
        # answer, not just fair candidacy.
        per_topic_k = max(2, config.FINAL_TOP_K // len(sub_queries))
        top_chunks = []
        seen_chunk_ids = set()

        for sub_query in sub_queries:
            sub_candidates = []
            for name, searcher in searchers.items():
                sub_candidates.extend(searcher.search(sub_query, top_k=config.RERANK_TOP_K))

            sub_top = reranker.rerank(sub_query, sub_candidates, top_k=per_topic_k)
            for chunk in sub_top:
                if chunk["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk["chunk_id"])
                    top_chunks.append(chunk)

        # 5. Build the prompt and generate an answer
        system_prompt = build_system_prompt(top_chunks, memory_summary=memory.get_summary())
        messages = recent_turns + [{"role": "user", "content": user_query}]
        answer = llm_client.generate(system_prompt, messages, max_tokens=800)

        print(f"\nAssistant: {answer}\n")
        if top_chunks:
            sources = sorted(set(f"{c['source']} (page {c['page']})" if c.get("page") else c["source"] for c in top_chunks))
            print(f"  [sources used: {', '.join(sources)}]\n")

        # 6. Update memory with this turn
        memory.add_turn("user", user_query)
        memory.add_turn("assistant", answer)


if __name__ == "__main__":
    main()