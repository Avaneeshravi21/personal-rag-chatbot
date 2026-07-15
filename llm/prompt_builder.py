"""
Builds the system prompt that tells the LLM how to use retrieved chunks.

Why this needs care instead of just dumping chunks into the prompt:
- The model needs explicit instructions to ONLY use the provided context
  (otherwise it may answer from its own general training knowledge,
  which defeats the point of RAG and can produce wrong info about
  YOUR specific documents).
- It needs to know how to cite sources, so answers are traceable back
  to a specific document/page.
- It needs an explicit "I don't know" escape hatch -- without this,
  LLMs tend to confidently make something up (hallucinate) rather than
  admit the retrieved context doesn't answer the question.
"""


def build_system_prompt(retrieved_chunks: list[dict], memory_summary: str = None) -> str:
    if not retrieved_chunks:
        context_block = "(No relevant documents were found for this query.)"
    else:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            page_info = f", page {chunk['page']}" if chunk.get("page") else ""
            parts.append(
                f"[Source {i}: {chunk['source']}{page_info}]\n{chunk['text']}"
            )
        context_block = "\n\n".join(parts)

    memory_block = ""
    if memory_summary:
        memory_block = f"\n\nSummary of earlier conversation (for context):\n{memory_summary}\n"

    system_prompt = f"""You are a helpful assistant answering questions using ONLY the retrieved context below, plus the conversation history provided.

RULES:
1. Base your answer strictly on the retrieved context. Do not use outside knowledge not present in the context.
2. If the context does not contain enough information to answer, say so clearly instead of guessing.
3. When you state a fact from the context, cite it inline like [Source 1] or [Source 2], matching the source numbers below.
4. Be concise and direct. Do not repeat the question back before answering.
5. If multiple sources agree, you can cite them together, e.g. [Source 1, Source 3].
{memory_block}
RETRIEVED CONTEXT:
{context_block}
"""
    return system_prompt
