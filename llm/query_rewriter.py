"""
Query rewriting for follow-up questions.

Why this is needed: imagine the conversation goes:
  User: "What is RETRO?"
  Assistant: "RETRO is a retrieval-augmented language model that..."
  User: "How does it compare to REALM?"

If we embed "How does it compare to REALM?" directly and search for it,
the word "it" carries no meaning on its own -- retrieval would likely
fail to find RETRO-related chunks, because the query vector has no idea
"it" refers to RETRO. We fix this by asking the LLM to rewrite the query
into a standalone version first: "How does RETRO compare to REALM?" --
THEN we embed and search using that rewritten version.

This is a cheap, fast LLM call (short output) that runs before retrieval,
not part of the final answer generation.
"""
from llm.client import LLMClient

REWRITE_INSTRUCTIONS = """Given the recent conversation history and a new user question, rewrite the question to be fully standalone -- resolving any pronouns or implicit references (like "it", "that", "the second one") using the conversation context.

If the question is already standalone (doesn't depend on prior context), return it unchanged.
Return ONLY the rewritten question, nothing else -- no preamble, no explanation."""


def rewrite_query(llm_client: LLMClient, query: str, recent_turns: list[dict]) -> str:
    if not recent_turns:
        return query  # nothing to resolve against on the first turn

    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent_turns
    )
    user_message = f"Conversation history:\n{history_text}\n\nNew question: {query}\n\nStandalone rewrite:"

    rewritten = llm_client.generate(
        system_prompt=REWRITE_INSTRUCTIONS,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=150,
    )
    return rewritten.strip()
