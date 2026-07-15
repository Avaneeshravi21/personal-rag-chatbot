"""
Query decomposition for multi-topic questions.

The problem this solves: a question like "How does RETRO compare to REALM?"
mentions TWO distinct topics, but we only pull one top-k pool of candidates.
If chunks about REALM happen to score slightly higher across the board,
they can crowd out ALL of the RETRO chunks from the top-k entirely --
not because RETRO is irrelevant, but because it's competing for the same
limited slots as REALM instead of getting its own guaranteed slots.

The fix: detect when a question involves multiple distinct topics, split
it into separate standalone sub-questions (one per topic), retrieve top
candidates for EACH sub-question independently, then merge everything
into one pool before reranking. This guarantees every topic mentioned
gets a fair shot at being represented, instead of topics competing
against each other for the same slots.

For simple single-topic questions, this returns the original question
unchanged as a single-item list -- no wasted extra searches.
"""
import json
from llm.client import LLMClient

DECOMPOSE_INSTRUCTIONS = """You analyze questions to decide if they need to be split for a document search system.

If the question asks about or compares MULTIPLE distinct topics/entities/concepts (e.g. "How does X compare to Y?", "What are the differences between A and B?"), split it into separate standalone sub-questions, one per topic -- each fully self-contained.

If the question is about a SINGLE topic, return it unchanged as the only item.

Respond with ONLY a JSON array of strings, nothing else. No explanation, no markdown formatting.

Examples:
Question: "How does RETRO compare to REALM?"
["What is RETRO and how does it work?", "What is REALM and how does it work?"]

Question: "What is retrieval augmented generation?"
["What is retrieval augmented generation?"]

Question: "What are my technical skills and what projects have I built?"
["What are my technical skills?", "What projects have I built?"]"""


def decompose_query(llm_client: LLMClient, query: str) -> list[str]:
    response = llm_client.generate(
        system_prompt=DECOMPOSE_INSTRUCTIONS,
        messages=[{"role": "user", "content": f"Question: {query!r}"}],
        max_tokens=200,
    )
    try:
        # models sometimes wrap JSON in ```json fences despite instructions -- strip if present
        cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        sub_queries = json.loads(cleaned)
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries) and sub_queries:
            return sub_queries
    except (json.JSONDecodeError, ValueError):
        pass

    # If parsing fails for any reason, fall back to treating it as a single-topic question
    # rather than crashing the whole pipeline over a formatting hiccup.
    print("[decompose] could not parse sub-questions, falling back to original query")
    return [query]