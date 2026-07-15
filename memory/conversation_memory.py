"""
Conversation memory: two layers, working together.

1. SHORT-TERM BUFFER: the last N raw turns, kept word-for-word. Cheap,
   fast, and exact -- good for immediate follow-ups ("what about the
   second one?").

2. LONG-TERM (SUMMARIZED) MEMORY: once the buffer gets too long, older
   turns get compressed into a short summary by the LLM, and that summary
   is what gets kept -- not the full raw text. Why not just keep
   everything forever? Because every turn we keep verbatim adds to every
   future prompt's length, costing more tokens (money) and eventually
   exceeding the model's context limit. Summarizing trades a little
   detail for a lot of efficiency -- this is the same buffer+summary
   memory pattern used in real conversational AI products.

Why not skip memory entirely and just re-search documents every turn?
Because conversation context isn't in your documents -- "what did I ask
you two questions ago" or "compare those two things I just mentioned"
requires remembering the CONVERSATION itself, not just retrieving from
your PDFs.
"""
import config
from llm.client import LLMClient

SUMMARIZE_INSTRUCTIONS = """Summarize the following conversation turns concisely, preserving key facts, names, and topics discussed, in 2-4 sentences. This summary will be used as background context for continuing the conversation, so keep it factual and dense, not conversational."""


class ConversationMemory:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.buffer: list[dict] = []          # recent raw turns: {"role": ..., "content": ...}
        self.summary: str = ""                  # running summary of everything older than the buffer

    def add_turn(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > config.MEMORY_SUMMARIZE_AFTER:
            self._summarize_oldest()

    def _summarize_oldest(self):
        """Compresses the oldest turns beyond MEMORY_BUFFER_TURNS into the running summary."""
        to_summarize = self.buffer[: -config.MEMORY_BUFFER_TURNS]
        self.buffer = self.buffer[-config.MEMORY_BUFFER_TURNS :]

        if not to_summarize:
            return

        turns_text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in to_summarize)
        prior_summary_text = f"Earlier summary: {self.summary}\n\n" if self.summary else ""
        user_message = f"{prior_summary_text}New turns to fold in:\n{turns_text}"

        new_summary = self.llm_client.generate(
            system_prompt=SUMMARIZE_INSTRUCTIONS,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=250,
        )
        self.summary = new_summary.strip()
        print(f"[memory] summarized {len(to_summarize)} older turns into running summary")

    def get_recent_turns(self) -> list[dict]:
        """Raw buffer turns, for the LLM's `messages` list (exact wording preserved)."""
        return self.buffer

    def get_summary(self) -> str:
        """Compressed memory of everything older than the current buffer."""
        return self.summary
