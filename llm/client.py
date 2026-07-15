"""
Thin wrapper around the LLM API (Anthropic or OpenAI).

Why wrap the API instead of calling it directly everywhere: if you ever
want to switch providers, benchmark Claude vs GPT, or add retry/error
handling, you only touch this one file, not every place that generates
text. This is the same "adapter" pattern we used for document loaders
in Week 1 -- different providers in, one consistent interface out.
"""
import config


class LLMClient:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL

        if self.provider == "anthropic":
            import anthropic
            if not config.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file."
                )
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            import openai
            if not config.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        elif self.provider == "groq":
            # Groq's API is OpenAI-compatible -- same client library, just a
            # different base_url and key. Free tier, no cost, good for testing.
            import openai
            if not config.GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY is not set. Add it to your .env file. "
                    "Get a free key at https://console.groq.com/keys"
                )
            self.client = openai.OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def generate(self, system_prompt: str, messages: list[dict], max_tokens: int = 1000) -> str:
        """
        messages: list of {"role": "user"|"assistant", "content": str},
        in chronological order (the actual conversation so far).
        """
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text

        elif self.provider in ("openai", "groq"):
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=openai_messages,
            )
            return response.choices[0].message.content