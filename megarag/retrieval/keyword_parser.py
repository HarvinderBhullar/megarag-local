"""Parse a user question into low-level and high-level keywords using Ollama."""
import json
import logging
from openai import OpenAI
from config.settings import get_settings

logger = logging.getLogger(__name__)

_SYSTEM = """You are a query analysis assistant.
Given a user question, extract search keywords in two categories.

Respond ONLY with a JSON object — no markdown fences:
{
  "low_level": ["specific entity names, numbers, dates, proper nouns"],
  "high_level": ["abstract concepts, topics, themes"]
}"""


def parse_keywords(question: str) -> dict[str, list[str]]:
    cfg = get_settings()
    client = OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")

    try:
        response = client.chat.completions.create(
            model=cfg.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("[query] keyword parsing failed: %s", exc)
        # Fallback: treat the whole question as low-level keywords
        return {"low_level": question.split(), "high_level": []}
