"""OpenRouter chat model setup.

By default the agent uses free models (see models.py). A command line tool has no model
dropdown, so the best few are chained as fallbacks: when one fails (rate limit, outage) or
replies badly, the next one takes over, ending with OpenRouter's free router.
"""

import logging
import os

import requests
from dotenv import load_dotenv

from .models import FALLBACK_MODEL, free_tool_models, model_reasoning

log = logging.getLogger(__name__)

MAX_MODELS = 3  # free models tried in order before the fallback router


def _chat(model: str, reasoning: dict | None = None):
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=model,
        temperature=0,
        reasoning=reasoning,
        max_retries=1,  # move on to the next model rather than retrying a rate-limited one
        app_title="macro-intelligence",
    )


def model_chain(api_key: str) -> list[tuple[str, dict | None]]:
    """Ids of the models to try, in order, with the reasoning setting for each."""
    try:
        ranked = free_tool_models(api_key)
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log.warning("Could not fetch OpenRouter models (%s); using %s", exc, FALLBACK_MODEL)
        ranked = []
    return [(m["id"], model_reasoning(m)) for m in ranked[:MAX_MODELS]] + [(FALLBACK_MODEL, None)]


def make_llm(model: str | None = None):
    """Chat model for the agent: `model` (or OPENROUTER_MODEL) if given, else free models with fallbacks."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    model = model or os.getenv("OPENROUTER_MODEL")
    if model:
        return _chat(model)

    chain = model_chain(api_key)
    log.info("Models, in order: %s", " -> ".join(m for m, _ in chain))
    first, *rest = [_chat(m, reasoning) for m, reasoning in chain]
    return first.with_fallbacks(rest) if rest else first
