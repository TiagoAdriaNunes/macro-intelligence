"""Pick free OpenRouter models, the same way shinychat-openrouter does (app/logic/models.R).

1. Download OpenRouter's public model catalogue.
2. Keep free text models that support tool calling (the planner relies on it).
3. Drop models this API key's account can't use (privacy settings, https://openrouter.ai/settings/privacy)
   and models whose provider wasn't fully up in the last 30 minutes.
4. Rank: highest Artificial Analysis intelligence index first, unscored models newest first,
   often rate-limited models last.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

MODELS_URL = "https://openrouter.ai/api/v1/models"  # public, no key needed
USER_MODELS_URL = "https://openrouter.ai/api/v1/models/user"  # needs the key
FALLBACK_MODEL = "openrouter/free"  # router that picks a free model per request; supports tools
REQUEST_TIMEOUT = 10  # seconds
# % of requests answered in the last 30 minutes; None to skip the check
MIN_UPTIME_30M: float | None = 100

# How much reasoning models think (OpenRouter `reasoning.effort`). The agent picks indicators and
# reads small tables, so low is faster; models without that level get the nearest one.
REASONING_EFFORT = "low"
EFFORT_LEVELS = ["max", "xhigh", "high", "medium", "low", "minimal", "none"]

# Ranked last whatever their score. Uptime doesn't count rate limiting (HTTP 429), so add models
# that are often rate limited here.
LOW_PRIORITY_MODELS = {
    "inclusionai/ling-3.1-flash",  # Novita's free capacity is shared by all OpenRouter users
}

# Free models that don't work as a chat model. The catalogue has no flag for this, so check new
# models with a real request.
EXCLUDED_MODELS = {
    "thinkingmachines/inkling:free",  # 403: only through apps listed on openrouter.ai/apps
    "thinkingmachines/inkling-small:free",
    "nvidia/nemotron-3.5-content-safety:free",  # safety classifier, not a chat model
    FALLBACK_MODEL,  # router; used only as the last fallback
}


def is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    return pricing.get("prompt") == "0" and pricing.get("completion") == "0"


def is_text_only(model: dict) -> bool:
    return (model.get("architecture") or {}).get("output_modalities") == ["text"]


def supports_tools(model: dict) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def intelligence(model: dict) -> float | None:
    """Artificial Analysis intelligence index; many free models have none."""
    score = ((model.get("benchmarks") or {}).get("artificial_analysis") or {}).get(
        "intelligence_index"
    )
    return score if isinstance(score, (int, float)) else None


def _get_json(url: str, **kwargs) -> dict:
    response = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def usable_ids(api_key: str) -> set[str] | None:
    """Ids of the models this account can use, or None if the list can't be fetched."""
    try:
        data = _get_json(USER_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"})["data"]
        return {m["id"] for m in data}
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log.warning("Could not fetch the models this account can use: %s", exc)
        return None


def uptime_30m(model_id: str) -> float | None:
    """Share (%) of requests the model's best provider answered in the last 30 minutes."""
    try:
        endpoints = _get_json(f"{MODELS_URL}/{model_id}/endpoints")["data"]["endpoints"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log.warning("Could not fetch the uptime of %s: %s", model_id, exc)
        return None
    values = [
        e["uptime_last_30m"]
        for e in endpoints
        if isinstance(e.get("uptime_last_30m"), (int, float))
    ]
    return max(values) if values else None


def keep_reliable(models: list[dict]) -> list[dict]:
    """Models with full uptime in the last 30 minutes; all of them if none passes."""
    if MIN_UPTIME_30M is None or not models:
        return models
    ids = [m["id"] for m in models]
    with ThreadPoolExecutor(max_workers=16) as pool:  # one request per model
        uptime = dict(zip(ids, pool.map(uptime_30m, ids), strict=True))
    reliable = [m for m in models if (uptime[m["id"]] or 0) >= MIN_UPTIME_30M]
    if not reliable:
        log.warning(
            "No free model has %s%% uptime in the last 30 minutes; keeping all", MIN_UPTIME_30M
        )
        return models
    dropped = sorted(set(ids) - {m["id"] for m in reliable})
    if dropped:
        log.info(
            "Left out for uptime below %s%% (or unknown): %s", MIN_UPTIME_30M, ", ".join(dropped)
        )
    return reliable


def _rank_key(model: dict) -> tuple:
    score = intelligence(model)
    return (
        model["id"] in LOW_PRIORITY_MODELS,
        score is None,
        -(score or 0),
        -(model.get("created") or 0),
    )


def free_tool_models(api_key: str | None = None) -> list[dict]:
    """Free, usable, reliable OpenRouter models with tool calling, best first."""
    catalogue = _get_json(MODELS_URL)["data"]
    models = [
        m
        for m in catalogue
        if is_free(m) and is_text_only(m) and supports_tools(m) and m["id"] not in EXCLUDED_MODELS
    ]
    usable = usable_ids(api_key) if api_key else None
    if usable is not None:
        dropped = sorted(m["id"] for m in models if m["id"] not in usable)
        if dropped:
            log.info("Left out as unusable with this account's settings: %s", ", ".join(dropped))
        models = [m for m in models if m["id"] in usable]
    return sorted(keep_reliable(models), key=_rank_key)


def nearest_effort(effort: str, efforts: list[str]) -> str:
    """The level among `efforts` closest to `effort`, as OpenRouter maps it (higher one on a tie)."""
    target = EFFORT_LEVELS.index(effort)
    return min(
        efforts, key=lambda e: (abs(EFFORT_LEVELS.index(e) - target), EFFORT_LEVELS.index(e))
    )


def model_reasoning(model: dict, effort: str | None = REASONING_EFFORT) -> dict | None:
    """OpenRouter `reasoning` setting for a model, or None to use its default."""
    reasoning = model.get("reasoning") or {}
    efforts = [e for e in reasoning.get("supported_efforts") or [] if e in EFFORT_LEVELS]
    if reasoning.get("mandatory"):  # always reasons, so it rejects "none"
        efforts = [e for e in efforts if e != "none"]
    if not effort or not efforts:
        return None
    return {"effort": nearest_effort(effort, efforts)}
