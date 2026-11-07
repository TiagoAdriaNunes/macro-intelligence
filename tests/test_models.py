"""Free model selection, with a fake OpenRouter catalogue: no network."""

import requests

from macro_intelligence import llm as llm_module
from macro_intelligence import models


def entry(id, score=None, created=0, free=True, tools=True, output=("text",), reasoning=None):
    price = "0" if free else "0.000001"
    m = {
        "id": id,
        "created": created,
        "pricing": {"prompt": price, "completion": price},
        "architecture": {"output_modalities": list(output)},
        "supported_parameters": ["tools", "temperature"] if tools else ["temperature"],
    }
    if score is not None:
        m["benchmarks"] = {"artificial_analysis": {"intelligence_index": score}}
    if reasoning:
        m["reasoning"] = reasoning
    return m


CATALOGUE = [
    entry("paid/model", score=60, free=False),
    entry("no-tools/model:free", score=50, tools=False),
    entry("music/model:free", output=("audio",)),
    entry("smart/model:free", score=13.6),
    entry("ok/model:free", score=5.5),
    entry("new/unscored:free", created=200),
    entry("old/unscored:free", created=100),
    entry("inclusionai/ling-3.1-flash", score=40),  # low priority
    entry("nvidia/nemotron-3.5-content-safety:free", score=30),  # excluded
    entry("openrouter/free"),  # router, excluded from the ranking
    entry("blocked/model:free", score=20),  # not usable with the account's settings
    entry("down/model:free", score=25),  # uptime below 100%
]


def fake_get_json(url, **kwargs):
    if url == models.USER_MODELS_URL:
        return {"data": [{"id": m["id"]} for m in CATALOGUE if m["id"] != "blocked/model:free"]}
    if url == models.MODELS_URL:
        return {"data": CATALOGUE}
    model_id = url.removeprefix(models.MODELS_URL + "/").removesuffix("/endpoints")
    return {
        "data": {"endpoints": [{"uptime_last_30m": 80 if model_id == "down/model:free" else 100}]}
    }


def test_free_tool_models_filters_and_ranks(monkeypatch):
    monkeypatch.setattr(models, "_get_json", fake_get_json)

    ids = [m["id"] for m in models.free_tool_models("key")]

    assert ids == [
        "smart/model:free",
        "ok/model:free",
        "new/unscored:free",
        "old/unscored:free",
        "inclusionai/ling-3.1-flash",
    ]


def test_keeps_all_models_when_none_is_reliable(monkeypatch):
    monkeypatch.setattr(models, "uptime_30m", lambda model_id: None)
    candidates = [entry("a:free"), entry("b:free")]
    assert models.keep_reliable(candidates) == candidates


def test_model_chain_falls_back_to_router_when_catalogue_fails(monkeypatch):
    def offline(api_key):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm_module, "free_tool_models", offline)
    assert llm_module.model_chain("key") == [(models.FALLBACK_MODEL, None)]


def test_model_chain_takes_top_models_then_router(monkeypatch):
    ranked = [entry(f"m{i}:free") for i in range(5)]
    monkeypatch.setattr(llm_module, "free_tool_models", lambda api_key: ranked)
    ids = [m for m, _ in llm_module.model_chain("key")]
    assert ids == ["m0:free", "m1:free", "m2:free", models.FALLBACK_MODEL]


def test_reasoning_effort_uses_nearest_supported_level():
    assert models.model_reasoning(entry("a")) is None
    assert models.model_reasoning(entry("a", reasoning={"supported_efforts": ["high", "low"]})) == {
        "effort": "low"
    }
    # low isn't offered: medium is one step away, none two
    assert models.model_reasoning(
        entry("a", reasoning={"supported_efforts": ["high", "medium", "none"]})
    ) == {"effort": "medium"}
    # a model that always reasons rejects "none"
    always = {"supported_efforts": ["high", "none"], "mandatory": True}
    assert models.model_reasoning(entry("a", reasoning=always)) == {"effort": "high"}
    assert models.nearest_effort("low", ["medium", "minimal"]) == "medium"  # tie: the higher one
