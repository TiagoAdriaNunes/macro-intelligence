"""Chat streaming (stream.answer_stream) with a fake LLM: no network."""

import asyncio

from test_graph import FakeLLM, fake_dataset

from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.stream import FAILED, answer_stream


def collect(graph, question, history=None) -> str:
    async def run():
        return "".join([chunk async for chunk in answer_stream(graph, question, history)])

    return asyncio.run(run())


def test_stream_shows_lookup_answer_sources_and_model():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m", "wb_gdp_growth"], countries=["BRA", "USA"]))
    graph = build_graph(llm, fetch=lambda ind, countries, start: [fake_dataset(ind)])

    text = collect(graph, "Inflation and growth?")

    assert text.startswith("_Looking up: IPCA inflation (12-month), GDP growth for BRA, USA._")
    assert text.count(llm.answer) == 1  # streamed or not, the answer appears once
    assert "**Sources**" in text and "series 13522" in text
    assert "Model: `fake/model`" in text


def test_stream_passes_history_to_the_models():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m"]))
    graph = build_graph(llm, fetch=lambda ind, countries, start: [fake_dataset(ind)])
    history = [
        {"role": "user", "content": "What is inflation in Brazil?"},
        {"role": "assistant", "content": "It was 4.58%."},
    ]

    collect(graph, "And a year ago?", history)

    contents = [m.content for m in llm.prompts[0]]
    assert contents[1:3] == ["What is inflation in Brazil?", "It was 4.58%."]


def test_stream_unmatched_question_has_no_lookup_line():
    graph = build_graph(FakeLLM(plan=Plan(indicators=[])), fetch=lambda *a: [])
    text = collect(graph, "Best pizza?")
    assert text.startswith("None of the indicators")
    assert "Looking up" not in text


def test_stream_reports_failure_instead_of_raising():
    graph = build_graph(FakeLLM(error="503 Service Unavailable"), fetch=lambda *a: [])
    assert collect(graph, "Selic?") == FAILED
