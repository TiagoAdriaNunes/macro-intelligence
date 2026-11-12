"""Evaluators and the experiment's questions: no network, no Phoenix server."""

import re
from datetime import date

import pytest
from langchain_core.runnables import RunnableLambda
from test_graph import FakeLLM, fake_dataset

from macro_intelligence.evaluators import (
    Verdict,
    expected_figures,
    grounding,
    make_answer_quality,
    plan_match,
)
from macro_intelligence.experiments import cached, dataset_name, load_questions, make_task
from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.indicators import CATALOG


def plan(**fields) -> dict:
    return Plan(**fields).model_dump()


def test_plan_match_ignores_order_and_fields_the_example_leaves_out():
    output = {"plan": plan(indicators=["wb_inflation", "wb_gdp_growth"], countries=["USA", "BRA"])}
    expected = {"indicators": ["wb_gdp_growth", "wb_inflation"], "countries": ["BRA", "USA"]}
    assert plan_match(output, expected)["score"] == 1


def test_plan_match_lists_each_difference():
    output = {"plan": plan(indicators=["br_ipca_12m"], countries=["ARG"], yearly_table=True)}
    expected = {"indicators": ["wb_inflation"], "countries": ["BRA"], "yearly_table": False}
    result = plan_match(output, expected)
    assert result["score"] == 0
    assert "indicators" in result["explanation"]
    assert "countries" in result["explanation"]
    assert "yearly_table" in result["explanation"]


def test_plan_match_needs_a_start_year_that_covers_the_question():
    expected = {"indicators": ["br_selic_target"], "start_year": 2023}
    for start_year, score in [(2022, 1), (2023, 1), (2025, 0)]:
        output = {"plan": plan(indicators=["br_selic_target"], start_year=start_year)}
        assert plan_match(output, expected)["score"] == score
    output = {"plan": plan(indicators=["br_selic_target"], start_year=2025)}
    assert "needs 2023 or earlier" in plan_match(output, expected)["explanation"]


def test_cache_fetches_once_and_keeps_no_failures(tmp_path):
    calls = []

    def flaky_fetch(ind, countries, start):
        calls.append(start)
        if len(calls) == 1:
            raise TimeoutError("read timed out")
        return [fake_dataset(ind, country=c) for c in countries]

    fetch = cached(flaky_fetch, tmp_path)
    ind, start = CATALOG["wb_inflation"], date(2023, 1, 1)
    with pytest.raises(TimeoutError):
        fetch(ind, ["JPN", "BRA"], start)
    assert list(tmp_path.iterdir()) == []  # the failure isn't cached
    first = fetch(ind, ["JPN", "BRA"], start)
    again = fetch(ind, ["BRA", "JPN"], start)  # same data in another order: from the cache
    assert len(calls) == 2
    assert again == first
    assert [p.name for p in tmp_path.iterdir()] == ["wb_inflation_BRA-JPN_2023-01-01.json"]


def test_plan_match_without_a_plan_scores_zero():
    assert plan_match({"plan": None}, {"indicators": []})["score"] == 0


@pytest.mark.parametrize(
    "answer",
    [
        "The dollar closed 2024 at R$ 6.1923.",
        "The dollar closed 2024 at 6.19 BRL.",  # rounded
        "O dólar fechou 2024 em R$ 6,19.",  # decimal comma
    ],
)
def test_expected_figures_match_at_the_precision_written(answer):
    assert expected_figures({"answer": answer}, {"figures": [6.1923]})["score"] == 1


def test_expected_figures_ignore_sign_and_list_what_is_missing():
    output = {"answer": "Argentina's GDP contracted 1.86% in 2023; Brazil grew 3.2%."}
    result = expected_figures(output, {"figures": [-1.86, 3.24, 6.02]})
    assert result["score"] == pytest.approx(2 / 3)
    assert result["explanation"] == "Not in the answer: 6.02"


def test_expected_figures_reject_a_figure_from_another_date():
    output = {"answer": "The Selic was 10.50% at the end of 2023."}  # June 2024's value
    assert expected_figures(output, {"figures": [11.75]})["score"] == 0


def test_expected_figures_have_no_score_when_the_example_lists_none():
    assert "score" not in expected_figures({"answer": "4.5%"}, {"indicators": []})


def test_grounding_scores_the_share_of_figures_found():
    check = {"figures": 4, "calculated": 1, "unverified": ["7.1%"]}
    result = grounding({"check": check})
    assert result["score"] == 0.75
    assert "7.1%" in result["explanation"]
    assert grounding({"check": {"figures": 2, "calculated": 0, "unverified": []}})["score"] == 1


def test_grounding_has_no_score_without_figures():
    assert "score" not in grounding({"check": None})
    assert "score" not in grounding({"check": {"figures": 0, "calculated": 0, "unverified": []}})


class JudgeLLM(FakeLLM):
    """Returns a fixed verdict from structured output."""

    verdict: Verdict | None = None

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: self.verdict)


def test_answer_quality_scores_the_criteria_met():
    verdict = Verdict(
        answers_question=True,
        units_and_dates=True,
        names_sources=False,
        correct_terms=True,
        concise=True,
        explanation="No series code.",
    )
    judge = make_answer_quality(JudgeLLM(verdict=verdict))
    output = {"answer": "Inflation was 4.58%.", "summaries": [{"name": "IPCA"}]}
    result = judge({"question": "Inflation?"}, output)
    assert result["score"] == pytest.approx(0.8)
    assert result["label"] == "fail: names_sources"
    assert result["explanation"] == "No series code."


def test_answer_quality_skips_answers_without_data():
    judge = make_answer_quality(JudgeLLM())  # would fail if called: no verdict
    assert "score" not in judge({"question": "What can I ask?"}, {"answer": "...", "summaries": []})


def test_questions_use_catalog_keys_country_codes_and_numbers():
    examples = load_questions()
    assert examples
    for e in examples:
        expected = e["output"]
        assert set(expected["indicators"]) <= set(CATALOG), e["input"]["question"]
        assert all(re.fullmatch(r"[A-Z]{3}", c) for c in expected.get("countries", []))
        assert all(isinstance(v, int | float) for v in expected.get("figures", []))
        Plan(**{k: v for k, v in expected.items() if k != "figures"})  # valid for the planner


def test_dataset_name_changes_with_the_questions():
    examples = load_questions()
    assert dataset_name(examples) == dataset_name(load_questions())
    assert dataset_name(examples) != dataset_name(examples[1:])


def test_task_runs_the_agent_and_keeps_what_evaluators_need():
    def fetch(ind, countries, start):
        return [fake_dataset(ind)]

    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m"]))
    output = make_task(build_graph(llm, fetch=fetch))({"question": "IPCA?", "history": []})
    assert output["plan"]["indicators"] == ["br_ipca_12m"]
    assert output["model"] == "fake/model"
    assert output["check"]["unverified"] == []  # 4.58 is in the data
    assert plan_match(output, {"indicators": ["br_ipca_12m"]})["score"] == 1
    assert expected_figures(output, {"figures": [4.58]})["score"] == 1
    assert grounding(output)["score"] == 1
