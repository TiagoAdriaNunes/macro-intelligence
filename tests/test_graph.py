"""Graph tests with a fake LLM and fake data: no network, no API key."""

import subprocess
import sys
from datetime import date

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from macro_intelligence import sources
from macro_intelligence.analysis import source_lines, summarize
from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.indicators import CATALOG


class FakeLLM(BaseChatModel):
    """Returns a fixed plan from structured output and a fixed answer from invoke."""

    plan: Plan | None = None  # None: answers without calling the tool
    answer: str = "Inflation was 4.58% in September 2026."
    name_: str = "fake/model"
    error: str | None = None  # raise this instead of answering
    prompts: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.error:
            raise ConnectionError(self.error)
        self.prompts.append(messages)
        message = AIMessage(self.answer, response_metadata={"model_name": self.name_})
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def plan(_):
            if self.error:
                raise ConnectionError(self.error)
            return self.plan

        return RunnableLambda(plan)


def fake_dataset(ind, country="BRA", values=(4.22, 4.58)):
    return {
        "indicator": ind.key,
        "name": ind.name,
        "source": "Test source",
        "code": ind.code,
        "country": country,
        "units": ind.units,
        "frequency": ind.frequency,
        "observations": [{"date": f"2026-0{8 + i}-01", "value": v} for i, v in enumerate(values)],
        "fetched_at": "2026-11-07T00:00:00+00:00",
    }


def test_happy_path_fetches_and_answers():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m"]))
    calls = []

    def fetch(ind, countries, start):
        calls.append((ind.key, countries, start))
        return [fake_dataset(ind)]

    result = build_graph(llm, fetch=fetch).invoke({"question": "What is inflation in Brazil?"})

    assert result["answer"] == llm.answer
    assert result["model"] == "fake/model"
    assert [c[0] for c in calls] == ["br_ipca_12m"]
    assert calls[0][2].year == date.today().year - 5
    prompt = llm.prompts[0][1].content
    assert "4.58" in prompt and "13522" in prompt


def test_unmatched_question_skips_fetch_and_llm_answer():
    llm = FakeLLM(plan=Plan(indicators=[]))

    def fetch(*args):
        raise AssertionError("should not fetch")

    result = build_graph(llm, fetch=fetch).invoke({"question": "Best pizza in Sao Paulo?"})

    assert "None of the indicators" in result["answer"]
    assert llm.prompts == []


def test_question_about_the_catalog_lists_it_without_fetching():
    llm = FakeLLM(plan=Plan(about_catalog=True, indicators=["br_selic_target"]))

    def fetch(*args):
        raise AssertionError("should not fetch")

    result = build_graph(llm, fetch=fetch).invoke({"question": "What can I ask you?"})

    assert "None of the indicators" not in result["answer"]
    assert "**Brazil**" in result["answer"] and "**Any country**" in result["answer"]
    assert all(ind.name in result["answer"] for ind in CATALOG.values())
    assert llm.prompts == []  # answered without the LLM


def test_one_failing_source_is_reported_not_fatal():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m", "wb_gdp_growth"], countries=["BRA"]))

    def fetch(ind, countries, start):
        if ind.source == "worldbank":
            raise ConnectionError("timeout")
        return [fake_dataset(ind)]

    result = build_graph(llm, fetch=fetch).invoke({"question": "Inflation and growth?"})

    assert result["errors"] == ["GDP growth (worldbank NY.GDP.MKTP.KD.ZG): timeout"]
    assert "failed to load" in llm.prompts[0][1].content


def test_all_sources_failing_returns_error_without_llm():
    llm = FakeLLM(plan=Plan(indicators=["br_selic_target"]))

    def fetch(ind, countries, start):
        raise ConnectionError("BCB is down")

    result = build_graph(llm, fetch=fetch).invoke({"question": "Selic?"})

    assert "couldn't retrieve data" in result["answer"]
    assert "BCB is down" in result["answer"]
    assert llm.prompts == []


def test_fallback_model_takes_over_after_errors_and_bad_replies():
    plan = Plan(indicators=["br_ipca_12m"])
    no_plan = FakeLLM(plan=None, answer="")  # skips the tool, then answers with nothing
    rate_limited = FakeLLM(plan=plan, error="429 Too Many Requests")
    working = FakeLLM(plan=plan, name_="good/model")
    llm = no_plan.with_fallbacks([rate_limited, working])

    result = build_graph(llm, fetch=lambda ind, c, s: [fake_dataset(ind)]).invoke(
        {"question": "IPCA?"}
    )

    assert result["plan"]["indicators"] == ["br_ipca_12m"]
    assert result["model"] == "good/model"
    assert len(working.prompts) == 1


def test_plan_normalizes_countries_and_clamps_years():
    plan = Plan(indicators=["wb_inflation"], countries=[" usa", "bra", "USA"], years_back=40)
    assert plan.countries == ["USA", "BRA"]
    assert plan.years_back == 10
    assert Plan(countries=[]).countries == ["BRA"]


def test_summarize_and_sources():
    ds = fake_dataset(CATALOG["br_ipca_12m"])
    s = summarize(ds)
    assert s["latest"] == {"date": "2026-09-01", "value": 4.58}
    assert s["changes"]["over_period"] == {
        "from": {"date": "2026-08-01", "value": 4.22},
        "change": 0.36,
    }
    assert source_lines([ds, ds]) == ["IPCA inflation (12-month) (BRA): Test source, series 13522"]


def test_bcb_drops_future_dates_and_condenses_daily(monkeypatch):
    today = pd.Timestamp(date.today())
    index = pd.DatetimeIndex(
        [
            today - pd.Timedelta(days=40),
            today - pd.Timedelta(days=1),
            today,
            today + pd.Timedelta(days=30),
        ],
        name="Date",
    )
    frame = pd.DataFrame({"br_selic_target": [15.0, 14.75, 14.75, 14.5]}, index=index)
    monkeypatch.setattr(sources.sgs, "get", lambda *a, **k: frame)

    [ds] = sources.fetch_indicator(CATALOG["br_selic_target"], ["BRA"], date(2026, 1, 1))

    dates = [o["date"] for o in ds["observations"]]
    assert today.date().isoformat() in dates
    assert all(d <= today.date().isoformat() for d in dates)
    assert len(dates) == len({d[:7] for d in dates})  # one value per month
    assert ds["frequency"].startswith("monthly")


def test_worldbank_rows_become_one_sorted_dataset_per_country(monkeypatch):
    requested = {}

    def fake_fetch(code, economy, time):
        requested["years"] = list(time)
        return [
            {"economy": "ARG", "time": "YR2025", "value": None},  # not published yet
            {"economy": "ARG", "time": "YR2024", "value": -1.3},
            {"economy": "BRA", "time": "YR2024", "value": 3.4},
            {"economy": "BRA", "time": "YR2023", "value": 3.2},
        ]

    monkeypatch.setattr(sources.wb.data, "fetch", fake_fetch)
    this_year = date.today().year

    datasets = sources.fetch_indicator(
        CATALOG["wb_gdp_growth"], ["BRA", "ARG"], date(this_year - 1, 1, 1)
    )

    assert [d["country"] for d in datasets] == ["BRA", "ARG"]
    assert datasets[0]["observations"] == [
        {"date": "2023", "value": 3.2},
        {"date": "2024", "value": 3.4},
    ]
    assert datasets[1]["observations"] == [{"date": "2024", "value": -1.3}]
    # a 1-year window still asks for the last 3 years, so published figures are found
    assert requested["years"] == [this_year - 2, this_year - 1, this_year]


def test_bcb_error_page_gives_a_clear_error(monkeypatch):
    def rejected(*args, **kwargs):
        raise ValueError("Expected object or value")  # python-bcb parsing BCB's HTML page

    monkeypatch.setattr(sources.sgs, "get", rejected)
    with pytest.raises(RuntimeError, match="BCB sent an error page"):
        sources.fetch_indicator(CATALOG["br_ipca_12m"], ["BRA"], date(2026, 1, 1))


def test_an_answer_sent_twice_is_kept_once():
    answer = "Selic is 13.75%.\n\nIt fell 1.25 p.p. this year."
    llm = FakeLLM(plan=Plan(indicators=["br_selic_target"]), answer=answer + answer)

    result = build_graph(llm, fetch=lambda ind, c, s: [fake_dataset(ind)]).invoke(
        {"question": "Selic?"}
    )

    assert result["answer"] == answer


def test_bcb_date_warning_is_hidden_by_a_filter_set_once():
    # In a fresh process, since pytest resets warning filters: importing sources sets the
    # filter once (changing filters per request isn't thread-safe, and fetches run in parallel)
    code = "; ".join(
        [
            "import warnings, macro_intelligence.sources",
            "m = 'Could not infer format, so each element will be parsed individually'",
            "warnings.warn_explicit(m, UserWarning, 'bcb/sgs/__init__.py', 400, module='bcb.sgs')",
            "warnings.warn_explicit(m, UserWarning, 'other.py', 1, module='other')",
        ]
    )
    stderr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stderr
    assert "bcb/sgs" not in stderr  # hidden
    assert "other.py" in stderr  # other warnings still show
