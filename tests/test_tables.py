"""Tables by year (tables.py), and how the chat stream adds them."""

from test_analysis import USD_BRL
from test_graph import FakeLLM, fake_dataset
from test_stream import show

from macro_intelligence import stream
from macro_intelligence.analysis import summarize
from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.indicators import CATALOG, MONTH_ENDS
from macro_intelligence.stream import FAILED, AnswerStream
from macro_intelligence.tables import yearly_frame, yearly_table_html, yearly_table_text

SUMMARY = summarize(USD_BRL)


def selic_over_two_years() -> dict:
    dataset = fake_dataset(CATALOG["br_selic_target"])
    dataset["observations"] = [
        {"date": "2025-12-31", "value": 15.0},
        {"date": "2026-11-06", "value": 13.75},
    ]
    return dataset


def graph_with_plan(plan: Plan, llm: FakeLLM | None = None):
    return build_graph(llm or FakeLLM(plan=plan), fetch=lambda ind, c, s: [selic_over_two_years()])


def test_frame_has_a_row_per_year():
    frame = yearly_frame(SUMMARY)
    assert list(frame["year"]) == ["2025", "2026"]
    assert list(frame.columns) == ["year", "date", "value", "change", "pct_change", "mean"]


def test_rates_have_no_percent_change_column():
    assert "pct_change" not in yearly_frame(summarize(selic_over_two_years())).columns


def test_html_and_text_show_the_values():
    html = yearly_table_html(SUMMARY)
    assert "USD/BRL exchange rate (BRA)" in html
    assert "5.5013" in html and "0.0011" in html and "series 1" in html

    text = yearly_table_text(SUMMARY)
    assert text.splitlines()[0] == "USD/BRL exchange rate (BRA), BRL per USD"
    assert "-0.0011" in text and "-0.02" in text


def test_chat_shows_the_table_after_the_answer_when_the_plan_asks():
    plan = Plan(indicators=["br_selic_target"], yearly_table=True)
    text = show(AnswerStream(graph_with_plan(plan), "Selic by year?"))

    assert "Year-end value" in text  # the great_tables HTML
    assert text.index("Year-end value") < text.index("**Sources**")


def test_chat_shows_no_table_otherwise():
    text = show(AnswerStream(graph_with_plan(Plan(indicators=["br_selic_target"])), "Selic?"))
    assert "Year-end value" not in text


def test_the_model_is_told_not_to_repeat_the_table():
    plan = Plan(indicators=["br_selic_target"], yearly_table=True)
    llm = FakeLLM(plan=plan)
    graph_with_plan(plan, llm).invoke({"question": "Selic by year?"})
    assert "don't list them again" in llm.prompts[0][-1].content


def test_html_follows_the_chat_theme():
    first, second = yearly_table_html(SUMMARY), yearly_table_html(SUMMARY)
    table_id = first.split('id="', 1)[1].split('"', 1)[0]

    assert table_id != second.split('id="', 1)[1].split('"', 1)[0]  # unique per table
    rule = f"#{table_id} .gt_table, #{table_id} .gt_table * {{ background-color: transparent;"
    assert rule in first
    assert first.index(rule) > first.index(f"#{table_id} .gt_table {{")  # after GT's own CSS
    assert "var(--bs-body-color)" in first and "var(--bs-border-color)" in first


def test_a_failing_table_leaves_the_sources_and_data_check(monkeypatch):
    def broken(summary):
        raise ValueError("rendering failed")

    monkeypatch.setattr(stream, "yearly_table_html", broken)
    plan = Plan(indicators=["br_selic_target"], yearly_table=True)

    text = show(AnswerStream(graph_with_plan(plan), "Selic by year?"))

    assert "**Sources**" in text and "Data check" in text
    assert FAILED not in text


def test_average_of_a_daily_series_is_labelled_as_over_month_ends():
    assert SUMMARY["frequency"] == MONTH_ENDS  # USD/BRL, a daily series reduced to month-ends
    assert "Average of month-ends" in yearly_table_html(SUMMARY)
    assert "Average of month-ends" in yearly_table_text(SUMMARY)

    monthly = summarize(fake_dataset(CATALOG["br_ipca_12m"]))  # two months of 2026
    assert "Year average" in yearly_table_text(monthly)
