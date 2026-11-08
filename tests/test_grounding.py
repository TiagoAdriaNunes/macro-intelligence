"""The data check (grounding.py) and the graph's verify node."""

from test_graph import FakeLLM, fake_dataset

from macro_intelligence.analysis import summarize
from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.grounding import check_answer, check_line, figures
from macro_intelligence.indicators import CATALOG

SUMMARY = summarize(fake_dataset(CATALOG["br_ipca_12m"], values=(4.8713, 4.2201, 4.5813)))


def unverified(answer: str) -> list[str]:
    return check_answer(answer, [SUMMARY])["unverified"]


def test_figures_parses_formats_and_skips_dates_and_small_counts():
    text = "On 2026-09-01 it was 4.58% (4,58% in Portuguese), 1,234.5 or 1.234,5; over 2 years, 3%."
    assert [(w, v) for w, v, _, _ in figures(text)] == [
        ("4.58", 4.58),
        ("4,58", 4.58),
        ("1,234.5", 1234.5),
        ("1.234,5", 1234.5),
        ("3", 3.0),
    ]


def test_figures_from_the_data_pass_at_their_written_precision():
    assert unverified("IPCA was 4.58% in Sep 2026, down from 4.87% (min 4.22%).") == []
    assert unverified("It rose 0.36 p.p. from the previous month, fell 0.29 p.p. overall.") == []
    assert unverified("Series 13522, 12-month inflation, 3 observations.") == []


def test_figures_not_in_the_data_are_reported_once():
    result = check_answer("Inflation hit 6.1% (6.1% is high); the target is 3%.", [SUMMARY])
    assert result == {"figures": 3, "calculated": 0, "unverified": ["6.1", "3"]}


def test_calculations_between_cited_values_are_checked():
    usd = summarize(fake_dataset(CATALOG["br_usd_brl"], values=(5.2301, 5.4212, 5.5013)))
    # 5.2301 -> 5.4212 isn't a precomputed change in the summary
    answer = "It rose from 5.2301 to 5.4212, up 0.19 BRL (3.65%); no change (0.0) the week before."
    assert check_answer(answer, [usd]) == {"figures": 5, "calculated": 2, "unverified": []}
    # the same calculation without quoting the values it comes from isn't accepted
    assert check_answer("It rose 0.19 BRL (3.65%).", [usd])["unverified"] == ["0.19", "3.65"]
    assert check_answer("It rose 0.29 BRL from 5.2301 to 5.4212.", [usd])["unverified"] == ["0.29"]


def test_a_zero_does_not_mark_a_data_value_as_quoted():
    # 0.0 is nearest to 0.29 (a change in the data); 4.58 - 0.29 = 4.29 must not count as calculated
    assert unverified("IPCA was 4.58%, monthly change 0.0%, up 4.29 p.p.") == ["4.29"]


def test_numbers_from_the_question_count_as_data():
    assert (
        check_answer("No, it's below 5%.", [SUMMARY], ["Is inflation above 5%?"])["unverified"]
        == []
    )


def test_percent_change_of_levels_is_in_the_data():
    usd = summarize(fake_dataset(CATALOG["br_usd_brl"], values=(5.2301, 5.5013)))
    assert usd["changes"]["over_period"]["pct_change"] == 5.1854
    assert check_answer("The real weakened about 5.2%, to 5.50 per USD.", [usd])["unverified"] == []
    assert "pct_change" not in SUMMARY["changes"]["over_period"]  # rates change in p.p., not %


def test_check_line():
    assert check_line(None) == "" and check_line({"figures": 0, "unverified": []}) == ""
    assert check_line({"figures": 4, "unverified": []}).startswith("✓ Data check: all 4 figures")
    assert "(1 calculated from it)" in check_line({"figures": 4, "calculated": 1, "unverified": []})
    assert "1 of 4 figures match neither the fetched data" in check_line(
        {"figures": 4, "unverified": ["6.1"]}
    )


def test_graph_verifies_the_answer():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m"]), answer="IPCA was 4.58%, not 9.9%.")
    result = build_graph(llm, fetch=lambda ind, c, s: [fake_dataset(ind)]).invoke(
        {"question": "IPCA?"}
    )
    assert result["check"] == {"figures": 2, "calculated": 0, "unverified": ["9.9"]}


def test_graph_skips_verify_when_no_model_answered():
    def fetch(ind, countries, start):
        raise ConnectionError("down")

    result = build_graph(FakeLLM(plan=Plan(indicators=["br_ipca_12m"])), fetch=fetch).invoke(
        {"question": "IPCA?"}
    )
    assert "check" not in result


def test_bare_years_are_not_checked_but_amounts_are():
    assert unverified("No data for 2000-2015; it was 4.58% in 2026.") == []
    assert unverified("Reserves reached 2,015 billion.") == ["2,015"]  # an amount, not a year
    assert unverified("It rose 2015%.") == ["2015"]  # a percentage is always checked
