"""Precomputed changes (analysis.summarize), so the model doesn't calculate them."""

from macro_intelligence.analysis import summarize
from macro_intelligence.grounding import check_answer

# USD/BRL month-end values from BCB, as fetched on 2026-11-08
USD_BRL = {
    "name": "USD/BRL exchange rate",
    "source": "Banco Central do Brasil (SGS)",
    "code": "1",
    "country": "BRA",
    "units": "BRL per USD",
    "frequency": "monthly (last daily value of each month)",
    "observations": [
        {"date": d, "value": v}
        for d, v in [
            ("2025-11-28", 5.3338),
            ("2025-12-31", 5.5024),
            ("2026-01-30", 5.2301),
            ("2026-02-27", 5.1495),
            ("2026-03-31", 5.2194),
            ("2026-04-30", 4.9886),
            ("2026-05-29", 5.0569),
            ("2026-06-30", 5.1766),
            ("2026-07-31", 5.0773),
            ("2026-08-31", 5.1816),
            ("2026-09-30", 5.1809),
            ("2026-10-30", 5.4212),
            ("2026-11-06", 5.5013),
        ]
    ],
}


def test_changes_for_a_level_series():
    changes = summarize(USD_BRL)["changes"]

    assert changes["year_to_date"] == {
        "from": {"date": "2025-12-31", "value": 5.5024},
        "change": -0.0011,
        "pct_change": -0.02,
    }
    assert changes["since_previous"]["from"]["date"] == "2026-10-30"
    assert changes["over_12_months"]["from"]["date"] == "2025-11-28"
    assert changes["since_min"] == {
        "from": {"date": "2026-04-30", "value": 4.9886},
        "change": 0.5127,
        "pct_change": 10.2774,
    }
    assert changes["over_period"]["from"]["date"] == "2025-11-28"
    assert "since_max" in changes  # the high (2025-12-31) isn't the latest


def test_the_year_to_date_answer_now_passes_the_check():
    # The model once calculated this as -0.01 (-0.20%); the right figures are in the data now
    answer = "Year to date the dollar is flat: -0.0011 BRL (-0.02%), from 5.5024 to 5.5013."
    assert check_answer(answer, [summarize(USD_BRL)])["unverified"] == []


def test_rates_change_in_points_and_annual_series_skip_monthly_windows():
    gdp = {
        **USD_BRL,
        "units": "% per year, constant prices",
        "frequency": "annual",
        "observations": [
            {"date": "2023", "value": 3.2},
            {"date": "2024", "value": 3.4},
            {"date": "2025", "value": 2.3},
        ],
    }
    summary = summarize(gdp)

    assert summary["change_units"] == "percentage points"
    assert set(summary["changes"]) == {"since_previous", "over_period", "since_max"}
    assert summary["changes"]["since_previous"] == {
        "from": {"date": "2024", "value": 3.4},
        "change": -1.1,
    }
