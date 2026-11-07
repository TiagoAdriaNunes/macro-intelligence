"""Fetch indicator data from official sources into one common shape.

Every fetcher returns a list of "dataset" dicts (plain JSON types, so graph state
stays serializable):

    {
        "indicator": "br_ipca_12m",
        "name": "IPCA inflation (12-month)",
        "source": "Banco Central do Brasil (SGS)",
        "code": "13522",
        "country": "BRA",
        "units": "% change over 12 months",
        "frequency": "monthly",
        "observations": [{"date": "2026-09-01", "value": 4.58}, ...],
        "fetched_at": "2026-11-07T05:30:00+00:00",
    }
"""

from datetime import UTC, date, datetime

import pandas as pd
import wbgapi as wb
from bcb import sgs

from .indicators import SOURCE_NAMES, Indicator

REQUEST_TIMEOUT = 30  # seconds

# The World Bank API sometimes takes about a minute to respond, so give it more room.
# wbgapi passes get_options straight to requests.get.
wb.get_options.setdefault("timeout", (10, 90))  # (connect, read) seconds


def fetch_indicator(ind: Indicator, countries: list[str], start: date) -> list[dict]:
    """Fetch one indicator from `start` until today. Returns one dataset per country."""
    if ind.source == "bcb":
        return [_fetch_bcb(ind, start)]
    if ind.source == "worldbank":
        return _fetch_worldbank(ind, countries, start)
    raise ValueError(f"Unknown source: {ind.source}")


def _dataset(ind: Indicator, country: str, frequency: str, observations: list[dict]) -> dict:
    return {
        "indicator": ind.key,
        "name": ind.name,
        "source": SOURCE_NAMES[ind.source],
        "code": ind.code,
        "country": country,
        "units": ind.units,
        "frequency": frequency,
        "observations": observations,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _fetch_bcb(ind: Indicator, start: date) -> dict:
    df = sgs.get({ind.key: int(ind.code)}, start=start, timeout=REQUEST_TIMEOUT)
    series = df[ind.key].dropna()

    # The Selic target is published ahead, up to the next Copom meeting. Drop future dates.
    series = series[series.index <= pd.Timestamp(date.today())]

    frequency = ind.frequency
    if ind.frequency == "daily":
        # Keep the last observation of each month so the LLM prompt stays small.
        series = series.groupby(series.index.to_period("M")).tail(1)
        frequency = "monthly (last daily value of each month)"

    observations = [
        {"date": ts.date().isoformat(), "value": float(value)} for ts, value in series.items()
    ]
    return _dataset(ind, "BRA", frequency, observations)


def _fetch_worldbank(ind: Indicator, countries: list[str], start: date) -> list[dict]:
    rows = wb.data.fetch(ind.code, economy=countries, time=range(start.year, date.today().year + 1))
    by_country: dict[str, list[dict]] = {c: [] for c in countries}
    for row in rows:
        if row["value"] is None:  # years not yet published come back empty
            continue
        year = row["time"].removeprefix("YR")
        by_country.setdefault(row["economy"], []).append(
            {"date": year, "value": float(row["value"])}
        )

    return [
        _dataset(ind, country, ind.frequency, sorted(obs, key=lambda o: o["date"]))
        for country, obs in by_country.items()
    ]
