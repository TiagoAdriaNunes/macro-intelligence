"""Fetch indicator data from official sources into one common shape, `Dataset`.

Datasets are plain dicts (JSON types), so the graph state stays serializable; the TypedDicts
below let the type checker verify every key.
"""

import warnings
from datetime import UTC, date, datetime
from typing import TypedDict

import pandas as pd
import wbgapi as wb
from bcb import sgs

from .indicators import MONTH_ENDS, SOURCE_NAMES, Indicator

REQUEST_TIMEOUT = 30  # seconds

# pandas warns that it guesses the format of BCB's dd/mm/yyyy dates; python-bcb parses them
# correctly. Set once here: changing filters per request isn't thread-safe, and the graph
# fetches indicators in parallel.
warnings.filterwarnings("ignore", "Could not infer format", UserWarning, r"bcb\.sgs")

# The World Bank API sometimes takes about a minute to respond, so give it more room.
# wbgapi passes get_options straight to requests.get.
wb.get_options.setdefault("timeout", (10, 90))  # (connect, read) seconds

# A year's World Bank figures come out during the next year, so always ask for a few years:
# otherwise a short window early in the year finds nothing published.
WORLD_BANK_MIN_YEARS = 3


class Observation(TypedDict):
    date: str  # YYYY-MM-DD, or YYYY for annual data
    value: float


class Dataset(TypedDict):
    indicator: str  # catalog key, e.g. "br_ipca_12m"
    name: str
    source: str  # e.g. "Banco Central do Brasil (SGS)"
    code: str  # the source's series code
    country: str  # ISO 3166-1 alpha-3
    units: str
    frequency: str  # as published, or MONTH_ENDS for a reduced daily series
    observations: list[Observation]
    fetched_at: str  # ISO 8601, UTC


def fetch_indicator(ind: Indicator, countries: list[str], start: date) -> list[Dataset]:
    """Fetch one indicator from `start` until today. Returns one dataset per country."""
    if ind.source == "bcb":
        return [_fetch_bcb(ind, start)]
    if ind.source == "worldbank":
        return _fetch_worldbank(ind, countries, start)
    raise ValueError(f"Unknown source: {ind.source}")


def _dataset(
    ind: Indicator, country: str, frequency: str, observations: list[Observation]
) -> Dataset:
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


def _dates(series: pd.Series) -> pd.DatetimeIndex:
    """The series' index as dates; pandas types `series.index` as a plain Index."""
    return pd.DatetimeIndex(series.index)


def _fetch_bcb(ind: Indicator, start: date) -> Dataset:
    try:
        df = sgs.get({ind.key: int(ind.code)}, start=start, timeout=REQUEST_TIMEOUT)
    except ValueError as exc:  # python-bcb can't parse the HTML page BCB sends on a rejection
        raise RuntimeError("BCB sent an error page instead of data") from exc
    assert isinstance(df, pd.DataFrame)  # sgs.get returns a list only with multi=False
    series = df[ind.key].dropna()

    # The Selic target is published ahead, up to the next Copom meeting. Drop future dates.
    series = series[_dates(series) <= pd.Timestamp(date.today())]

    frequency = ind.frequency
    if ind.frequency == "daily":
        # Keep the last observation of each month so the LLM prompt stays small.
        series = series.groupby(_dates(series).to_period("M")).tail(1)
        frequency = MONTH_ENDS

    observations: list[Observation] = [
        {"date": ts.date().isoformat(), "value": float(value)}
        for ts, value in zip(_dates(series), series, strict=True)
    ]
    return _dataset(ind, "BRA", frequency, observations)


def _fetch_worldbank(ind: Indicator, countries: list[str], start: date) -> list[Dataset]:
    this_year = date.today().year
    first_year = min(start.year, this_year - WORLD_BANK_MIN_YEARS + 1)
    rows = wb.data.fetch(ind.code, economy=countries, time=range(first_year, this_year + 1))
    by_country: dict[str, list[Observation]] = {c: [] for c in countries}
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
