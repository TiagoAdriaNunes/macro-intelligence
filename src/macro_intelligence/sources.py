"""Fetch indicator data from official sources into one common shape, `Dataset`.

Datasets are plain dicts (JSON types), so the graph state stays serializable; the TypedDicts
below let the type checker verify every key.
"""

import time
import warnings
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TypedDict

import httpx
import pandas as pd
import requests
import wbgapi as wb
from bcb import sgs
from bcb.exceptions import BCBAPIServerError, BCBRateLimitError, SGSError
from loguru import logger

from .indicators import MONTH_ENDS, SOURCE_NAMES, Indicator

REQUEST_TIMEOUT = 30  # seconds; a 10-year daily BCB series can take 10-20
FETCH_ATTEMPTS = 3  # a source that stalls often answers a fresh request in seconds
RETRY_SECONDS = 2  # wait before the 2nd attempt; doubles each time

# pandas warns that it guesses the format of BCB's dd/mm/yyyy dates; python-bcb parses them
# correctly. Set once here: changing filters per request isn't thread-safe, and the graph
# fetches indicators in parallel.
warnings.filterwarnings("ignore", "Could not infer format", UserWarning, r"bcb\.sgs")

# When the World Bank API stalls, a new request usually beats waiting longer on the same one,
# so fetch_indicator() gives up on a request after 30 seconds and tries again.
# wbgapi passes get_options straight to requests.get.
wb.get_options.setdefault("timeout", (10, 30))  # (connect, read) seconds

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
    """Fetch one indicator from `start` until today. Returns one dataset per country.
    Tries again after a timeout or server error, up to FETCH_ATTEMPTS times in all."""
    if ind.source == "bcb":
        return _with_retries(ind, lambda: [_fetch_bcb(ind, start)])
    if ind.source == "worldbank":
        return _with_retries(ind, lambda: _fetch_worldbank(ind, countries, start))
    raise ValueError(f"Unknown source: {ind.source}")


def _temporary(exc: Exception) -> bool:
    """A failure worth another try: timeout, lost connection, server error or rate limit."""
    if isinstance(exc, requests.Timeout | requests.ConnectionError):  # wbgapi
        return True
    if isinstance(exc, httpx.TransportError | BCBAPIServerError | BCBRateLimitError):  # python-bcb
        return True
    if isinstance(exc, wb.APIError):
        return exc.code is not None and (exc.code == 429 or exc.code >= 500)
    # python-bcb raises a plain SGSError for HTTP 5xx too, and for rejections such as a daily
    # window over 10 years, which no retry fixes
    return isinstance(exc, SGSError) and "server error" in str(exc)


def _with_retries(ind: Indicator, fetch: Callable[[], list[Dataset]]) -> list[Dataset]:
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            return fetch()
        except Exception as exc:
            if attempt == FETCH_ATTEMPTS or not _temporary(exc):
                raise
            wait = RETRY_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "Fetching {} failed ({}); trying again in {} s", ind.key, type(exc).__name__, wait
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


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
