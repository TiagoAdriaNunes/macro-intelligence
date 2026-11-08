"""Deterministic calculations on fetched datasets.

Numbers are computed here with pandas, not by the LLM. The LLM only explains them: each
summary includes the changes questions usually ask about (since the previous observation,
year to date, over 12 months, over the whole period, since the low and since the high) and
a row per year, so the model rarely has to calculate.
"""

from typing import NotRequired, TypedDict

import pandas as pd

from .sources import Dataset

RECENT_POINTS = 12  # recent observations passed to the LLM per dataset


class Point(TypedDict):
    date: str
    value: float


# "from" is a Python keyword, so this one uses the functional syntax
Change = TypedDict(
    "Change",
    {"from": Point, "change": float, "pct_change": NotRequired[float]},  # % only for levels
)


class YearRow(TypedDict):
    year: str
    end: Point  # last value of the year; the latest, for the current year
    mean: NotRequired[float]  # monthly or daily data only
    change: NotRequired[float]  # from the previous year's end
    pct_change: NotRequired[float]


class Summary(TypedDict):
    """What the model is given about one dataset. A dataset without observations has only
    the first fields and `note`."""

    name: str
    source: str
    code: str
    country: str
    units: str
    frequency: str
    observations: int  # how many
    note: NotRequired[str]
    period: NotRequired[str]
    latest: NotRequired[Point]
    min: NotRequired[Point]
    max: NotRequired[Point]
    mean: NotRequired[float]
    change_units: NotRequired[str]
    changes: NotRequired[dict[str, Change]]  # by name, e.g. "year_to_date"
    yearly: NotRequired[list[YearRow]]
    recent: NotRequired[list[Point]]


def _num(value: float) -> float:
    return round(float(value), 4)  # plain float, so state stays JSON-serializable


def _point(s: pd.Series, date: str) -> Point:
    return {"date": date, "value": _num(s[date])}


def _change(s: pd.Series, start: str, is_rate: bool) -> Change:
    """From the observation dated `start` to the latest. A change in a rate is in percentage
    points, so only levels (e.g. BRL per USD) also get a percent change."""
    first, last = float(s[start]), float(s.iloc[-1])
    change: Change = {"from": _point(s, start), "change": _num(last - first)}
    if not is_rate and first != 0:
        change["pct_change"] = _num((last - first) / abs(first) * 100)
    return change


def _change_starts(s: pd.Series) -> dict[str, str]:
    """Start date of each change to precompute, by name; the end is always the latest."""
    dates = [str(d) for d in s.index]
    latest = dates[-1]
    starts = {"since_previous": dates[-2] if len(dates) > 1 else latest}
    if len(latest) >= 7:  # monthly or daily (YYYY-MM-DD); annual series are dated by year only
        year, month = int(latest[:4]), latest[5:7]
        before_this_year = [d for d in dates if d < f"{year}-01"]
        if before_this_year:  # the last value of the previous year
            starts["year_to_date"] = before_this_year[-1]
        a_year_ago = [d for d in dates if d[:7] == f"{year - 1}-{month}"]
        if a_year_ago:
            starts["over_12_months"] = a_year_ago[-1]
    starts["over_period"] = dates[0]
    starts["since_min"] = str(s.idxmin())
    starts["since_max"] = str(s.idxmax())
    return {name: start for name, start in starts.items() if start != latest}


def _yearly(s: pd.Series, is_rate: bool) -> list[YearRow]:
    """One row per calendar year: its last value (the latest, for the current year), the mean
    of its values for monthly data (of its month-end values, for a daily series reduced by
    sources.py), and the change from the previous year's last value."""
    by_year: dict[str, list[tuple[str, float]]] = {}
    for date, value in s.items():  # dates are sorted
        by_year.setdefault(str(date)[:4], []).append((str(date), float(value)))
    rows: list[YearRow] = []
    previous_end: float | None = None
    for year, points in by_year.items():
        end_date, end = points[-1]
        row: YearRow = {"year": year, "end": {"date": end_date, "value": _num(end)}}
        if len(points) > 1:
            row["mean"] = _num(sum(value for _, value in points) / len(points))
        if previous_end is not None:
            row["change"] = _num(end - previous_end)
            if not is_rate and previous_end != 0:
                row["pct_change"] = _num((end - previous_end) / abs(previous_end) * 100)
        rows.append(row)
        previous_end = end
    return rows


def summarize(dataset: Dataset) -> Summary:
    """Key statistics for one dataset, its changes up to the latest value, and its most
    recent observations."""
    obs = dataset["observations"]
    summary: Summary = {
        "name": dataset["name"],
        "source": dataset["source"],
        "code": dataset["code"],
        "country": dataset["country"],
        "units": dataset["units"],
        "frequency": dataset["frequency"],
        "observations": len(obs),
    }
    if not obs:
        summary["note"] = "No data published for this period."
        return summary

    s = pd.Series([o["value"] for o in obs], index=[o["date"] for o in obs])
    is_rate = "%" in dataset["units"]
    first_date, last_date = str(s.index[0]), str(s.index[-1])
    summary["period"] = f"{first_date} to {last_date}"
    summary["latest"] = _point(s, last_date)
    summary["min"] = _point(s, str(s.idxmin()))
    summary["max"] = _point(s, str(s.idxmax()))
    summary["mean"] = _num(s.mean())
    summary["change_units"] = "percentage points" if is_rate else dataset["units"]
    summary["changes"] = {
        name: _change(s, start, is_rate) for name, start in _change_starts(s).items()
    }
    summary["yearly"] = _yearly(s, is_rate)
    summary["recent"] = [
        {"date": str(d), "value": _num(v)} for d, v in s.tail(RECENT_POINTS).items()
    ]
    return summary


def source_lines(datasets: list[Dataset]) -> list[str]:
    """One citation line per series, for printing under the answer."""
    seen, lines = set(), []
    for d in datasets:
        key = (d["source"], d["code"], d["country"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{d['name']} ({d['country']}): {d['source']}, series {d['code']}")
    return lines
