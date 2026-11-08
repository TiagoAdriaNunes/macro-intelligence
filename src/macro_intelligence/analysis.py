"""Deterministic calculations on fetched datasets.

Numbers are computed here with pandas, not by the LLM. The LLM only explains them: each
summary includes the changes questions usually ask about (since the previous observation,
year to date, over 12 months, over the whole period, since the low and since the high) and
a row per year, so the model rarely has to calculate.
"""

import pandas as pd

RECENT_POINTS = 12  # recent observations passed to the LLM per dataset


def _num(value: float) -> float:
    return round(float(value), 4)  # plain float, so state stays JSON-serializable


def _point(s: pd.Series, date: str) -> dict:
    return {"date": date, "value": _num(s[date])}


def _change(s: pd.Series, start: str, is_rate: bool) -> dict:
    """From the observation dated `start` to the latest. A change in a rate is in percentage
    points, so only levels (e.g. BRL per USD) also get a percent change."""
    first, last = float(s[start]), float(s.iloc[-1])
    change = {"from": _point(s, start), "change": _num(last - first)}
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


def _yearly(s: pd.Series, is_rate: bool) -> list[dict]:
    """One row per calendar year: its last value (the latest, for the current year), its mean
    for monthly or daily data, and the change from the previous year's last value."""
    by_year: dict[str, list[tuple[str, float]]] = {}
    for date, value in s.items():  # dates are sorted
        by_year.setdefault(str(date)[:4], []).append((str(date), float(value)))
    rows: list[dict] = []
    previous_end: float | None = None
    for year, points in by_year.items():
        end_date, end = points[-1]
        row: dict = {"year": year, "end": {"date": end_date, "value": _num(end)}}
        if len(points) > 1:
            row["mean"] = _num(sum(value for _, value in points) / len(points))
        if previous_end is not None:
            row["change"] = _num(end - previous_end)
            if not is_rate and previous_end != 0:
                row["pct_change"] = _num((end - previous_end) / abs(previous_end) * 100)
        rows.append(row)
        previous_end = end
    return rows


def summarize(dataset: dict) -> dict:
    """Key statistics for one dataset, its changes up to the latest value, and its most
    recent observations."""
    meta = {k: dataset[k] for k in ("name", "source", "code", "country", "units", "frequency")}
    obs = dataset["observations"]
    if not obs:
        return {**meta, "observations": 0, "note": "No data published for this period."}

    s = pd.Series([o["value"] for o in obs], index=[o["date"] for o in obs])
    is_rate = "%" in dataset["units"]
    first_date, last_date = str(s.index[0]), str(s.index[-1])
    return {
        **meta,
        "observations": len(s),
        "period": f"{first_date} to {last_date}",
        "latest": _point(s, last_date),
        "min": _point(s, str(s.idxmin())),
        "max": _point(s, str(s.idxmax())),
        "mean": _num(s.mean()),
        "change_units": "percentage points" if is_rate else dataset["units"],
        "changes": {name: _change(s, start, is_rate) for name, start in _change_starts(s).items()},
        "yearly": _yearly(s, is_rate),
        "recent": [{"date": d, "value": _num(v)} for d, v in s.tail(RECENT_POINTS).items()],
    }


def source_lines(datasets: list[dict]) -> list[str]:
    """One citation line per series, for printing under the answer."""
    seen, lines = set(), []
    for d in datasets:
        key = (d["source"], d["code"], d["country"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{d['name']} ({d['country']}): {d['source']}, series {d['code']}")
    return lines
