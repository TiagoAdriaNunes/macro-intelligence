"""Deterministic calculations on fetched datasets.

Numbers are computed here with pandas, not by the LLM. The LLM only explains them.
"""

import pandas as pd

RECENT_POINTS = 12  # recent observations passed to the LLM per dataset


def _num(value) -> float:
    return round(float(value), 4)  # plain float, so state stays JSON-serializable


def summarize(dataset: dict) -> dict:
    """Key statistics for one dataset, plus its most recent observations."""
    meta = {k: dataset[k] for k in ("name", "source", "code", "country", "units", "frequency")}
    obs = dataset["observations"]
    if not obs:
        return {**meta, "observations": 0, "note": "No data published for this period."}

    s = pd.Series([o["value"] for o in obs], index=[o["date"] for o in obs])
    first_date, last_date = s.index[0], s.index[-1]
    summary = {
        **meta,
        "observations": len(s),
        "period": f"{first_date} to {last_date}",
        "latest": {"date": last_date, "value": _num(s.iloc[-1])},
        "previous": {"date": s.index[-2], "value": _num(s.iloc[-2])} if len(s) > 1 else None,
        "change_from_previous": _num(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None,
        "first": {"date": first_date, "value": _num(s.iloc[0])},
        "change_over_period": _num(s.iloc[-1] - s.iloc[0]),
        "min": {"date": s.idxmin(), "value": _num(s.min())},
        "max": {"date": s.idxmax(), "value": _num(s.max())},
        "mean": _num(s.mean()),
        "recent": [{"date": d, "value": _num(v)} for d, v in s.tail(RECENT_POINTS).items()],
    }
    # Percent change only for levels (e.g. BRL per USD); a change in a rate is in percentage points
    if "%" not in dataset["units"] and s.iloc[0] != 0:
        summary["pct_change_over_period"] = _num((s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0]) * 100)
    return summary


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
