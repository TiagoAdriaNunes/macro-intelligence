"""Check that the figures in an answer come from the fetched data.

Plain Python, no LLM: every number the model wrote is looked up among the values it was given
(at the precision it was written, ignoring sign: "fell 0.29 p.p." matches a change of -0.29).
A difference or percent change between two values the answer quotes counts as checked too.
Anything else is reported, so a figure from the model's memory, or a wrong calculation, is
visible in the reply.
"""

import re
from collections.abc import Sequence
from datetime import date
from itertools import combinations

# Numbers as written in an answer: 4.58, 1,234.5, 4,58 (decimal comma), 1.234,56
FIGURE = re.compile(r"(?<![\w.,])\d[\d.,]*\d(?![\w])|(?<![\w.,])\d(?![\w.,]?\d)")
DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")  # observation dates, not figures
SMALL_INTEGER = 10  # bare integers below this ("2 years", list numbers) aren't checked unless a %
YEAR = re.compile(r"(19|20)\d\d")  # a bare year ("no data before 2015") isn't checked either


def _parse(token: str) -> tuple[float, int]:
    """Value and number of decimals of a written number."""
    if "," in token and "." in token:  # the last separator is the decimal one
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        token = token.replace("." if decimal == "," else ",", "").replace(",", ".")
    elif "," in token:
        groups = token.split(",")
        if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]):  # 1,234 thousands
            token = token.replace(",", "")
        else:  # 4,58 decimal comma
            token = token.replace(",", ".")
    elif token.count(".") > 1:  # 1.234.567 thousands
        token = token.replace(".", "")
    decimals = len(token.split(".")[1]) if "." in token else 0
    return float(token), decimals


def figures(text: str) -> list[tuple[str, float, int, bool]]:
    """(as written, value, decimals, followed by %) of each figure in `text` worth checking."""
    text = DATE.sub(" ", text)
    found = []
    for match in FIGURE.finditer(text):
        written = match.group()
        value, decimals = _parse(written)
        percent = text[match.end() : match.end() + 2].lstrip().startswith("%")
        if not percent and ((decimals == 0 and value < SMALL_INTEGER) or YEAR.fullmatch(written)):
            continue
        found.append((written, value, decimals, percent))
    return found


def _numbers_in(text: str) -> set[float]:
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def _known_numbers(summaries: list[dict], texts: list[str]) -> tuple[set[float], set[float]]:
    """Numbers in the data given to the model, as (values, labels). Values are the series' figures
    and numbers in the question; labels are dates, years, codes, names and counts."""
    values: set[float] = set()
    labels: set[float] = {float(date.today().year)}

    def walk(obj, key=""):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            (labels if key == "observations" else values).add(abs(float(obj)))
        elif isinstance(obj, str):
            labels.update(_numbers_in(obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, key)

    walk(summaries)
    for text in texts:
        values.update(_numbers_in(text))
    for s in summaries:  # every year in each period, for "since 2021" or "in 2023"
        years = re.findall(r"\b(\d{4})\b", s.get("period", ""))
        if len(years) == 2:
            labels.update(float(y) for y in range(int(years[0]), int(years[1]) + 1))
    return values, labels


def _calculations(cited: set[float]) -> set[float]:
    """Differences and percent changes between pairs of cited data values."""
    results = set()
    for a, b in combinations(cited, 2):
        difference = abs(a - b)
        results.add(difference)
        results.update(difference / abs(base) * 100 for base in (a, b) if base)
    return results


def check_answer(answer: str, summaries: list[dict], texts: Sequence[str] = ()) -> dict:
    """Look up each figure in `answer` among the numbers in `summaries` and `texts` (the question).

    Returns {"figures": how many were checked, "calculated": how many are a difference or percent
    change between two data values the answer also cites, "unverified": the rest, as written}.
    A figure with % must match a value; other figures may also match a date, code or count.
    """
    values, labels = _known_numbers(summaries, list(texts))
    everything = values | labels
    checked = figures(answer)
    cited: set[float] = set()  # data values the answer quotes
    pending = []
    for written, value, decimals, percent in checked:
        if value == 0:  # needs no source, and must not mark a data value as quoted
            continue
        tolerance = 0.5 * 10**-decimals + 1e-9  # 4.58 matches 4.5813
        hit = min(values if percent else everything, key=lambda k: abs(value - k), default=None)
        if hit is not None and abs(value - hit) <= tolerance:
            if hit in values:
                cited.add(hit)
        else:
            pending.append((written, value, tolerance))

    calculations = _calculations(cited)
    calculated, unverified = 0, []
    for written, value, tolerance in pending:
        if any(abs(value - c) <= tolerance for c in calculations):
            calculated += 1
        elif written not in unverified:
            unverified.append(written)
    return {"figures": len(checked), "calculated": calculated, "unverified": unverified}


def check_line(check: dict | None) -> str:
    """One line for the reply, or "" when there was nothing to check."""
    if not check or not check["figures"]:
        return ""
    n, missing, calculated = check["figures"], check["unverified"], check.get("calculated", 0)
    if not missing:
        detail = f" ({calculated} calculated from it)" if calculated else ""
        return f"✓ Data check: all {n} figures match the fetched data{detail}."
    return (
        f"⚠ Data check: {len(missing)} of {n} figures match neither the fetched data nor a "
        f"calculation from it, so they may be wrong: {', '.join(missing)}"
    )
