"""Evaluators for the agent.

Plain functions of the task output (and the example's expected values), so any experiment
runner can call them; experiments.py runs them with Phoenix. An evaluator returns a dict with
`score` (0 to 1), `label` and `explanation`, and leaves out `score` when it doesn't apply to an
example (e.g. no figures to check), so averages only count the examples it applies to.

- plan_match:       did the planner pick the expected indicators, countries, dates and flags?
- expected_figures: does the answer give the correct figures for the question's date?
- grounding:        share of the answer's figures found in the data (grounding.py, no LLM)
- answer_quality:   an LLM judge for the analyst prompt's rules (units, dates, sources...)
"""

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .graph import ChatModel, _each_model
from .grounding import figures

Result = dict[str, Any]

PLAN_FLAGS = ("about_catalog", "yearly_table")  # checked only when the example sets them


def plan_match(output: dict, expected: dict) -> Result:
    """1 when the plan has exactly the expected indicators, and, where the example gives them,
    the expected countries and flags and a start early enough; else 0 with the differences."""
    plan = output.get("plan")
    if plan is None:
        return {"score": 0.0, "label": "no plan", "explanation": "The planner returned no plan."}
    differences = []
    got, want = sorted(plan["indicators"]), sorted(expected.get("indicators", []))
    if got != want:
        differences.append(f"indicators {got}, expected {want}")
    if "countries" in expected and sorted(plan["countries"]) != sorted(expected["countries"]):
        differences.append(
            f"countries {sorted(plan['countries'])}, expected {sorted(expected['countries'])}"
        )
    # start_year in the example: the latest start that still covers the question's dates
    if "start_year" in expected and plan["start_year"] > expected["start_year"]:
        differences.append(
            f"start_year {plan['start_year']}, needs {expected['start_year']} or earlier"
        )
    for flag in PLAN_FLAGS:
        if flag in expected and plan[flag] != expected[flag]:
            differences.append(f"{flag} {plan[flag]}, expected {expected[flag]}")
    if differences:
        return {"score": 0.0, "label": "mismatch", "explanation": "; ".join(differences)}
    return {"score": 1.0, "label": "match", "explanation": "The plan matches."}


def _decimals(value: float) -> int:
    text = repr(float(value))
    return len(text.split(".")[1]) if "." in text and "e" not in text else 0


def _tolerance(decimals: int) -> float:
    return 0.5 * 10**-decimals + 1e-9  # 4.58 matches 4.5813


def expected_figures(output: dict, expected: dict) -> Result:
    """Share of the example's `figures` the answer gives, at the precision it writes them
    ("6.19" matches 6.1923) and ignoring sign ("fell 1.86%" matches -1.86)."""
    wanted = expected.get("figures")
    if not wanted:
        return {"label": "no figures expected", "explanation": "The example lists no figures."}
    written = figures(output.get("answer", ""))
    missing = [
        value
        for value in wanted
        if not any(
            abs(abs(value) - value_in_answer) <= _tolerance(decimals) + _tolerance(_decimals(value))
            for _, value_in_answer, decimals, _ in written
        )
    ]
    score = 1 - len(missing) / len(wanted)
    if not missing:
        return {"score": score, "label": "correct", "explanation": f"All {len(wanted)} found."}
    return {
        "score": score,
        "label": "missing",
        "explanation": f"Not in the answer: {', '.join(str(v) for v in missing)}",
    }


def grounding(output: dict) -> Result:
    """Share of the answer's figures that are in the data or calculated from it."""
    check = output.get("check")
    if not check or not check["figures"]:
        return {"label": "no figures", "explanation": "The answer has no figures to check."}
    n, unverified = check["figures"], check["unverified"]
    # `unverified` lists each figure once, as written; `figures` counts every occurrence
    score = max(0.0, 1 - len(unverified) / n)
    if not unverified:
        return {"score": score, "label": "grounded", "explanation": f"All {n} figures match."}
    return {
        "score": score,
        "label": "unverified",
        "explanation": f"{len(unverified)} of {n} figures not in the data: {', '.join(unverified)}",
    }


# LLM judge

JUDGE_PROMPT = """You grade answers from a macroeconomic research assistant. The assistant was given a question and data (JSON) and had to answer using only that data.

Grade the answer on each criterion below, true or false:
- answers_question: it answers the question asked, or says clearly what the data doesn't cover.
- units_and_dates: each figure it gives has its unit (%, p.p., BRL per USD...) and observation date or year.
- names_sources: it names the source (e.g. Banco Central do Brasil, World Bank) and the series code of each figure.
- correct_terms: a change in a rate given in % is called percentage points (p.p.), not percent, and it doesn't compare figures of different frequencies or units as if comparable.
- concise: it quotes only the figures the question needs, without listing every statistic.

Don't check the figures themselves against the data; that is checked separately. In `explanation`, say briefly what failed, or "All criteria met"."""

JUDGE_DATA_CHARS = 12_000  # data shown to the judge; the rest is cut


class Verdict(BaseModel):
    """The judge's grade for one answer."""

    answers_question: bool
    units_and_dates: bool
    names_sources: bool
    correct_terms: bool
    concise: bool
    explanation: str = Field(description="What failed, briefly, or 'All criteria met'.")


CRITERIA = [name for name in Verdict.model_fields if name != "explanation"]


def make_answer_quality(llm: ChatModel) -> Callable[[dict, dict], Result]:
    """The `answer_quality` evaluator, judged by `llm` (a chat model or fallback chain)."""
    judge = _each_model(llm, lambda m: m.with_structured_output(Verdict))

    def answer_quality(input: dict, output: dict) -> Result:
        if not output.get("summaries"):  # catalog listing, or no data: nothing the model wrote
            return {"label": "skipped", "explanation": "No model answer from data."}
        data = json.dumps(output["summaries"], ensure_ascii=False)[:JUDGE_DATA_CHARS]
        message = f"Question: {input['question']}\n\nData:\n{data}\n\nAnswer:\n{output['answer']}"
        verdict = judge.invoke([("system", JUDGE_PROMPT), ("human", message)])
        if not isinstance(verdict, Verdict):
            raise TypeError(f"The judge returned {type(verdict).__name__}, not a Verdict")
        failed = [c for c in CRITERIA if not getattr(verdict, c)]
        return {
            "score": 1 - len(failed) / len(CRITERIA),
            "label": "pass" if not failed else "fail: " + ", ".join(failed),
            "explanation": verdict.explanation,
            "metadata": verdict.model_dump(),
        }

    return answer_quality
