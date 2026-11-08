"""LangGraph workflow: plan -> fetch_data -> answer -> verify.

    START -> plan --(indicators chosen)--> fetch_data -> answer -> verify -> END
             plan --(nothing matches)----> no_data -------------------------> END

Only `plan` and `answer` call the LLM. Fetching, calculations and `verify` (which checks every
figure in the answer against the fetched data) are plain Python.
"""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableWithFallbacks
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .analysis import summarize
from .grounding import check_answer
from .indicators import CATALOG, Indicator, catalog_text
from .sources import fetch_indicator

IndicatorKey = Literal[tuple(CATALOG)]  # type: ignore[valid-type]

MAX_YEARS_BACK = 10  # BCB limits daily series to 10 years per request


class Plan(BaseModel):
    """Which data to fetch for the user's question."""

    indicators: list[IndicatorKey] = Field(
        default_factory=list,
        description="Catalog keys needed to answer the question. Empty if none apply.",
    )
    countries: list[str] = Field(
        default_factory=lambda: ["BRA"],
        description="ISO 3166-1 alpha-3 codes (e.g. BRA, USA, ARG) for World Bank indicators.",
    )
    years_back: int = Field(5, description=f"Years of history needed, 1 to {MAX_YEARS_BACK}.")

    @field_validator("countries")
    @classmethod
    def _normalize_countries(cls, v: list[str]) -> list[str]:
        codes = list(dict.fromkeys(c.strip().upper() for c in v if c.strip()))
        return codes or ["BRA"]

    @field_validator("years_back")
    @classmethod
    def _clamp_years(cls, v: int) -> int:
        return max(1, min(v, MAX_YEARS_BACK))


class State(TypedDict):
    question: str
    plan: NotRequired[dict]
    datasets: NotRequired[list[dict]]
    errors: NotRequired[list[str]]
    answer: NotRequired[str]
    model: NotRequired[str]  # model that wrote the answer
    summaries: NotRequired[list[dict]]  # the data the model was given
    check: NotRequired[dict]  # verify: {"figures": n, "unverified": [...]}
    # Earlier chat turns, oldest first: [{"role": "user" | "assistant", "content": "..."}]
    history: NotRequired[list[dict]]


PLANNER_PROMPT = """You plan data retrieval for a macroeconomic research assistant. Today is {today}.

Pick the indicators needed to answer the user's latest question, using only these catalog keys:
{catalog}

Rules:
- Pick only what the question needs. Pick nothing if the catalog can't answer it.
- Use the earlier conversation to resolve follow-ups (e.g. "and Argentina?" repeats the last question for Argentina).
- "Brazil only" indicators always return Brazilian data.
- For "any country" indicators, list every country the question mentions as ISO alpha-3 codes.
- years_back: how many years of history the question needs (1-{max_years}). Default 5."""

ANALYST_PROMPT = """You are a macroeconomic research assistant. Answer the latest question using only the data provided with it.

- Never use outside knowledge for figures. If the data doesn't cover something, say so.
- Give numbers with their units and observation date, rounded sensibly (usually 1-2 decimals).
- Name the source and series code for each figure.
- Don't treat different frequencies or units as comparable (e.g. monthly vs. annual inflation).
- A change in a rate given in % is in percentage points (p.p.), not percent.
- Be concise. Use Markdown: short paragraphs, bullet lists, a table for comparisons."""

HISTORY_MESSAGES = 6  # earlier messages passed to the model
HISTORY_CHARS = 2000  # per message; long answers are cut


def _history_messages(state: State) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in state.get("history", [])[-HISTORY_MESSAGES:]:
        content = str(turn.get("content", ""))[:HISTORY_CHARS]
        if turn.get("role") == "user":
            messages.append(HumanMessage(content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content))
    return messages


Fetcher = Callable[[Indicator, list[str], date], list[dict]]


def _fetch_all(
    fetch: Fetcher, keys: list[str], countries: list[str], start: date
) -> tuple[list[dict], list[str]]:
    """Fetch the indicators at the same time, since a World Bank request can take a minute.
    Returns the datasets, and an error line for each indicator that failed."""

    def fetch_one(key: str) -> tuple[list[dict], str | None]:
        ind = CATALOG[key]
        try:
            return fetch(ind, countries, start), None
        except Exception as exc:  # one failing source shouldn't sink the whole answer
            return [], f"{ind.name} ({ind.source} {ind.code}): {exc}"

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(fetch_one, keys))
    datasets = [dataset for found, _ in results for dataset in found]
    errors = [error for _, error in results if error]
    return datasets, errors


def _years_ago(today: date, years: int) -> date:
    try:
        return today.replace(year=today.year - years)
    except ValueError:  # Feb 29
        return today.replace(year=today.year - years, day=28)


ChatModel = BaseChatModel | RunnableWithFallbacks  # one model, or a chain built by llm.py


def _chat_model(runnable: Runnable) -> BaseChatModel:
    if not isinstance(runnable, BaseChatModel):
        raise TypeError(f"Expected a chat model, got {type(runnable).__name__}")
    return runnable


def _each_model(llm: ChatModel, build: Callable[[BaseChatModel], Runnable]) -> Runnable:
    """Apply `build` to every model of a fallback chain (see llm.py), so a bad reply moves on
    to the next model too, not only an error."""
    if not isinstance(llm, RunnableWithFallbacks):
        return build(llm)
    first, *rest = (build(_chat_model(m)) for m in (llm.runnable, *llm.fallbacks))
    return first.with_fallbacks(rest)


def _require_plan(result: object) -> Plan:
    if not isinstance(result, Plan):  # the model answered without calling the tool
        raise ValueError("The model returned no plan.")
    return result


def _require_text(message: AIMessage) -> AIMessage:
    if not message.text.strip():  # some free models end with nothing, or answer only in thinking
        raise ValueError("The model returned an empty answer.")
    return message


def build_graph(llm: ChatModel, fetch: Fetcher = fetch_indicator):
    """Compile the agent graph. `llm` is a LangChain chat model with tool calling, or a chain
    of them built with `with_fallbacks`."""
    planner = _each_model(
        llm, lambda m: m.with_structured_output(Plan) | RunnableLambda(_require_plan)
    )
    writer = _each_model(llm, lambda m: m | RunnableLambda(_require_text))

    def plan(state: State) -> dict:
        system = PLANNER_PROMPT.format(
            today=date.today().isoformat(), catalog=catalog_text(), max_years=MAX_YEARS_BACK
        )
        messages = [
            SystemMessage(system),
            *_history_messages(state),
            HumanMessage(state["question"]),
        ]
        result = planner.invoke(messages)
        return {"plan": result.model_dump()}

    def route_after_plan(state: State) -> Literal["fetch_data", "no_data"]:
        return "fetch_data" if state["plan"]["indicators"] else "no_data"

    def fetch_data(state: State) -> dict:
        p = state["plan"]
        start = _years_ago(date.today(), p["years_back"])
        datasets, errors = _fetch_all(fetch, p["indicators"], p["countries"], start)
        return {"datasets": datasets, "errors": errors}

    def answer(state: State) -> dict:
        datasets, errors = state.get("datasets", []), state.get("errors", [])
        if not any(d["observations"] for d in datasets):
            detail = (
                "\n".join(f"- {e}" for e in errors) or "- The sources returned no observations."
            )
            return {"answer": f"I couldn't retrieve data to answer this question.\n{detail}"}

        summaries = [summarize(d) for d in datasets]
        payload = json.dumps(summaries, indent=1, ensure_ascii=False)
        message = f"Question: {state['question']}\n\nData:\n{payload}"
        if errors:
            message += "\n\nThese series failed to load:\n" + "\n".join(f"- {e}" for e in errors)
        messages = [SystemMessage(ANALYST_PROMPT), *_history_messages(state), HumanMessage(message)]
        response = writer.invoke(messages)
        return {
            "answer": response.text,
            "model": response.response_metadata.get("model_name", ""),
            "summaries": summaries,
        }

    def verify(state: State) -> dict:
        if not state.get("summaries"):  # no model answer to check (e.g. data failed to load)
            return {}
        return {"check": check_answer(state["answer"], state["summaries"], [state["question"]])}

    def no_data(state: State) -> dict:
        available = "\n".join(f"- {ind.name}" for ind in CATALOG.values())
        return {
            "answer": "None of the indicators I can fetch match this question. Available data:\n"
            + available
        }

    # LangGraph's type hints reject any TypedDict here, though it's the documented way
    builder = StateGraph(State)  # pyrefly: ignore[bad-specialization]
    builder.add_node("plan", plan)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("answer", answer)
    builder.add_node("verify", verify)
    builder.add_node("no_data", no_data)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", route_after_plan)
    builder.add_edge("fetch_data", "answer")
    builder.add_edge("answer", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("no_data", END)
    return builder.compile()
