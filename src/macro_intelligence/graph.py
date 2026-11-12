"""LangGraph workflow: plan -> fetch_data -> answer -> verify.

    START -> plan --(indicators chosen)--> fetch_data -> answer -> verify -> END
             plan --(asks what's covered)-> catalog --------------------------> END
             plan --(nothing matches)----> no_data ---------------------------> END

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

from .analysis import Summary, summarize
from .grounding import Check, check_answer
from .indicators import CATALOG, Indicator, catalog_markdown, catalog_text
from .sources import Dataset, fetch_indicator

IndicatorKey = Literal[tuple(CATALOG)]  # type: ignore[valid-type]

MAX_YEARS_BACK = 10  # BCB limits daily series to 10 years per request
DEFAULT_YEARS_BACK = 5


def _default_start_year() -> int:
    return date.today().year - DEFAULT_YEARS_BACK


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
    # A year, not a number of years back: models read "end of 2023" as a year reliably, but often
    # get the years between it and today wrong
    start_year: int = Field(
        default_factory=_default_start_year,
        description="First year of data needed: the earliest year the question mentions, or "
        "the current year minus N for 'the last N years'.",
    )
    about_catalog: bool = Field(
        False,
        description="True only when the user asks in general what data or questions this "
        "assistant covers, with no country or topic to look up.",
    )
    yearly_table: bool = Field(
        False,
        description="True when the user asks how values evolved over several years, or for "
        "values by year, so a table by year helps.",
    )

    @field_validator("countries")
    @classmethod
    def _normalize_countries(cls, v: list[str]) -> list[str]:
        codes = list(dict.fromkeys(c.strip().upper() for c in v if c.strip()))
        return codes or ["BRA"]

    @field_validator("start_year")
    @classmethod
    def _clamp_start_year(cls, v: int) -> int:
        this_year = date.today().year
        return max(this_year - MAX_YEARS_BACK, min(v, this_year))


class Turn(TypedDict):
    """An earlier chat message."""

    role: Literal["user", "assistant"]
    content: str


class State(TypedDict):
    question: str
    plan: NotRequired[Plan]
    datasets: NotRequired[list[Dataset]]
    errors: NotRequired[list[str]]
    answer: NotRequired[str]
    model: NotRequired[str]  # model that wrote the answer
    summaries: NotRequired[list[Summary]]  # the data the model was given
    check: NotRequired[Check]
    history: NotRequired[list[Turn]]  # earlier chat messages, oldest first


PLANNER_PROMPT = """You plan data retrieval for a macroeconomic research assistant. Today is {today}.

Pick the indicators needed to answer the user's latest question, using only these catalog keys:
{catalog}

Rules:
- Pick only what the question needs. Pick nothing if the catalog can't answer it.
- Set about_catalog (and pick nothing) only for general questions about what you can answer, e.g. "what can I ask?" or "which data do you have?".
- When the question names a country, look its data up instead, even if it says "indicators": "indicators for France" means every "any country" indicator for FRA.
- Use the earlier conversation to resolve follow-ups (e.g. "and Argentina?" repeats the last question for Argentina).
- "Brazil only" indicators always return Brazilian data.
- For "any country" indicators, list every country the question mentions as ISO alpha-3 codes.
- start_year: the first year of data the question needs. Use the earliest year it mentions ("at the end of 2023" -> 2023, "from 2019 to 2023" -> 2019), or this year minus N for "the last N years". For follow-ups, keep the earlier question's period. Default {default_start}; no earlier than {min_year}.
- Set yearly_table when the user asks how something evolved over several years or for values by year."""

ANALYST_PROMPT = """You are a macroeconomic research assistant. Answer the latest question using only the data provided with it.

- Never use outside knowledge for figures. If the data doesn't cover something, say so.
- Give numbers with their units and observation date, rounded sensibly (usually 1-2 decimals; keep more for a small change, e.g. -0.0011, rather than rounding it to a different number).
- For changes, quote the precomputed `changes` (since_previous, year_to_date, over_12_months, over_period, since_min, since_max) instead of calculating. If you must calculate, also quote the two values you used.
- When the question spans several years or asks how something evolved, describe the trend from `yearly` (year-end values, changes from the previous year).
- Quote only the figures the question needs; don't list every statistic provided.
- Name the source and series code for each figure.
- Don't treat different frequencies or units as comparable (e.g. monthly vs. annual inflation).
- A change in a rate given in % is in percentage points (p.p.), not percent.
- Be concise. Use Markdown: short paragraphs, bullet lists, a table for comparisons."""

HISTORY_MESSAGES = 6  # earlier messages passed to the model
HISTORY_CHARS = 2000  # per message; long answers are cut


def _history_messages(state: State) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in state.get("history", [])[-HISTORY_MESSAGES:]:
        content = turn["content"][:HISTORY_CHARS]
        messages.append(HumanMessage(content) if turn["role"] == "user" else AIMessage(content))
    return messages


Fetcher = Callable[[Indicator, list[str], date], list[Dataset]]


def _fetch_all(
    fetch: Fetcher, keys: list[str], countries: list[str], start: date
) -> tuple[list[Dataset], list[str]]:
    """Fetch the indicators at the same time, since a World Bank request can take a minute.
    Returns the datasets, and an error line for each indicator that failed."""

    def fetch_one(key: str) -> tuple[list[Dataset], str | None]:
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


def _drop_repeat(text: str) -> str:
    """The text once, when a provider sent the whole answer twice in a row."""
    stripped = text.strip()
    half = len(stripped) // 2
    first, second = stripped[:half].strip(), stripped[half:].strip()
    return first if first and first == second else text


def _require_text(message: AIMessage) -> AIMessage:
    if not message.text.strip():  # some free models end with nothing, or answer only in thinking
        raise ValueError("The model returned an empty answer.")
    return message


CATALOG_ANSWER = (
    "I answer questions with official data from the Banco Central do Brasil and the World Bank, "
    f"going back up to {MAX_YEARS_BACK} years.\n\n{catalog_markdown()}\n\n"
    "Ask for the latest value, how something changed (this year, over 12 months, since its low) "
    'or a comparison, e.g. "How has the Selic rate changed this year?" or '
    '"Compare GDP growth in Brazil, Argentina and Chile".'
)


def _route_after_plan(state: State) -> Literal["catalog", "fetch_data", "no_data"]:
    plan = state["plan"]
    if plan.about_catalog:
        return "catalog"
    return "fetch_data" if plan.indicators else "no_data"


def _catalog(state: State) -> dict:
    return {"answer": CATALOG_ANSWER}


def _no_data(state: State) -> dict:
    return {
        "answer": f"None of the indicators I can fetch match this question.\n\n{CATALOG_ANSWER}"
    }


def build_graph(llm: ChatModel, fetch: Fetcher = fetch_indicator):
    """Compile the agent graph. `llm` is a LangChain chat model with tool calling, or a chain
    of them built with `with_fallbacks`."""
    planner = _each_model(
        llm, lambda m: m.with_structured_output(Plan) | RunnableLambda(_require_plan)
    )
    writer = _each_model(llm, lambda m: m | RunnableLambda(_require_text))

    def plan(state: State) -> dict:
        today = date.today()
        system = PLANNER_PROMPT.format(
            today=today.isoformat(),
            catalog=catalog_text(),
            default_start=_default_start_year(),
            min_year=today.year - MAX_YEARS_BACK,
        )
        messages = [
            SystemMessage(system),
            *_history_messages(state),
            HumanMessage(state["question"]),
        ]
        result = planner.invoke(messages)
        return {"plan": result}

    def fetch_data(state: State) -> dict:
        plan = state["plan"]
        today = date.today()
        # From January of start_year, but within the 10 years the BCB serves for daily series
        start = max(date(plan.start_year, 1, 1), _years_ago(today, MAX_YEARS_BACK))
        datasets, errors = _fetch_all(fetch, plan.indicators, plan.countries, start)
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
        if state["plan"].yearly_table:
            message += (
                "\n\nA table of the `yearly` values is shown below your answer, so don't list "
                "them again (no table or list by year): sum up the trend in a few sentences."
            )
        messages = [SystemMessage(ANALYST_PROMPT), *_history_messages(state), HumanMessage(message)]
        response = writer.invoke(messages)
        return {
            "answer": _drop_repeat(response.text),
            "model": response.response_metadata.get("model_name", ""),
            "summaries": summaries,
        }

    def verify(state: State) -> dict:
        if not state.get("summaries"):  # no model answer to check (e.g. data failed to load)
            return {}
        return {"check": check_answer(state["answer"], state["summaries"], [state["question"]])}

    # LangGraph's type hints reject any TypedDict here, though it's the documented way
    builder = StateGraph(State)  # pyrefly: ignore[bad-specialization]
    builder.add_node("plan", plan)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("answer", answer)
    builder.add_node("verify", verify)
    builder.add_node("catalog", _catalog)
    builder.add_node("no_data", _no_data)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_after_plan)
    builder.add_edge("fetch_data", "answer")
    builder.add_edge("answer", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("catalog", END)
    builder.add_edge("no_data", END)
    return builder.compile()
