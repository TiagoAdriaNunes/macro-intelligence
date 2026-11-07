"""Run the graph and turn its progress into Markdown chunks for a chat UI.

_Looking up: IPCA inflation (12-month), Selic target rate..._   <- after `plan`
<answer, streamed token by token>                                <- `answer` node
Sources, data check, model                                       <- after `verify`
"""

import logging
from collections.abc import AsyncIterator

from langgraph.graph.state import CompiledStateGraph

from .analysis import source_lines
from .grounding import check_line
from .indicators import CATALOG

log = logging.getLogger(__name__)

FAILED = "**No model could answer right now.** The free models may be busy; please try again in a minute."


def looking_up(plan: dict) -> str:
    names = [CATALOG[key].name for key in plan.get("indicators", [])]
    countries = plan.get("countries", [])
    world_bank = any(CATALOG[key].source == "worldbank" for key in plan.get("indicators", []))
    where = f" for {', '.join(countries)}" if world_bank and countries else ""
    return f"_Looking up: {', '.join(names)}{where}._\n\n"


def footer(datasets: list[dict], model: str, check: dict | None = None) -> str:
    lines = source_lines(datasets)
    parts = []
    if lines:
        parts.append("**Sources**\n\n" + "\n".join(f"- {line}" for line in lines))
    if line := check_line(check):
        parts.append(line)
    if model:
        parts.append(f"<small>Model: `{model}`</small>")
    return "\n\n---\n\n" + "\n\n".join(parts) if parts else ""


async def answer_stream(
    graph: CompiledStateGraph, question: str, history: list[dict] | None = None
) -> AsyncIterator[str]:
    """Markdown chunks of the agent's reply to `question`, given earlier `history` turns."""
    streamed = False  # whether the answer arrived token by token
    datasets: list[dict] = []
    model = ""
    try:
        async for mode, payload in graph.astream(
            {"question": question, "history": history or []}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = payload
                text = getattr(chunk, "text", "")
                if meta.get("langgraph_node") == "answer" and isinstance(text, str) and text:
                    streamed = True
                    yield text
                continue

            for node, update in payload.items():
                update = update or {}
                if node == "plan" and update["plan"]["indicators"]:
                    yield looking_up(update["plan"])
                elif node == "fetch_data":
                    datasets = update.get("datasets", [])
                elif node in ("answer", "no_data"):
                    if not streamed:  # deterministic message, or a model that didn't stream
                        yield update["answer"]
                    model = update.get("model", "")
                elif node == "verify":
                    yield footer(datasets, model, update.get("check"))
    except Exception:
        log.exception("Agent failed for question %r", question)
        yield ("\n\n" if streamed else "") + FAILED
