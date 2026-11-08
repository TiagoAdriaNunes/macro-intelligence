"""Run the graph and turn its progress into Markdown chunks for a chat UI.

_Looking up: IPCA inflation (12-month), Selic target rate..._   <- after `plan`
<answer, streamed token by token>                                <- `answer` node
Sources, data check, model                                       <- after `verify`

When a model fails mid-answer, the next one starts over: its text replaces the partial one,
so the reply always shows the answer the data check verified.
"""

import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from .analysis import source_lines
from .grounding import check_line
from .indicators import CATALOG

log = logging.getLogger(__name__)

FAILED = "**No model could answer right now.** The free models may be busy; please try again in a minute."


@dataclass(frozen=True)
class Replace:
    """Replace the whole reply shown so far with `text`."""

    text: str


Chunk = str | Replace  # text to append, or a replacement


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


class AnswerStream:
    """The agent's reply to `question`, given earlier `history` turns, as Markdown chunks:
    `async for chunk in AnswerStream(graph, question)`.

    Once the stream ends, `answer` holds the answer alone, without the lookup line and footer,
    to pass as history with later questions. It stays None when no answer was written.
    """

    def __init__(
        self, graph: CompiledStateGraph, question: str, history: list[dict] | None = None
    ) -> None:
        self.graph = graph
        self.question = question
        self.history = history or []
        self.answer: str | None = None
        self._lookup = ""  # the "Looking up" line shown before the answer
        self._streamed = ""  # answer text streamed so far
        self._run_id: str | None = None  # the model run that streamed it
        self._datasets: list[dict] = []
        self._model = ""

    async def __aiter__(self) -> AsyncIterator[Chunk]:
        inputs = {"question": self.question, "history": self.history}
        try:
            async for mode, payload in self.graph.astream(
                inputs, stream_mode=["updates", "messages"]
            ):
                if mode == "messages" and isinstance(payload, tuple):
                    for chunk in self._on_token(*payload):
                        yield chunk
                elif isinstance(payload, dict):  # "updates": {node: its state update}
                    for node, update in payload.items():
                        for chunk in self._on_update(node, update or {}):
                            yield chunk
        except Exception:
            log.exception("Agent failed for question %r", self.question)
            if self._streamed and self.answer is None:  # drop a failed model's partial text
                yield Replace(self._lookup + FAILED)
            else:
                yield ("\n\n" if self.answer else "") + FAILED

    def _on_token(self, message: BaseMessage, meta: dict[str, Any]) -> Iterator[Chunk]:
        text = str(message.text)
        if meta.get("langgraph_node") != "answer" or not text:
            return
        if self._streamed and message.id != self._run_id:  # a model failed mid-answer
            self._streamed = ""
            yield Replace(self._lookup)
        self._run_id = message.id
        self._streamed += text
        yield text

    def _on_update(self, node: str, update: dict[str, Any]) -> Iterator[Chunk]:
        if node == "plan" and update["plan"]["indicators"]:
            self._lookup = looking_up(update["plan"])
            yield self._lookup
        elif node == "fetch_data":
            self._datasets = update.get("datasets", [])
        elif node in ("answer", "no_data"):
            answer = self.answer = update["answer"]
            self._model = update.get("model", "")
            if not self._streamed:  # a fixed message, or a model that didn't stream
                yield answer
            elif self._streamed != answer:  # show exactly the text the data check verifies
                yield Replace(self._lookup + answer)
        elif node == "verify":
            yield footer(self._datasets, self._model, update.get("check"))
