"""Run the graph and turn its progress into Markdown chunks for a chat UI.

_Looking up: IPCA inflation (12-month), Selic target rate..._   <- after `plan`
<answer, streamed token by token>                                <- `answer` node
Sources, data check, model                                       <- after `verify`

When a model fails mid-answer, the next one starts over: its text replaces the partial one,
so the reply always shows the answer the data check verified.
"""

from collections.abc import AsyncIterable, AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from shiny.ui import HTML

from .analysis import Summary, source_lines
from .graph import Plan, Turn
from .grounding import Check, check_line
from .indicators import CATALOG
from .sources import Dataset
from .tables import yearly_table_html

FAILED = "**No model could answer right now.** The free models may be busy; please try again in a minute."


@dataclass(frozen=True)
class Replace:
    """Replace the whole reply shown so far with `text`."""

    text: str


@dataclass(frozen=True)
class Table:
    """A table to show as HTML (great_tables output), which the chat doesn't sanitize."""

    html: str


Chunk = str | Replace | Table  # Markdown to append, a replacement, or a table


class Message(Protocol):
    """A chat message being streamed, such as shinychat's `message_stream_context()`."""

    async def append(self, message_chunk: str | HTML) -> None: ...

    async def replace(self, message_chunk: str) -> None: ...


async def send(chunks: AsyncIterable[Chunk], message: Message) -> None:
    """Show a reply's chunks in a chat message."""
    async for chunk in chunks:
        if isinstance(chunk, Replace):
            await message.replace(chunk.text)
        elif isinstance(chunk, Table):
            await message.append(HTML(chunk.html))
        else:
            await message.append(chunk)


def looking_up(plan: Plan) -> str:
    names = [CATALOG[key].name for key in plan.indicators]
    world_bank = any(CATALOG[key].source == "worldbank" for key in plan.indicators)
    where = f" for {', '.join(plan.countries)}" if world_bank and plan.countries else ""
    return f"_Looking up: {', '.join(names)}{where}._\n\n"


def footer(datasets: list[Dataset], model: str, check: Check | None = None) -> str:
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
        self, graph: CompiledStateGraph, question: str, history: list[Turn] | None = None
    ) -> None:
        self.graph = graph
        self.question = question
        self.history = history or []
        self._reset()

    def _reset(self) -> None:
        self.answer: str | None = None
        self._lookup = ""  # the "Looking up" line shown before the answer
        self._streamed = ""  # answer text streamed so far
        self._run_id: str | None = None  # the model run that streamed it
        self._datasets: list[Dataset] = []
        self._summaries: list[Summary] = []  # the data the model was given
        self._yearly_table = False  # whether the plan asked for tables by year
        self._model = ""

    async def __aiter__(self) -> AsyncIterator[Chunk]:
        self._reset()  # each iteration runs the graph again
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
            logger.exception("Agent failed for question {!r}", self.question)
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
        if node == "plan":
            plan: Plan = update["plan"]
            self._yearly_table = plan.yearly_table
            if plan.indicators:
                self._lookup = looking_up(plan)
                yield self._lookup
        elif node == "fetch_data":
            self._datasets = update.get("datasets", [])
        elif node in ("answer", "catalog", "no_data"):
            answer = self.answer = update["answer"]
            self._model = update.get("model", "")
            self._summaries = update.get("summaries", [])
            if not self._streamed:  # a fixed message, or a model that didn't stream
                yield answer
            elif self._streamed != answer:  # show exactly the text the data check verifies
                yield Replace(self._lookup + answer)
        elif node == "verify":
            yield from self._tables()
            yield footer(self._datasets, self._model, update.get("check"))

    def _tables(self) -> Iterator[Table]:
        """A table by year for each series with more than one year, when the plan asks.
        A table that fails is left out, so the sources and data check still follow."""
        if not self._yearly_table:
            return
        for summary in self._summaries:
            if len(summary.get("yearly", [])) < 2:
                continue
            try:
                html = yearly_table_html(summary)
            except Exception:
                logger.exception("Table by year failed for {}", summary.get("name"))
                continue
            yield Table(html)
