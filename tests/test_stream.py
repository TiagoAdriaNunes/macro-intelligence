"""Chat streaming (stream.AnswerStream) with fake LLMs: no network."""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from test_graph import FakeLLM, fake_dataset

from macro_intelligence.graph import Plan, build_graph
from macro_intelligence.stream import FAILED, AnswerStream, send


class FakeMessage:
    """A chat message, like shinychat's message_stream_context()."""

    def __init__(self):
        self.text = ""

    async def append(self, message_chunk: str) -> None:
        self.text += message_chunk

    async def replace(self, message_chunk: str) -> None:
        self.text = message_chunk


def show(stream: AnswerStream) -> str:
    """The reply as the chat shows it at the end, sent the way the app sends it."""
    message = FakeMessage()
    asyncio.run(send(stream, message))
    return message.text


def collect(graph, question, history=None) -> tuple[str, AnswerStream]:
    """The reply as shown, and the stream (for its `answer`)."""
    stream = AnswerStream(graph, question, history)
    return show(stream), stream


class StreamingLLM(BaseChatModel):
    """Streams its answer word by word, optionally failing after some words."""

    text: str
    fail_after: int | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-stream"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for i, word in enumerate(self.text.split(" ")):
            if i == self.fail_after:
                raise ConnectionError("the provider dropped the stream")
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                run_manager.on_llm_new_token(word + " ", chunk=chunk)
            yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return generate_from_stream(self._stream(messages, stop, run_manager, **kwargs))

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: Plan(indicators=["br_ipca_12m"]))


def test_stream_shows_lookup_answer_sources_and_model():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m", "wb_gdp_growth"], countries=["BRA", "USA"]))
    graph = build_graph(llm, fetch=lambda ind, countries, start: [fake_dataset(ind)])

    text, stream = collect(graph, "Inflation and growth?")

    assert text.startswith("_Looking up: IPCA inflation (12-month), GDP growth for BRA, USA._")
    assert text.count(llm.answer) == 1  # streamed or not, the answer appears once
    assert "**Sources**" in text and "series 13522" in text
    assert "Model: `fake/model`" in text
    assert stream.answer == llm.answer  # without the lookup line and footer


def test_stream_passes_history_to_the_models():
    llm = FakeLLM(plan=Plan(indicators=["br_ipca_12m"]))
    graph = build_graph(llm, fetch=lambda ind, countries, start: [fake_dataset(ind)])
    history = [
        {"role": "user", "content": "What is inflation in Brazil?"},
        {"role": "assistant", "content": "It was 4.58%."},
    ]

    collect(graph, "And a year ago?", history)

    contents = [m.content for m in llm.prompts[0]]
    assert contents[1:3] == ["What is inflation in Brazil?", "It was 4.58%."]


def test_stream_unmatched_question_has_no_lookup_line():
    graph = build_graph(FakeLLM(plan=Plan(indicators=[])), fetch=lambda *a: [])
    text, _ = collect(graph, "Best pizza?")
    assert text.startswith("None of the indicators")
    assert "Looking up" not in text


def test_stream_reports_failure_instead_of_raising():
    graph = build_graph(FakeLLM(error="503 Service Unavailable"), fetch=lambda *a: [])
    text, stream = collect(graph, "Selic?")
    assert text == FAILED
    assert stream.answer is None


def test_a_model_failing_mid_answer_leaves_no_partial_text():
    first = StreamingLLM(text="Inflation hit 9.99% and", fail_after=3)
    second = StreamingLLM(text="Inflation was 4.58% in September 2026.")
    graph = build_graph(first.with_fallbacks([second]), fetch=lambda ind, c, s: [fake_dataset(ind)])

    text, stream = collect(graph, "Inflation?")

    assert "9.99" not in text
    assert text.count("Inflation was 4.58% in September 2026.") == 1
    assert "✓ Data check" in text
    assert stream.answer.strip() == "Inflation was 4.58% in September 2026."


def test_partial_text_is_dropped_when_every_model_fails():
    failing = StreamingLLM(text="Inflation hit 9.99% and", fail_after=3)
    graph = build_graph(failing, fetch=lambda ind, c, s: [fake_dataset(ind)])

    text, _ = collect(graph, "Inflation?")

    assert text == "_Looking up: IPCA inflation (12-month)._\n\n" + FAILED


def test_a_stream_can_be_shown_twice():
    graph = build_graph(
        FakeLLM(plan=Plan(indicators=["br_ipca_12m"])), fetch=lambda ind, c, s: [fake_dataset(ind)]
    )
    stream = AnswerStream(graph, "Inflation?")
    assert show(stream) == show(stream)
