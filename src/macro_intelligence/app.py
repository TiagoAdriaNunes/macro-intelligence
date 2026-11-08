"""Chat interface built with shinychat.

Run it with `uv run macro-intelligence-app`, or `uv run shiny run src/macro_intelligence/app.py --reload`
while developing.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

from langchain_core.runnables import RunnableWithFallbacks
from shiny import App, reactive, ui
from shinychat import Chat, page_chat

# Absolute imports: `shiny run src/macro_intelligence/app.py` loads this file outside its package
from macro_intelligence.graph import ChatModel, build_graph
from macro_intelligence.llm import make_llm
from macro_intelligence.stream import AnswerStream, send
from macro_intelligence.tracing import setup_tracing

log = logging.getLogger(__name__)
setup_tracing()  # only when PHOENIX_TRACING=true

TITLE = "macro-intelligence"
MODELS_CACHE_SECONDS = 3600  # how long the chosen free models are reused before checking again
RETRY_SECONDS = 60  # how soon to check again when no free model could be chosen

SUGGESTIONS = [
    "How have Brazil's 12-month inflation and the Selic rate changed over the last 2 years?",
    "Compare GDP growth in Brazil, Argentina and Chile",
    "Is unemployment in Brazil falling?",
    "How has the US dollar moved against the real this year?",
]

GREETING = (
    "**Ask about Brazil's economy, or compare countries.** I look the figures up in official data "
    "(Banco Central do Brasil and the World Bank) and explain them, citing each source.\n\n"
    + "\n".join(f'- <span class="suggestion submit">{s}</span>' for s in SUGGESTIONS)
)

app_ui = page_chat(
    TITLE,
    id="chat",
    sidebar=False,
    greeting=GREETING,
    placeholder="Ask about inflation, interest rates, GDP, unemployment...",
    footer=ui.p(
        "Answers are written by free AI models from the data shown in each reply; check the sources.",
        class_="text-muted small text-center mb-0",
    ),
)


# The model chain (llm.make_llm) takes a few seconds to build: it checks OpenRouter's catalogue
# and each free model's uptime. Shared by all sessions and rebuilt every MODELS_CACHE_SECONDS.
@dataclass
class _CachedModels:
    llm: ChatModel | None = None
    expires: float = 0.0  # time.monotonic() after which it's rebuilt


_cache = _CachedModels()
_cache_lock = threading.Lock()


def cached_llm() -> ChatModel:
    with _cache_lock:
        llm = _cache.llm
        if llm is None or time.monotonic() > _cache.expires:
            llm = make_llm()
            # A single model is either OPENROUTER_MODEL (cheap to rebuild) or the fallback router
            # alone, when no free model could be chosen (e.g. the catalogue didn't load): retry soon
            ttl = MODELS_CACHE_SECONDS if isinstance(llm, RunnableWithFallbacks) else RETRY_SECONDS
            _cache.llm, _cache.expires = llm, time.monotonic() + ttl
        return llm


def server(input, output, session):
    chat = Chat("chat", history=False)
    turns: list[dict] = []  # earlier questions and answers, for follow-ups ("and Argentina?")

    # In the background, as chat.append_message_stream() does, so the session stays responsive;
    # a question sent while one is being answered waits its turn.
    @reactive.extended_task
    async def respond(question: str) -> None:
        try:
            llm = await asyncio.to_thread(cached_llm)  # network calls: keep the app responsive
        except RuntimeError as exc:  # e.g. no OPENROUTER_API_KEY
            await chat.append_message(f"**Setup problem:** {exc}")
            return

        reply = AnswerStream(build_graph(llm), question, list(turns))
        async with chat.message_stream_context() as message:
            await send(reply, message)
        if reply.answer:
            turns.append({"role": "user", "content": question})
            turns.append({"role": "assistant", "content": reply.answer})

    @chat.on_user_submit
    def _(user_input: str):
        respond.invoke(user_input)

    @reactive.effect
    def _report_errors():  # the agent's own failures are shown in the reply; this is the rest
        if error := respond.error():
            log.error("The reply failed", exc_info=error)
            ui.notification_show(f"The reply failed: {error}", type="error")


app = App(app_ui, server)


def main() -> None:
    from shiny import run_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Shiny sets run_app to None only when running in the browser (Pyodide)
    run_app("macro_intelligence.app:app", launch_browser=True)  # pyrefly: ignore[not-callable]
