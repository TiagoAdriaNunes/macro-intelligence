"""Chat interface built with shinychat.

Run it with `uv run macro-intelligence-app`, or `uv run shiny run src/macro_intelligence/app.py --reload`
while developing.
"""

import asyncio
import logging
import threading
import time

from shiny import App, ui
from shinychat import Chat, page_chat

# Absolute imports: `shiny run src/macro_intelligence/app.py` loads this file outside its package
from macro_intelligence.graph import build_graph
from macro_intelligence.llm import make_llm
from macro_intelligence.stream import answer_stream
from macro_intelligence.tracing import setup_tracing

log = logging.getLogger(__name__)
setup_tracing()  # only when PHOENIX_TRACING=true

TITLE = "macro-intelligence"
MODELS_CACHE_SECONDS = 3600  # how long the chosen free models are reused before checking again

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
_llm_lock = threading.Lock()
_llm_cache: dict = {"llm": None, "built_at": 0.0}


def cached_llm():
    with _llm_lock:
        if (
            _llm_cache["llm"] is None
            or time.monotonic() - _llm_cache["built_at"] > MODELS_CACHE_SECONDS
        ):
            _llm_cache["llm"] = make_llm()
            _llm_cache["built_at"] = time.monotonic()
        return _llm_cache["llm"]


def server(input, output, session):
    chat = Chat("chat", history=False)

    @chat.on_user_submit
    async def _(user_input: str):
        # Earlier turns for follow-up questions; the last message is the one just sent
        history = [
            {"role": m["role"], "content": str(m["content"])}
            for m in chat.messages()[:-1]
            if m["role"] in ("user", "assistant")
        ]
        try:
            llm = await asyncio.to_thread(cached_llm)  # network calls: keep the app responsive
        except RuntimeError as exc:  # e.g. no OPENROUTER_API_KEY
            await chat.append_message(f"**Setup problem:** {exc}")
            return
        await chat.append_message_stream(answer_stream(build_graph(llm), user_input, history))


app = App(app_ui, server)


def main() -> None:
    from shiny import run_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_app("macro_intelligence.app:app", launch_browser=True)
