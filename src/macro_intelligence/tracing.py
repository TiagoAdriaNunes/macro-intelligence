"""Optional tracing with Arize Phoenix: free, open source, runs on your machine.

1. Turn tracing on:   PHOENIX_TRACING=true in .env
2. Start the app:     uv run macro-intelligence-app   (it starts Phoenix too, UI at http://localhost:6006)
   For the CLI, start Phoenix first: uv run python -m macro_intelligence.tracing
3. Ask questions: each one appears in Phoenix as a trace, with every graph node, prompt,
   model call and fallback.

Phoenix and its LangChain instrumentation are dev dependencies; tracing stays off without them.
"""

import atexit
import functools
import importlib.util
import os
import socket
import subprocess
import sys
import time

from dotenv import load_dotenv
from loguru import logger

PROJECT_NAME = "macro-intelligence"
PHOENIX_PORTS = {6006: "web UI and HTTP traces", 4317: "gRPC traces"}
TRACES_PORT = 4317  # where traces go when PHOENIX_COLLECTOR_ENDPOINT isn't set
STARTUP_SECONDS = 60  # how long start_phoenix() waits for the server


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _tracing_on() -> bool:
    load_dotenv()
    return os.getenv("PHOENIX_TRACING", "").strip().lower() in ("1", "true", "yes")


def _local_phoenix_down() -> bool:
    """Traces go to a local Phoenix (no PHOENIX_COLLECTOR_ENDPOINT) that isn't running."""
    return not os.getenv("PHOENIX_COLLECTOR_ENDPOINT") and not _port_in_use(TRACES_PORT)


def start_phoenix() -> None:
    """Start a local Phoenix for as long as this process runs, when tracing is on and Phoenix
    isn't running yet. Call it before setup_tracing()."""
    if not _tracing_on() or not _local_phoenix_down() or not importlib.util.find_spec("phoenix"):
        return  # setup_tracing() explains a missing Phoenix
    logger.info("Starting Phoenix (UI at http://localhost:6006)")
    server = subprocess.Popen(
        [sys.executable, "-m", "macro_intelligence.tracing"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(server.terminate)
    deadline = time.monotonic() + STARTUP_SECONDS
    while not _port_in_use(TRACES_PORT):
        if server.poll() is not None or time.monotonic() > deadline:
            logger.warning(
                "Phoenix didn't start; run `uv run python -m macro_intelligence.tracing` to see why."
            )
            return
        time.sleep(0.5)


@functools.cache  # the first call decides; later calls return the same answer
def setup_tracing() -> bool:
    """Send LangChain/LangGraph traces to Phoenix when PHOENIX_TRACING is true."""
    if not _tracing_on():
        return False
    # Without a running Phoenix, every batch of traces would log a connection error
    if _local_phoenix_down():
        logger.warning(
            "PHOENIX_TRACING is on but Phoenix isn't running, so tracing is off. Start it with "
            "`uv run python -m macro_intelligence.tracing`, then restart."
        )
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        logger.warning(
            "PHOENIX_TRACING is on but Phoenix isn't installed; run `uv sync` to add dev dependencies."
        )
        return False

    # Endpoint: PHOENIX_COLLECTOR_ENDPOINT, else the local server's gRPC port (TRACES_PORT).
    # Batched so tracing never slows answers; pending spans are sent when the process exits.
    provider = register(project_name=PROJECT_NAME, batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    logger.info("Tracing to Phoenix, project {!r}", PROJECT_NAME)
    return True


def serve() -> None:
    """Start the Phoenix server, with its telemetry and internet downloads off."""
    busy = [f"{port} ({use})" for port, use in PHOENIX_PORTS.items() if _port_in_use(port)]
    if busy:
        sys.exit(
            f"Port {', '.join(busy)} is already in use. If Phoenix is already running, open "
            "http://localhost:6006; otherwise stop the program using the port and try again."
        )
    # Telemetry: the FullStory and Scarf pixels in the UI. External resources: web fonts and
    # sandbox downloads.
    os.environ.setdefault("PHOENIX_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("PHOENIX_ALLOW_EXTERNAL_RESOURCES", "false")
    from phoenix.server.main import main

    sys.argv = ["phoenix", "serve"]
    main()


if __name__ == "__main__":
    serve()
