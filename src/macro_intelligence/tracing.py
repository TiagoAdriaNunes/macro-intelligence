"""Optional tracing with Arize Phoenix: free, open source, runs on your machine.

1. Start Phoenix:          uv run python -m macro_intelligence.tracing   (UI at http://localhost:6006)
2. Turn tracing on:        PHOENIX_TRACING=true in .env
3. Ask questions in the app or CLI: each one appears in Phoenix as a trace, with every graph
   node, prompt, model call and fallback.

Phoenix and its LangChain instrumentation are dev dependencies; tracing stays off without them.
"""

import logging
import os
import socket
import sys

from dotenv import load_dotenv

log = logging.getLogger(__name__)

PROJECT_NAME = "macro-intelligence"
_enabled = False


def setup_tracing() -> bool:
    """Send LangChain/LangGraph traces to Phoenix when PHOENIX_TRACING is true. Safe to call twice."""
    global _enabled
    load_dotenv()
    if _enabled or os.getenv("PHOENIX_TRACING", "").strip().lower() not in ("1", "true", "yes"):
        return _enabled
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        log.warning(
            "PHOENIX_TRACING is on but Phoenix isn't installed; run `uv sync` to add dev dependencies."
        )
        return False

    # Endpoint: PHOENIX_COLLECTOR_ENDPOINT, else the local server at http://localhost:6006.
    # Batched so tracing never slows answers; pending spans are sent when the process exits.
    provider = register(project_name=PROJECT_NAME, batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _enabled = True
    log.info("Tracing to Phoenix, project %r", PROJECT_NAME)
    return True


PHOENIX_PORTS = {6006: "web UI and HTTP traces", 4317: "gRPC traces"}


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


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
