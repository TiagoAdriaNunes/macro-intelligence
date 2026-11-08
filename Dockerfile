FROM python:3.11-slim

# uv installs the exact versions in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock changes.
# --no-dev leaves out Phoenix tracing and the test tools.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# OPENROUTER_API_KEY is provided at runtime (fly secrets set), never baked into the image
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8080 LOG_LEVEL=INFO
EXPOSE 8080

CMD ["macro-intelligence-app"]
