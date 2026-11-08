# macro-intelligence

AI research assistant for macroeconomic questions, built with Python, LangGraph and OpenRouter.

Ask a question in plain language. The agent picks the indicators it needs, fetches them from official sources (Banco Central do Brasil and the World Bank), and has an LLM explain the figures using only that data, with sources cited. Answers written from data end with a check that flags any figure not found in it.

Data available: Brazilian inflation (IPCA), Selic rate, USD/BRL and unemployment; GDP growth, inflation and unemployment for any country.

## Install

Requires [uv](https://docs.astral.sh/uv/) and an OpenRouter API key.

```bash
uv sync
cp .env.example .env
```

### OpenRouter API key

1. Create an account at [openrouter.ai](https://openrouter.ai), then a key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. Paste it into `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

The default free models need no credits. If the app skips some as unusable with your account, allow them in your [privacy settings](https://openrouter.ai/settings/privacy).

## Run

```bash
uv run macro-intelligence-app                                    # chat app in the browser
uv run macro-intelligence "Is unemployment in Brazil falling?"   # command line
```

## Tracing (optional)

To see each step, prompt and model call in [Arize Phoenix](https://arize.com/docs/phoenix), set `PHOENIX_TRACING=true` in `.env`. The chat app then starts Phoenix too (UI at <http://localhost:6006>). For the command line, start it first:

```bash
uv run python -m macro_intelligence.tracing
```

## Deploy to Fly.io

Requires [`flyctl`](https://fly.io/docs/flyctl/install/) and a Fly.io account.

```bash
flyctl auth login
flyctl apps create <your-app-name>   # then set app = "<your-app-name>" in fly.toml
flyctl secrets set OPENROUTER_API_KEY=<your key>
flyctl deploy --ha=false
```

`--ha=false` keeps a single machine, which Shiny needs because each chat session lives on one machine. The machine stops when idle and starts on the next visit. There's no login: anyone with the URL uses your OpenRouter key.

## Development

```bash
uv run pytest           # tests (offline, no API key needed)
uv run ruff format      # format
uv run ruff check       # lint
uv run pyrefly check    # type check
```

## License

MIT
