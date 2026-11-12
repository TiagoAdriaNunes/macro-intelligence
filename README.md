# macro-intelligence

AI research assistant for macroeconomic questions, built with Python, Shiny, LangGraph and OpenRouter.

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

With `PHOENIX_TRACING=true` in `.env`, the chat app starts [Arize Phoenix](https://arize.com/docs/phoenix) automatically. Open <http://localhost:6006> to see each question's steps, prompts and model calls.

## Evaluation (optional)

`evals/questions.yaml` holds test questions about fixed past dates, each with the plan and figures expected. An experiment runs the agent on every question and scores it, and Phoenix keeps the results so you can compare models or code changes side by side. It all runs locally, apart from the OpenRouter calls.

```bash
uv run python -m macro_intelligence.tracing                    # start Phoenix, in another terminal
uv run python -m macro_intelligence.experiments                # run and score every question
uv run python -m macro_intelligence.experiments --dry-run 3    # try 3 questions, not saved
uv run python -m macro_intelligence.experiments --model free   # on the free models
uv run python -m macro_intelligence.experiments --refresh      # fetch the data again, not from evals/.cache
```

Open <http://localhost:6006>, then **Datasets**, to compare experiments. Each answer gets four scores:

| Score | Checks | How |
|---|---|---|
| `plan_match` | the planner picked the expected indicators, countries, start year and flags | code |
| `expected_figures` | the answer gives the correct figures for the question's date | code |
| `grounding` | every figure in the answer comes from the fetched data (`grounding.py`) | code |
| `answer_quality` | units, dates, sources, p.p. vs %, concision | LLM judge |

Experiments use `openai/gpt-4o-mini` (paid, but a full run costs well under a cent), or `OPENROUTER_MODEL` if set, or `--model`. One fixed model makes experiments comparable; the free models change over time and from question to question. The data each question needs is fetched once and kept in `evals/.cache`, so a slow World Bank or BCB moment doesn't count against the model. The judge uses `EVAL_JUDGE_MODEL` (or `--judge-model`) if set, else the same model; a different, stronger judge grades more fairly. `--no-judge` skips it.

## Data catalog

`catalog/contracts/` has a data contract for each indicator in the [Open Data Contract Standard](https://bitol-io.github.io/open-data-contract-standard/latest/home) (ODCS v3.1.0): source, fields, coverage and how fresh the data is. `catalog/ontology.yaml` lists the concepts the indicators measure (inflation, interest rate, exchange rate, unemployment, growth) and how they relate, and [`catalog/ontology.md`](catalog/ontology.md) draws them as a diagram. Both are generated from `indicators.py` and `ontology.py`; after changing those, run:

```bash
uv run python -m macro_intelligence.contracts
```

To browse the contracts as web pages, generate an HTML catalog in `catalog/html/` (not committed) and open `catalog/html/index.html`:

```bash
uvx --from datacontract-cli datacontract catalog --files="catalog/contracts/*.odcs.yaml" --output=catalog/html
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
