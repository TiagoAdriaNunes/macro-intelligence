"""Run the agent on evals/questions.yaml and score its answers, as a Phoenix experiment.

1. Start Phoenix:        uv run python -m macro_intelligence.tracing
2. Run an experiment:    uv run python -m macro_intelligence.experiments
3. Compare experiments at http://localhost:6006, under Datasets.

The questions ask about fixed past dates, so the data doesn't change and each question can list
the correct figures to check. The data fetched is kept in evals/.cache, so later experiments
don't wait on (or time out on) the sources; --refresh fetches it again.

Everything stays on your machine, apart from the OpenRouter calls the agent and judge make.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from .evaluators import expected_figures, grounding, make_answer_quality, plan_match
from .graph import Fetcher, build_graph
from .llm import make_llm
from .logs import setup_logging
from .sources import fetch_indicator
from .tracing import _port_in_use

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"
QUESTIONS = EVALS_DIR / "questions.yaml"
CACHE_DIR = EVALS_DIR / ".cache"
DATASET_PREFIX = "macro-questions"
EXPECTED_KEYS = (
    "indicators",
    "countries",
    "start_year",
    "about_catalog",
    "yearly_table",
    "figures",
)
PHOENIX_UI_PORT = 6006
TASK_SECONDS = 300  # free models can take minutes when they fall back
DEFAULT_MODEL = "openai/gpt-4o-mini"  # cheap, paid, and the same on every run
FREE = "free"  # --model value for the free models (see models.py)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    """Examples as {"input": {question, history}, "output": expected plan fields and figures}."""
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {
            "input": {"question": item["question"], "history": item.get("history", [])},
            "output": {k: item[k] for k in EXPECTED_KEYS if k in item},
        }
        for item in items
    ]


def dataset_name(examples: list[dict]) -> str:
    """Name of the Phoenix dataset for these examples. Editing the questions makes a new dataset,
    so the experiments compared side by side always ran on the same questions."""
    digest = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:8]
    return f"{DATASET_PREFIX}-{digest}"


def make_task(graph):
    """The experiment task: run the agent on one example and keep what the evaluators need."""

    def task(input: dict) -> dict:
        state = graph.invoke({"question": input["question"], "history": input.get("history", [])})
        plan = state.get("plan")
        return {
            "answer": state.get("answer", ""),
            "model": state.get("model", ""),
            "plan": plan.model_dump() if plan else None,
            "check": state.get("check"),
            "errors": state.get("errors", []),
            "summaries": state.get("summaries", []),
        }

    return task


def cached(fetch: Fetcher, directory: Path = CACHE_DIR) -> Fetcher:
    """`fetch`, keeping each result on disk: the questions ask about past dates, so later
    experiments reuse the data instead of waiting on (or timing out on) the sources. Failures
    aren't kept, so the next experiment tries them again."""

    def fetch_cached(ind, countries, start):
        where = ind.country or "-".join(sorted(countries))
        path = directory / f"{ind.key}_{where}_{start.isoformat()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        datasets = fetch(ind, countries, start)  # fetch_indicator retries failures itself
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(datasets, ensure_ascii=False), encoding="utf-8")
        return datasets

    return fetch_cached


def _commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _llm(model: str):
    if model == FREE:
        os.environ.pop("OPENROUTER_MODEL", None)  # else make_llm() would use it
        return make_llm()
    return make_llm(model)


def _dataset(client, examples: list[dict]):
    name = dataset_name(examples)
    try:
        return client.datasets.get_dataset(dataset=name)
    except ValueError:  # not created yet
        logger.info("Creating dataset {}", name)
        return client.datasets.create_dataset(
            name=name,
            inputs=[e["input"] for e in examples],
            outputs=[e["output"] for e in examples],
            dataset_description=f"{len(examples)} questions from evals/questions.yaml",
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="python -m macro_intelligence.experiments",
        description="Score the agent on evals/questions.yaml, as a Phoenix experiment.",
    )
    parser.add_argument(
        "--model",
        help=f"OpenRouter model id, or {FREE!r} for the free models. "
        f"Default: OPENROUTER_MODEL, else {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--judge-model",
        help="Model for the answer_quality judge. Default: EVAL_JUDGE_MODEL, else as --model.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--name", help="Experiment name. Default: the model id.")
    parser.add_argument(
        "--dry-run",
        type=int,
        metavar="N",
        help="Run N random questions without saving the experiment.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Fetch the data again instead of using the cache."
    )
    args = parser.parse_args()

    load_dotenv()
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    logging.getLogger("alembic").setLevel(logging.WARNING)  # Phoenix sets up its database on import
    examples = load_questions()
    if args.refresh:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    if not os.getenv("PHOENIX_BASE_URL") and not _port_in_use(PHOENIX_UI_PORT):
        sys.exit(
            "Phoenix isn't running. Start it with `uv run python -m macro_intelligence.tracing`, "
            "then try again."
        )

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.client import Client

    # The experiment sets its own tracer provider while each task and evaluator runs, so the
    # model calls show up under each example
    LangChainInstrumentor().instrument()

    model = args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
    judge_model = args.judge_model or os.getenv("EVAL_JUDGE_MODEL") or model
    if model == FREE:
        logger.warning(
            "The free models change over time and fall back per question, so this experiment "
            "won't compare well with others."
        )
    if not args.no_judge and judge_model == model:
        logger.warning(
            "The judge is the agent's model; set EVAL_JUDGE_MODEL to grade with another."
        )
    try:
        graph = build_graph(_llm(model), fetch=cached(fetch_indicator))
        evaluators: dict[str, Callable[..., dict]] = {
            "plan_match": plan_match,
            "expected_figures": expected_figures,
            "grounding": grounding,
        }
        if not args.no_judge:
            evaluators["answer_quality"] = make_answer_quality(_llm(judge_model))
    except RuntimeError as exc:  # no API key
        sys.exit(str(exc))

    client = Client()
    ran = client.experiments.run_experiment(
        dataset=_dataset(client, examples),
        task=make_task(graph),
        evaluators=evaluators,
        experiment_name=args.name or model,
        experiment_description=f"{len(examples)} questions, commit {_commit() or '?'}",
        experiment_metadata={
            "model": model,
            "judge": None if args.no_judge else judge_model,
            "commit": _commit(),
        },
        dry_run=args.dry_run or False,
        timeout=TASK_SECONDS,
        retries=1,  # a failed task already went through every fallback model
    )
    if not args.dry_run:
        url = client.experiments.get_experiment_url(
            dataset_id=ran["dataset_id"], experiment_id=ran["experiment_id"]
        )
        print(f"\nResults: {url}")


if __name__ == "__main__":
    main()
