"""Command line entry point: `macro-intelligence "your question"`."""

import argparse
import json
import logging
import sys

from .analysis import source_lines
from .graph import build_graph
from .grounding import check_line
from .llm import make_llm
from .tracing import setup_tracing


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="macro-intelligence", description="Ask questions about economic data."
    )
    parser.add_argument("question", nargs="+", help="Question in plain language.")
    parser.add_argument(
        "--model",
        help="OpenRouter model id. Default: OPENROUTER_MODEL, or the best free models with fallbacks.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show the models tried and the data plan."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP request otherwise
    setup_tracing()

    try:
        llm = make_llm(args.model)
    except RuntimeError as exc:
        sys.exit(str(exc))

    try:
        result = build_graph(llm).invoke({"question": " ".join(args.question)})
    except Exception as exc:
        sys.exit(f"No model could answer: {exc}")

    if args.verbose and "plan" in result:
        print("Plan:", json.dumps(result["plan"]), end="\n\n")
    print(result["answer"])

    sources = source_lines(result.get("datasets", []))
    if sources:
        print("\nSources:")
        for line in sources:
            print(f"- {line}")
    if check := check_line(result.get("check")):
        print(f"\n{check}")
    if result.get("model"):
        print(f"\nModel: {result['model']}")


if __name__ == "__main__":
    main()
