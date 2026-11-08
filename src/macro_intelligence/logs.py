"""Logging with loguru: one format for this package and for the libraries it uses, which log
with the standard logging module (httpx, LangChain, OpenTelemetry...).

Call setup_logging() once at startup (the app and the CLI do). Elsewhere, use loguru directly:
`from loguru import logger`, with `{}` placeholders: logger.info("Models: {}", names).
"""

import inspect
import logging
import sys

from loguru import logger

FORMAT = "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> <cyan>{name}</cyan>: {message}"


class InterceptHandler(logging.Handler):
    """Send standard logging records to loguru (the recipe from loguru's documentation)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Find the code that logged, past the logging module's own frames
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO") -> None:
    """Log to stderr from `level` up (DEBUG, INFO, WARNING...), standard logging included."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=FORMAT)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP request otherwise
