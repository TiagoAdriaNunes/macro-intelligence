"""Logging setup (logs.py): loguru, with standard logging sent through it."""

import logging
import sys

import pytest
from loguru import logger

from macro_intelligence.logs import setup_logging


@pytest.fixture
def restore_logging():
    """Undo setup_logging's global changes, so other tests log as before."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    logger.remove()
    logger.add(sys.stderr)


@pytest.mark.usefixtures("restore_logging")
def test_levels_and_standard_logging_go_through_loguru(capsys):
    setup_logging("WARNING")

    logger.info("not shown below WARNING")
    logger.warning("shown from this package")
    logging.getLogger("some.library").warning("shown from a library")

    err = capsys.readouterr().err
    assert "not shown below WARNING" not in err
    assert "shown from this package" in err
    assert "shown from a library" in err and "WARNING" in err
