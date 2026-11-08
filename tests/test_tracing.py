import pytest

from macro_intelligence import tracing


@pytest.fixture(autouse=True)
def tracing_env(monkeypatch):
    """No local .env, no remote endpoint, and setup_tracing() deciding afresh."""
    monkeypatch.setattr(tracing, "load_dotenv", lambda: None)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    tracing.setup_tracing.cache_clear()
    yield
    tracing.setup_tracing.cache_clear()


def test_tracing_is_off_unless_enabled(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "false")
    assert tracing.setup_tracing() is False


def test_tracing_stays_off_when_phoenix_is_not_running(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "true")
    monkeypatch.setattr(tracing, "_port_in_use", lambda port: False)
    assert tracing.setup_tracing() is False


class FakeServer:
    def __init__(self, *args, **kwargs):
        self.args = args

    def poll(self):
        return None  # still running

    def terminate(self):
        pass


def test_start_phoenix_starts_the_server_and_waits_for_it(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "true")
    ports = iter([False, False, True])  # down at the check, then up after one wait
    monkeypatch.setattr(tracing, "_port_in_use", lambda port: next(ports))
    started, stoppers = [], []
    monkeypatch.setattr(
        tracing.subprocess, "Popen", lambda *a, **k: started.append(a) or FakeServer()
    )
    monkeypatch.setattr(tracing.atexit, "register", stoppers.append)
    monkeypatch.setattr(tracing.time, "sleep", lambda s: None)

    tracing.start_phoenix()

    assert started[0][0][-2:] == ["-m", "macro_intelligence.tracing"]
    assert len(stoppers) == 1  # stopped when the app exits


def test_start_phoenix_does_nothing_when_off_or_running(monkeypatch):
    monkeypatch.setattr(tracing.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setenv("PHOENIX_TRACING", "false")
    tracing.start_phoenix()
    monkeypatch.setenv("PHOENIX_TRACING", "true")
    monkeypatch.setattr(tracing, "_port_in_use", lambda port: True)  # already running
    tracing.start_phoenix()
