from macro_intelligence import tracing


def test_tracing_is_off_unless_enabled(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "false")  # load_dotenv doesn't override set variables
    monkeypatch.setattr(tracing, "_enabled", False)
    assert tracing.setup_tracing() is False


def test_tracing_stays_off_when_phoenix_is_not_running(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "true")
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "load_dotenv", lambda: None)  # ignore a local .env
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_port_in_use", lambda port: False)
    assert tracing.setup_tracing() is False
