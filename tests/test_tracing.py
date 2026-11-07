from macro_intelligence import tracing


def test_tracing_is_off_unless_enabled(monkeypatch):
    monkeypatch.setenv("PHOENIX_TRACING", "false")  # load_dotenv doesn't override set variables
    monkeypatch.setattr(tracing, "_enabled", False)
    assert tracing.setup_tracing() is False
