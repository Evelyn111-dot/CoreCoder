import pytest
from pydantic import ValidationError

from corecoder.backend.settings import Settings


def test_settings_load_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_NAME", "Test Agent API")
    monkeypatch.setenv("ALI_API_KEY", "test-key")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setenv("RAG_MAX_TOP_K", "8")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.4")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "20")
    monkeypatch.setenv("AGENT_MAX_CONTEXT_TOKENS", "64000")

    settings = Settings.from_env()

    assert settings.app_name == "Test Agent API"
    assert settings.api_key == "test-key"
    assert settings.knowledge_dir == tmp_path
    assert settings.rag_top_k == 4
    assert settings.rag_max_top_k == 8
    assert settings.rag_score_threshold == 0.4
    assert settings.agent_max_rounds == 20
    assert settings.agent_max_context_tokens == 64_000


def test_settings_reject_top_k_above_maximum():
    with pytest.raises(
        ValidationError,
        match="RAG_TOP_K 不能大于 RAG_MAX_TOP_K",
    ):
        Settings(
            rag_top_k=11,
            rag_max_top_k=10,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rag_top_k", 0),
        ("rag_max_top_k", 0),
        ("rag_score_threshold", 1.1),
        ("agent_max_rounds", 0),
        ("agent_max_context_tokens", 999),
    ],
)
def test_settings_reject_invalid_ranges(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})
