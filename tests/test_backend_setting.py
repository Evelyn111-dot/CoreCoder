import pytest
from pydantic import ValidationError

from corecoder.backend.settings import Settings


def test_settings_load_from_environment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "APP_NAME",
        "Test Agent API",
    )

    monkeypatch.setenv(
        "MYSQL_HOST",
        "localhost",
    )

    monkeypatch.setenv(
        "MYSQL_PORT",
        "3307",
    )

    monkeypatch.setenv(
        "MYSQL_USER",
        "test-user",
    )

    monkeypatch.setenv(
        "MYSQL_PASSWORD",
        "test-password",
    )

    monkeypatch.setenv(
        "MYSQL_DATABASE",
        "test-corecoder",
    )

    monkeypatch.setenv(
        "ALI_API_KEY",
        "test-key",
    )

    monkeypatch.setenv(
        "KNOWLEDGE_DIR",
        str(tmp_path),
    )

    monkeypatch.setenv(
        "RAG_TOP_K",
        "4",
    )

    monkeypatch.setenv(
        "RAG_MAX_TOP_K",
        "8",
    )

    monkeypatch.setenv(
        "RAG_SCORE_THRESHOLD",
        "0.4",
    )

    monkeypatch.setenv(
        "AGENT_MAX_ROUNDS",
        "20",
    )

    monkeypatch.setenv(
        "AGENT_MAX_CONTEXT_TOKENS",
        "64000",
    )

    settings = Settings.from_env()

    assert settings.app_name == "Test Agent API"
    assert settings.mysql_host == "localhost"
    assert settings.mysql_port == 3307
    assert settings.mysql_user == "test-user"

    assert (
        settings.mysql_password
        == "test-password"
    )

    assert (
        settings.mysql_database
        == "test-corecoder"
    )

    assert settings.api_key == "test-key"
    assert settings.knowledge_dir == tmp_path
    assert settings.rag_top_k == 4
    assert settings.rag_max_top_k == 8

    assert (
        settings.rag_score_threshold
        == 0.4
    )

    assert settings.agent_max_rounds == 20

    assert (
        settings.agent_max_context_tokens
        == 64_000
    )


def test_settings_reject_top_k_above_maximum():
    with pytest.raises(
        ValidationError,
        match=(
            "RAG_TOP_K 不能大于 "
            "RAG_MAX_TOP_K"
        ),
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
        (
            "agent_max_context_tokens",
            999,
        ),
    ],
)
def test_settings_reject_invalid_ranges(
    field,
    value,
):
    with pytest.raises(ValidationError):
        Settings(
            **{
                field: value,
            }
        )