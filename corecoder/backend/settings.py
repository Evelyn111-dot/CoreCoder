from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """后端配置。

    配置优先从环境变量读取，其次读取项目根目录的 .env。
    字段在应用启动时完成类型和范围校验，避免错误配置进入运行阶段。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(
        default="CoreCoder RAG API",
        validation_alias="APP_NAME",
    )

    api_key: str = Field(
        default="not-configured",
        validation_alias="ALI_API_KEY",
    )

    model: str = Field(
        default="qwen3.7-plus",
        validation_alias="LLM_MODEL",
    )

    base_url: str = Field(
        default=(
            "https://dashscope.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        validation_alias="LLM_BASE_URL",
    )

    knowledge_dir: Path = Field(
        default=Path("knowledge"),
        validation_alias="KNOWLEDGE_DIR",
    )

    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias="RAG_TOP_K",
    )

    rag_max_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias="RAG_MAX_TOP_K",
    )

    rag_score_threshold: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        validation_alias="RAG_SCORE_THRESHOLD",
    )

    agent_max_rounds: int = Field(
        default=15,
        ge=1,
        le=100,
        validation_alias="AGENT_MAX_ROUNDS",
    )

    agent_max_context_tokens: int = Field(
        default=128_000,
        ge=1_000,
        validation_alias="AGENT_MAX_CONTEXT_TOKENS",
    )

    @model_validator(mode="after")
    def validate_top_k(self):
        if self.rag_top_k > self.rag_max_top_k:
            raise ValueError(
                "RAG_TOP_K 不能大于 RAG_MAX_TOP_K"
            )
        return self

    @classmethod
    def from_env(cls):
        """保留原调用方式，并由 Pydantic 完成环境变量加载。"""
        return cls()