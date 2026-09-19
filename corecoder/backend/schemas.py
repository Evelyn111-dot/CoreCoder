from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    message: str = Field(
        min_length=1,
        max_length=20_000,
    )

    session_id: str | None = Field(
        default=None,
        max_length=128,
    )


class SessionRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    session_id: str = Field(
        min_length=1,
        max_length=128,
    )


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
    )

    description: str | None = Field(
        default=None,
        max_length=500,
    )


class IngestRequest(BaseModel):
    knowledge_base_id: int = Field(
        ge=1,
    )

    paths: list[str] = Field(
        min_length=1,
        max_length=100,
    )

    chunk_size: int = Field(
        default=800,
        ge=100,
        le=8000,
    )

    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=2000,
    )


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
    )