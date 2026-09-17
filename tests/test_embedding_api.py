import os

import pytest

from corecoder.rag.embedding import DashScopeEmbedder


@pytest.fixture
def embedder():
    if not os.getenv("ALI_API_KEY"):
        pytest.skip(
            "未配置 ALI_API_KEY，跳过真实 API 测试"
        )

    return DashScopeEmbedder()


def test_embed_query(embedder):
    vector = embedder.embed_query(
        "Java线程池是怎么工作的？"
    )

    assert isinstance(vector, list)
    assert len(vector) == 1024
    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_embed_documents(embedder):
    texts = [
        "Java线程池可以复用线程。",
        "Python是一种编程语言。",
        "RAG可以用于知识库问答。",
    ]

    vectors = embedder.embed_documents(texts)

    assert len(vectors) == len(texts)

    for vector in vectors:
        assert isinstance(vector, list)
        assert len(vector) == 1024


def test_query_and_document(embedder):
    query_vector = embedder.embed_query(
        "Java线程池"
    )

    document_vectors = embedder.embed_documents(
        [
            "Java线程池用于管理和复用线程。",
            "Python是一种编程语言。",
        ]
    )

    assert len(query_vector) == 1024
    assert len(document_vectors) == 2

    for vector in document_vectors:
        assert len(vector) == 1024