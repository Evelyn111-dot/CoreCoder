from corecoder.rag import (
    DocumentChunk,
    InMemoryVectorStore,
    RagPipeline,
)
from corecoder.tools.knowledge import (
    KnowledgeSearchTool,
)


class FakeEmbedder:
    def embed_documents(
        self,
        texts,
    ):
        return [
            [1.0, 0.0]
            for _ in texts
        ]

    def embed_query(
        self,
        text,
    ):
        return [1.0, 0.0]


def make_tool(
    max_top_k=5,
):
    store = InMemoryVectorStore()

    store.add(
        [
            DocumentChunk(
                "first",
                "one.md",
            ),
            DocumentChunk(
                "second",
                "two.md",
            ),
            DocumentChunk(
                "third",
                "three.md",
            ),
        ],
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
    )

    pipeline = RagPipeline(
        FakeEmbedder(),
        store,
        score_threshold=0.0,
    )

    return KnowledgeSearchTool(
        pipeline,
        max_top_k=max_top_k,
    )


def test_agent_can_choose_top_k():
    result = make_tool().execute(
        "knowledge",
        top_k=2,
    )

    assert "Result 1" in result
    assert "Result 2" in result
    assert "Result 3" not in result


def test_backend_caps_agent_top_k():
    tool = make_tool(
        max_top_k=2
    )

    result = tool.execute(
        "knowledge",
        top_k=999,
    )

    assert "Result 3" not in result

    schema = tool.schema()

    top_k_schema = (
        schema["function"]
        ["parameters"]
        ["properties"]
        ["top_k"]
    )

    assert top_k_schema["maximum"] == 2


def test_empty_query_is_rejected():
    result = make_tool().execute(
        "   ",
        top_k=2,
    )

    assert result.startswith(
        "Error:"
    )