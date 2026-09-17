from corecoder.rag import DocumentChunk, InMemoryVectorStore, RagPipeline


class FakeEmbedder:
    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "Java" in text else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.0] if "Java" in text else [0.0, 1.0]


def test_retrieve_and_build_context():
    store = InMemoryVectorStore()
    store.add(
        [
            DocumentChunk("Java线程池复用线程", "java.md"),
            DocumentChunk("Python类型提示", "python.md"),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    pipeline = RagPipeline(
        FakeEmbedder(),
        store,
        top_k=1,
        score_threshold=0.5,
    )
    result = pipeline.retrieve("Java并发")
    assert result.items[0].chunk.source == "java.md"
    assert "仅将下列资料作为参考" in pipeline.build_context("Java并发")


def test_empty_store_does_not_call_embedding():
    pipeline = RagPipeline(FakeEmbedder())
    assert pipeline.retrieve("anything").items == []

def test_retrieve_allows_per_request_top_k():
    store = InMemoryVectorStore()

    store.add(
        [
            DocumentChunk(
                "Java A",
                "a.md",
            ),
            DocumentChunk(
                "Java B",
                "b.md",
            ),
            DocumentChunk(
                "Java C",
                "c.md",
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
        top_k=3,
        score_threshold=0.0,
    )

    result = pipeline.retrieve(
        "Java",
        top_k=1,
    )

    assert len(result.items) == 1