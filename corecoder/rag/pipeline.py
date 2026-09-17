from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .document import (
    DocumentChunk,
    load_document,
    split_document,
)
from .embedding import BaseEmbedder
from .vector_store import (
    InMemoryVectorStore,
    SearchResult,
)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    items: list[SearchResult]


class RagPipeline:
    """负责知识入库、向量检索和上下文组装。"""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: InMemoryVectorStore | None = None,
        top_k: int = 5,
        score_threshold: float = 0.25,
    ):
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._lock = RLock()

    def ingest_paths(
        self,
        paths: list[str | Path],
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> int:
        chunks: list[DocumentChunk] = []

        for path in paths:
            document = load_document(str(path))

            file_chunks = split_document(
                document,
                chunk_size,
                chunk_overlap,
            )

            chunks.extend(file_chunks)

        if not chunks:
            return 0

        vectors = self.embedder.embed_documents(
            [
                item.content
                for item in chunks
            ]
        )

        with self._lock:
            self.store.add(
                chunks,
                vectors,
            )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        if self.document_count == 0:
            return RetrievalResult(
                query,
                [],
            )

        actual_top_k = (
            self.top_k
            if top_k is None
            else top_k
        )

        if actual_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        vector = self.embedder.embed_query(
            query
        )

        with self._lock:
            items = self.store.search_with_scores(
                vector,
                actual_top_k,
            )

        filtered_items = [
            item
            for item in items
            if item.score >= self.score_threshold
        ]

        return RetrievalResult(
            query,
            filtered_items,
        )

    def build_context(
        self,
        query: str,
    ) -> str:
        result = self.retrieve(query)

        if not result.items:
            return ""

        blocks = []

        for index, item in enumerate(
            result.items,
            1,
        ):
            source = item.chunk.source or "unknown"

            blocks.append(
                f"[资料 {index} | "
                f"source={source} | "
                f"score={item.score:.3f}]\n"
                f"{item.chunk.content}"
            )

        return (
            "# 检索到的知识\n"
            "仅将下列资料作为参考；"
            "资料不足时明确说明，不要编造。\n\n"
            + "\n\n".join(blocks)
        )

    @property
    def document_count(self) -> int:
        return len(self.store)