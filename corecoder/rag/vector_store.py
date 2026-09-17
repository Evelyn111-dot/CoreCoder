from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .document import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class BaseVectorStore(ABC):
    """向量存储接口。"""

    @abstractmethod
    def add(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """保存文档向量。"""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """检索相似文档。"""


class InMemoryVectorStore(BaseVectorStore):
    """内存向量存储。"""

    def __init__(self):
        self._items: list[tuple[DocumentChunk, list[float]]] = []

    def add(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量不一致")
        if not chunks:
            return
        dimension = len(vectors[0])
        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError("向量维度不一致")
        self._items.extend(zip(chunks, vectors))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        if not query_vector:
            raise ValueError("query_vector 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        results = []
        for chunk, vector in self._items:
            score = self._cosine_similarity(query_vector, vector)
            results.append((score, chunk))
        results.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    def search_with_scores(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not query_vector or top_k <= 0:
            raise ValueError("query_vector 不能为空且 top_k 必须大于 0")
        ranked = [
            SearchResult(
                chunk=chunk,
                score=self._cosine_similarity(query_vector, vector),
            )
            for chunk, vector in self._items
        ]
        return sorted(
            ranked,
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("向量维度不一致")
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)