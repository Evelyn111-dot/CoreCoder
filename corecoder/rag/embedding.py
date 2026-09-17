from __future__ import annotations

from abc import ABC, abstractmethod
import os
import time

import requests


class BaseEmbedder(ABC):
    """Embedding 接口。"""

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """生成文档向量。"""
        pass

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """生成查询向量。"""
        pass


class DashScopeEmbedder(BaseEmbedder):
    """DashScope Embedding HTTP 实现。"""

    DEFAULT_MODEL = "text-embedding-v4"
    DEFAULT_DIMENSION = 1024
    MAX_BATCH_SIZE = 10

    BASE_URL = (
        "https://dashscope.aliyuncs.com"
        "/api/v1/services/embeddings/"
        "text-embedding/text-embedding"
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("ALI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "未找到 ALI_API_KEY"
            )

        if dimension <= 0:
            raise ValueError(
                "dimension 必须大于 0"
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries 不能小于 0"
            )

        if timeout <= 0:
            raise ValueError(
                "timeout 必须大于 0"
            )

        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        self._validate_texts(texts)

        vectors = []

        for batch in self._batch(
            texts,
            self.MAX_BATCH_SIZE,
        ):
            batch_vectors = self._request(
                batch,
                text_type="document",
            )

            vectors.extend(batch_vectors)

        return vectors

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        self._validate_text(text)

        vectors = self._request(
            [text],
            text_type="query",
        )

        return vectors[0]

    def _request(
        self,
        texts: list[str],
        text_type: str,
    ) -> list[list[float]]:

        payload = {
            "model": self.model,
            "input": {
                "texts": texts,
            },
            "parameters": {
                "text_type": text_type,
                "dimension": self.dimension,
                "output_type": "dense",
            },
        }

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        self._format_api_error(
                            response
                        )
                    )

                data = response.json()

                return self._parse_response(
                    data,
                    len(texts),
                )

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as error:

                last_error = error

                if attempt >= self.max_retries:
                    break

                time.sleep(
                    self.retry_delay
                    * (attempt + 1)
                )

        raise RuntimeError(
            "Embedding 请求失败"
        ) from last_error

    def _parse_response(
        self,
        data: dict,
        expected_size: int,
    ) -> list[list[float]]:

        output = data.get("output")

        if not output:
            raise RuntimeError(
                "Embedding API 返回结果为空"
            )

        embeddings = output.get(
            "embeddings"
        )

        if not embeddings:
            raise RuntimeError(
                "Embedding API 未返回 embeddings"
            )

        embeddings = sorted(
            embeddings,
            key=lambda item: item["text_index"],
        )

        vectors = [
            item["embedding"]
            for item in embeddings
        ]

        if len(vectors) != expected_size:
            raise RuntimeError(
                "Embedding 返回数量异常: "
                f"expected={expected_size}, "
                f"actual={len(vectors)}"
            )

        for vector in vectors:
            if len(vector) != self.dimension:
                raise RuntimeError(
                    "Embedding 向量维度异常: "
                    f"expected={self.dimension}, "
                    f"actual={len(vector)}"
                )

        return vectors

    @staticmethod
    def _batch(
        texts: list[str],
        batch_size: int,
    ):
        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            yield texts[
                start:start + batch_size
            ]

    @staticmethod
    def _validate_text(
        text: str,
    ):
        if not isinstance(text, str):
            raise TypeError(
                "text 必须是字符串"
            )

        if not text.strip():
            raise ValueError(
                "text 不能为空"
            )

    def _validate_texts(
        self,
        texts: list[str],
    ):
        if not isinstance(texts, list):
            raise TypeError(
                "texts 必须是 list"
            )

        if not texts:
            raise ValueError(
                "texts 不能为空"
            )

        for text in texts:
            self._validate_text(text)

    @staticmethod
    def _format_api_error(
        response: requests.Response,
    ) -> str:

        try:
            data = response.json()
        except ValueError:
            data = response.text

        return (
            "DashScope API 请求失败: "
            f"status={response.status_code}, "
            f"response={data}"
        )