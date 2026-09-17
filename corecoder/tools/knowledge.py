from __future__ import annotations

from corecoder.rag import RagPipeline

from .base import Tool


class KnowledgeSearchTool(Tool):
    """让 Agent 主动检索知识库的只读工具。"""

    name = "search_knowledge"

    description = (
        "Search the indexed knowledge base for information relevant "
        "to the current task. "
        "Use this tool when the answer may depend on project documents "
        "or domain knowledge. "
        "Choose a focused query and an appropriate top_k: "
        "use 2-3 for a narrow fact, "
        "4-6 for comparison or explanation, "
        "and more only for broad synthesis."
    )

    def __init__(
        self,
        pipeline: RagPipeline,
        max_top_k: int = 10,
    ):
        if max_top_k <= 0:
            raise ValueError("max_top_k 必须大于 0")

        self.pipeline = pipeline
        self.max_top_k = max_top_k

        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Focused semantic search query "
                        "for the knowledge base."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_top_k,
                    "default": 3,
                    "description": (
                        "Maximum number of document chunks "
                        "to retrieve."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        if not isinstance(query, str):
            return "Error: query 必须是字符串"

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            return "Error: top_k 必须是整数"

        query = query.strip()

        if not query:
            return "Error: query 不能为空"

        # 模型可以选择数量，但服务端始终限制最大值。
        safe_top_k = max(
            1,
            min(top_k, self.max_top_k),
        )

        result = self.pipeline.retrieve(
            query=query,
            top_k=safe_top_k,
        )

        if not result.items:
            return (
                "No relevant knowledge found. "
                "Answer cautiously and state when "
                "the knowledge base is insufficient."
            )

        blocks = []

        for index, item in enumerate(
            result.items,
            1,
        ):
            source = item.chunk.source or "unknown"

            blocks.append(
                f"[Result {index} | "
                f"source={source} | "
                f"score={item.score:.3f}]\n"
                f"{item.chunk.content}"
            )

        return "\n\n".join(blocks)