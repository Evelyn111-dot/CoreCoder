"""作用是集中保存知识库相关 SQL，避免把 SQL 全写进接口或者 Service。"""
from __future__ import annotations

from typing import Any

from .database import Database


class KnowledgeRepository:
    """知识库元数据的原生 SQL 访问层。"""

    def __init__(
        self,
        database: Database,
    ):
        self.database = database

    def create_base(
        self,
        name: str,
        description: str | None,
    ) -> int:
        """创建知识库，返回知识库自增 ID。"""

        return self.database.insert(
            """
            INSERT INTO knowledge_bases(
                name,
                description
            )
            VALUES (%s, %s)
            """,
            (
                name,
                description,
            ),
        )

    def get_base(
        self,
        knowledge_base_id: int,
    ) -> dict[str, Any] | None:
        """根据 ID 查询知识库。"""

        return self.database.fetch_one(
            """
            SELECT
                id,
                name,
                description,
                status,
                created_at,
                updated_at
            FROM knowledge_bases
            WHERE id = %s
            """,
            (knowledge_base_id,),
        )

    def list_bases(
        self,
    ) -> list[dict[str, Any]]:
        """查询所有知识库。"""

        return self.database.fetch_all(
            """
            SELECT
                id,
                name,
                description,
                status,
                created_at,
                updated_at
            FROM knowledge_bases
            ORDER BY id DESC
            """
        )

    def find_completed_by_checksum(
        self,
        knowledge_base_id: int,
        checksum: str,
    ) -> dict[str, Any] | None:
        """查询相同内容是否已经成功入库。"""

        return self.database.fetch_one(
            """
            SELECT
                id,
                file_name,
                source_uri,
                checksum,
                status,
                chunk_count
            FROM knowledge_documents
            WHERE knowledge_base_id = %s
              AND checksum = %s
              AND status = 'completed'
            LIMIT 1
            """,
            (
                knowledge_base_id,
                checksum,
            ),
        )

    def prepare_document(
        self,
        knowledge_base_id: int,
        file_name: str,
        source_uri: str,
        checksum: str,
    ) -> int:
        """准备文档记录。

        新文件会插入新记录。

        如果同一路径的文件已经存在，说明可能是文件内容更新，
        此时更新原记录，并重新标记为 processing。
        """

        with (
            self.database.transaction()
            as connection,
            connection.cursor()
            as cursor,
        ):
            cursor.execute(
                """
                SELECT id
                FROM knowledge_documents
                WHERE knowledge_base_id = %s
                  AND source_uri = %s
                LIMIT 1
                FOR UPDATE
                """,
                (
                    knowledge_base_id,
                    source_uri,
                ),
            )

            existing = cursor.fetchone()

            if existing:
                document_id = int(
                    existing["id"]
                )

                cursor.execute(
                    """
                    UPDATE knowledge_documents
                    SET file_name = %s,
                        checksum = %s,
                        status = 'processing',
                        chunk_count = 0,
                        error_message = NULL
                    WHERE id = %s
                    """,
                    (
                        file_name,
                        checksum,
                        document_id,
                    ),
                )

                return document_id

            cursor.execute(
                """
                INSERT INTO knowledge_documents(
                    knowledge_base_id,
                    file_name,
                    source_uri,
                    checksum,
                    status
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    'processing'
                )
                """,
                (
                    knowledge_base_id,
                    file_name,
                    source_uri,
                    checksum,
                ),
            )

            return int(
                cursor.lastrowid
            )

    def mark_completed(
        self,
        document_id: int,
        chunk_count: int,
    ) -> None:
        """将文档标记为入库成功。"""

        self.database.execute(
            """
            UPDATE knowledge_documents
            SET status = 'completed',
                chunk_count = %s,
                error_message = NULL
            WHERE id = %s
            """,
            (
                chunk_count,
                document_id,
            ),
        )

    def mark_failed(
        self,
        document_id: int,
        error_message: str,
    ) -> None:
        """将文档标记为失败并保存错误原因。"""

        self.database.execute(
            """
            UPDATE knowledge_documents
            SET status = 'failed',
                error_message = %s
            WHERE id = %s
            """,
            (
                error_message[:2000],
                document_id,
            ),
        )

    def list_documents(
        self,
        knowledge_base_id: int,
    ) -> list[dict[str, Any]]:
        """查询指定知识库下的全部文档。"""

        return self.database.fetch_all(
            """
            SELECT
                id,
                knowledge_base_id,
                file_name,
                source_uri,
                checksum,
                status,
                chunk_count,
                error_message,
                created_at,
                updated_at
            FROM knowledge_documents
            WHERE knowledge_base_id = %s
            ORDER BY id DESC
            """,
            (knowledge_base_id,),
        )

    def get_document(
            self,
            knowledge_base_id: int,
            document_id: int,
    ) -> dict[str, Any] | None:
        """查询指定知识库下的指定文档。"""

        return self.database.fetch_one(
            """
            SELECT
                id,
                knowledge_base_id,
                file_name,
                source_uri,
                status
            FROM knowledge_documents
            WHERE id = %s
              AND knowledge_base_id = %s
            """,
            (
                document_id,
                knowledge_base_id,
            ),
        )

    def delete_document(
            self,
            knowledge_base_id: int,
            document_id: int,
    ) -> int:
        """删除文档数据库记录。"""

        return self.database.execute(
            """
            DELETE FROM knowledge_documents
            WHERE id = %s
              AND knowledge_base_id = %s
            """,
            (
                document_id,
                knowledge_base_id,
            ),
        )

    def delete_base(
            self,
            knowledge_base_id: int,
    ) -> int:
        """删除知识库。

        knowledge_documents 会通过外键 ON DELETE CASCADE 自动删除。
        """

        return self.database.execute(
            """
            DELETE FROM knowledge_bases
            WHERE id = %s
            """,
            (knowledge_base_id,),
        )