import hashlib
import threading
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

from corecoder.rag import RagPipeline

from .knowledge_repository import (
    KnowledgeRepository,
)
from .file_storage import FileStorage

class SessionService:
    """管理用户会话及会话级互斥锁。"""

    def __init__(
        self,
        agent_factory,
    ):
        self._agent_factory = agent_factory
        self._agents = {}
        self._locks = {}
        self._guard = threading.Lock()

    def get(
        self,
        user_id,
        session_id,
    ):
        key = (
            user_id,
            session_id,
        )

        with self._guard:
            if key not in self._agents:
                self._agents[key] = (
                    self._agent_factory()
                )

                self._locks[key] = (
                    threading.Lock()
                )

            return (
                self._agents[key],
                self._locks[key],
            )

    def delete(
        self,
        user_id,
        session_id,
    ):
        key = (
            user_id,
            session_id,
        )

        with self._guard:
            agent = self._agents.get(
                key
            )

            lock = self._locks.get(
                key
            )

        if (
            agent is None
            or lock is None
        ):
            return False

        with lock, self._guard:
            if (
                self._agents.get(key)
                is not agent
            ):
                return False

            self._agents.pop(
                key,
                None,
            )

            self._locks.pop(
                key,
                None,
            )

        return True


@dataclass(frozen=True)
class IngestItem:
    path: str
    status: str
    document_id: int | None
    chunks: int = 0
    reason: str | None = None


class KnowledgeService:
    """处理知识库和文档入库业务流程。"""

    def __init__(
            self,
            pipeline: RagPipeline,
            knowledge_dir: Path,
            repository: KnowledgeRepository,
            file_storage: FileStorage | None = None,
    ):
        self.pipeline = pipeline

        self.knowledge_dir = (
            knowledge_dir.resolve()
        )

        self.repository = repository
        self.file_storage = file_storage

    def create_base(
        self,
        name: str,
        description: str | None = None,
    ):
        """创建知识库。"""

        knowledge_base_id = (
            self.repository.create_base(
                name,
                description,
            )
        )

        return self.repository.get_base(
            knowledge_base_id
        )

    def list_bases(
        self,
    ):
        """查询知识库列表。"""

        return self.repository.list_bases()

    def list_documents(
        self,
        knowledge_base_id: int,
    ):
        """查询指定知识库的文档。"""

        if (
            self.repository.get_base(
                knowledge_base_id
            )
            is None
        ):
            raise LookupError(
                "知识库不存在"
            )

        return (
            self.repository.list_documents(
                knowledge_base_id
            )
        )

    def _ingest_one(
            self,
            knowledge_base_id,
            path,
            file_name,
            display_path,
            chunk_size,
            chunk_overlap,
    ):
        checksum = self._checksum(
            path
        )

        existing = (
            self.repository
            .find_completed_by_checksum(
                knowledge_base_id,
                checksum,
            )
        )

        if existing:
            return IngestItem(
                path=display_path,
                status="skipped",
                document_id=int(
                    existing["id"]
                ),
                chunks=int(
                    existing[
                        "chunk_count"
                    ]
                ),
                reason=(
                    "相同内容已经完成入库"
                ),
            )

        document_id = (
            self.repository
            .prepare_document(
                knowledge_base_id,
                file_name,
                str(path),
                checksum,
            )
        )

        try:
            chunk_count = (
                self.pipeline
                .ingest_paths(
                    [path],
                    chunk_size,
                    chunk_overlap,
                )
            )

            self.repository.mark_completed(
                document_id,
                chunk_count,
            )

            return IngestItem(
                path=display_path,
                status="completed",
                document_id=document_id,
                chunks=chunk_count,
            )

        except Exception as error:  # noqa: BLE001
            self.repository.mark_failed(
                document_id,
                str(error),
            )

            return IngestItem(
                path=display_path,
                status="failed",
                document_id=document_id,
                reason=str(error),
            )

    def ingest(
            self,
            knowledge_base_id,
            paths,
            chunk_size,
            chunk_overlap,
    ):
        if (
                self.repository.get_base(
                    knowledge_base_id
                )
                is None
        ):
            raise LookupError(
                "知识库不存在"
            )

        items = []

        for value in paths:
            path = self._safe_path(
                value
            )

            items.append(
                self._ingest_one(
                    knowledge_base_id,
                    path,
                    path.name,
                    value,
                    chunk_size,
                    chunk_overlap,
                )
            )

        return [
            asdict(item)
            for item in items
        ]

    def _safe_path(
        self,
        value: str,
    ) -> Path:
        """限制只能读取 KNOWLEDGE_DIR 下的文件。"""

        candidate = (
            self.knowledge_dir
            / value
        ).resolve()

        if (
            candidate
            != self.knowledge_dir
            and self.knowledge_dir
            not in candidate.parents
        ):
            raise ValueError(
                "知识文件必须位于 "
                "KNOWLEDGE_DIR 中"
            )

        if not candidate.is_file():
            raise FileNotFoundError(
                value
            )

        return candidate

    @staticmethod
    def _checksum(
        path: Path,
    ) -> str:
        """计算文件 SHA-256，用于判断文件内容是否重复。"""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(
                lambda: file.read(
                    64 * 1024
                ),
                b"",
            ):
                digest.update(
                    block
                )

        return digest.hexdigest()

    async def ingest_uploads(
            self,
            knowledge_base_id,
            uploads,
            chunk_size,
            chunk_overlap,
    ):
        """保存并入库管理员上传的多个文件。"""

        if (
                self.repository.get_base(
                    knowledge_base_id
                )
                is None
        ):
            raise LookupError(
                "知识库不存在"
            )

        if self.file_storage is None:
            raise RuntimeError(
                "文件存储未配置"
            )

        items = []

        for upload in uploads:
            original_name = Path(
                upload.filename
                or "unnamed"
            ).name

            try:
                stored = (
                    await self.file_storage
                    .save(
                        knowledge_base_id,
                        upload,
                    )
                )

                item = (
                    self._ingest_one(
                        knowledge_base_id,
                        stored.path,
                        stored.original_name,
                        stored.original_name,
                        chunk_size,
                        chunk_overlap,
                    )
                )

                if item.status == "skipped":
                    self.file_storage.delete_if_managed(
                        knowledge_base_id,
                        stored.path,
                    )

                items.append(
                    item
                )

            except (
                    ValueError,
                    OSError,
            ) as error:
                items.append(
                    IngestItem(
                        path=original_name,
                        status="failed",
                        document_id=None,
                        reason=str(error),
                    )
                )

        self.file_storage.remove_empty_directory(
            knowledge_base_id
        )

        return [
            asdict(item)
            for item in items
        ]

    def _ingest_one(
            self,
            knowledge_base_id,
            path,
            file_name,
            display_path,
            chunk_size,
            chunk_overlap,
    ):
        checksum = self._checksum(
            path
        )

        existing = (
            self.repository
            .find_completed_by_checksum(
                knowledge_base_id,
                checksum,
            )
        )

        if existing:
            return IngestItem(
                path=display_path,
                status="skipped",
                document_id=int(
                    existing["id"]
                ),
                chunks=int(
                    existing[
                        "chunk_count"
                    ]
                ),
                reason=(
                    "相同内容已经完成入库"
                ),
            )

        document_id = (
            self.repository
            .prepare_document(
                knowledge_base_id,
                file_name,
                str(path),
                checksum,
            )
        )

        try:
            chunk_count = (
                self.pipeline
                .ingest_paths(
                    [path],
                    chunk_size,
                    chunk_overlap,
                )
            )

            self.repository.mark_completed(
                document_id,
                chunk_count,
            )

            return IngestItem(
                path=display_path,
                status="completed",
                document_id=document_id,
                chunks=chunk_count,
            )

        except Exception as error:  # noqa: BLE001
            self.repository.mark_failed(
                document_id,
                str(error),
            )

            return IngestItem(
                path=display_path,
                status="failed",
                document_id=document_id,
                reason=str(error),
            )

    def delete_document(
            self,
            knowledge_base_id,
            document_id,
    ):
        document = (
            self.repository
            .get_document(
                knowledge_base_id,
                document_id,
            )
        )

        if document is None:
            raise LookupError(
                "知识库文档不存在"
            )

        source = (
            document["source_uri"]
        )

        self.pipeline.store.delete_by_source(
            source
        )

        if self.file_storage:
            self.file_storage.delete_if_managed(
                knowledge_base_id,
                source,
            )

            self.file_storage.remove_empty_directory(
                knowledge_base_id
            )

        deleted = (
            self.repository
            .delete_document(
                knowledge_base_id,
                document_id,
            )
        )

        return deleted > 0

    def delete_base(
            self,
            knowledge_base_id,
    ):
        if (
                self.repository.get_base(
                    knowledge_base_id
                )
                is None
        ):
            raise LookupError(
                "知识库不存在"
            )

        documents = (
            self.repository
            .list_documents(
                knowledge_base_id
            )
        )

        for document in documents:
            source = (
                document["source_uri"]
            )

            self.pipeline.store.delete_by_source(
                source
            )

            if self.file_storage:
                self.file_storage.delete_if_managed(
                    knowledge_base_id,
                    source,
                )

        if self.file_storage:
            self.file_storage.remove_empty_directory(
                knowledge_base_id
            )

        deleted = (
            self.repository
            .delete_base(
                knowledge_base_id
            )
        )

        return deleted > 0