from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


SUPPORTED_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".text",
    ".pdf",
    ".docx",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".go",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rs",
}


@dataclass(frozen=True)
class StoredFile:
    original_name: str
    path: Path
    size: int


class FileStorage:
    """将管理员上传的文件保存到 knowledge/{知识库ID}/。"""

    def __init__(
        self,
        knowledge_dir: Path,
        max_size_mb: int,
    ):
        self.knowledge_dir = (
            knowledge_dir.resolve()
        )

        self.max_size_bytes = (
            max_size_mb
            * 1024
            * 1024
        )

    async def save(
        self,
        knowledge_base_id: int,
        upload: UploadFile,
    ) -> StoredFile:
        """保存一个上传文件。"""

        original_name = Path(
            upload.filename or ""
        ).name

        if not original_name:
            raise ValueError(
                "上传文件缺少文件名"
            )

        suffix = (
            Path(original_name)
            .suffix
            .lower()
        )

        if (
            suffix
            not in SUPPORTED_SUFFIXES
        ):
            raise ValueError(
                "不支持的文件类型: "
                f"{suffix or '无扩展名'}"
            )

        directory = (
            self._base_directory(
                knowledge_base_id
            )
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        stored_name = (
            f"{uuid4().hex}_"
            f"{original_name}"
        )

        target = (
            directory
            / stored_name
        )

        size = 0

        try:
            with target.open(
                "wb"
            ) as output:
                while block := (
                    await upload.read(
                        64 * 1024
                    )
                ):
                    size += len(block)

                    if (
                        size
                        > self.max_size_bytes
                    ):
                        raise ValueError(
                            "文件超过大小限制: "
                            f"{self.max_size_bytes // 1024 // 1024} MB"
                        )

                    output.write(
                        block
                    )

        except Exception:
            target.unlink(
                missing_ok=True
            )
            raise

        finally:
            await upload.close()

        if size == 0:
            target.unlink(
                missing_ok=True
            )

            raise ValueError(
                "不能上传空文件"
            )

        return StoredFile(
            original_name=(
                original_name
            ),
            path=target.resolve(),
            size=size,
        )

    def delete_if_managed(
        self,
        knowledge_base_id: int,
        value: str | Path,
    ) -> bool:
        """只删除 knowledge/{id}/ 中的正式上传文件。"""

        candidate = (
            Path(value).resolve()
        )

        directory = (
            self._base_directory(
                knowledge_base_id
            )
        )

        if (
            directory
            not in candidate.parents
        ):
            return False

        if not candidate.is_file():
            return False

        candidate.unlink()

        return True

    def remove_empty_directory(
        self,
        knowledge_base_id: int,
    ) -> None:
        """删除已经为空的知识库文件夹。"""

        directory = (
            self._base_directory(
                knowledge_base_id
            )
        )

        if (
            directory.is_dir()
            and not any(
                directory.iterdir()
            )
        ):
            directory.rmdir()

    def _base_directory(
        self,
        knowledge_base_id: int,
    ) -> Path:
        if knowledge_base_id <= 0:
            raise ValueError(
                "knowledge_base_id "
                "必须大于 0"
            )

        return (
            self.knowledge_dir
            / str(
                knowledge_base_id
            )
        ).resolve()