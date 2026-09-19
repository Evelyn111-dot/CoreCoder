import json
import logging
import queue
import threading
import uuid
from typing import Annotated

from fastapi.openapi.utils import get_openapi

from .file_storage import FileStorage

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    StreamingResponse,
)
from pymysql.err import IntegrityError

from corecoder import Agent, LLM
from corecoder.rag import (
    DashScopeEmbedder,
    RagPipeline,
)
from corecoder.tools import (
    GlobTool,
    GrepTool,
    ReadFileTool,
)
from corecoder.tools.knowledge import (
    KnowledgeSearchTool,
)

from .database import Database
from .knowledge_repository import (
    KnowledgeRepository,
)
from .schemas import (
    ChatRequest,
    CreateKnowledgeBaseRequest,
    IngestRequest,
    SearchRequest,
    SessionRequest,
)
from .services import (
    KnowledgeService,
    SessionService,
)
from .settings import Settings


log = logging.getLogger(
    __name__
)


def create_app(
    settings=None,
    rag=None,
):
    settings = (
        settings
        or Settings.from_env()
    )

    database = Database(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )

    llm = LLM(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    rag = (
        rag
        or RagPipeline(
            DashScopeEmbedder(
                api_key=(
                    settings.api_key
                )
            ),
            top_k=(
                settings.rag_top_k
            ),
            score_threshold=(
                settings
                .rag_score_threshold
            ),
        )
    )

    def agent_factory():
        return Agent(
            llm,
            [
                ReadFileTool(),
                GrepTool(),
                GlobTool(),
                KnowledgeSearchTool(
                    rag,
                    max_top_k=(
                        settings
                        .rag_max_top_k
                    ),
                ),
            ],
            max_context_tokens=(
                settings
                .agent_max_context_tokens
            ),
            max_rounds=(
                settings
                .agent_max_rounds
            ),
        )

    sessions = SessionService(
        agent_factory
    )

    knowledge_repository = (
        KnowledgeRepository(
            database
        )
    )

    file_storage = FileStorage(
        settings.knowledge_dir,
        settings.max_upload_size_mb,
    )

    knowledge = KnowledgeService(
        pipeline=rag,
        knowledge_dir=settings.knowledge_dir,
        repository=knowledge_repository,
        file_storage=file_storage,
    )

    app = FastAPI(
        title=settings.app_name,
        version="0.7.0",
    )

    app.state.database = database
    app.state.sessions = sessions
    app.state.knowledge = knowledge

    @app.get("/health")
    def health():
        return {
            "success": True,
            "service": (
                settings.app_name
            ),
            "chunks": (
                rag.document_count
            ),
        }

    @app.post(
        "/v1/knowledge/bases",
        status_code=201,
    )
    def create_knowledge_base(
        body: CreateKnowledgeBaseRequest,
    ):
        try:
            return (
                knowledge.create_base(
                    body.name,
                    body.description,
                )
            )

        except IntegrityError as error:
            raise HTTPException(
                409,
                "知识库名称已经存在",
            ) from error

    @app.get(
        "/v1/knowledge/bases"
    )
    def list_knowledge_bases():
        return {
            "items": (
                knowledge.list_bases()
            )
        }

    @app.get(
        "/v1/knowledge/bases/"
        "{knowledge_base_id}/documents"
    )
    def list_knowledge_documents(
        knowledge_base_id: int,
    ):
        try:
            return {
                "items": (
                    knowledge
                    .list_documents(
                        knowledge_base_id
                    )
                )
            }

        except LookupError as error:
            raise HTTPException(
                404,
                str(error),
            ) from error

    @app.post("/v1/knowledge/bases/{knowledge_base_id}/documents/upload")
    async def upload_knowledge_documents(
            knowledge_base_id: int,
            files: Annotated[list[UploadFile], File(...)],
            chunk_size: Annotated[int, Form()] = 800,
            chunk_overlap: Annotated[int, Form()] = 100,
    ):
        if not files:
            raise HTTPException(
                status_code=400,
                detail="请至少上传一个文件",
            )

        if not 100 <= chunk_size <= 8000:
            raise HTTPException(
                status_code=400,
                detail="chunk_size 必须在 100 到 8000 之间",
            )

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=400,
                detail="chunk_overlap 必须大于等于 0 且小于 chunk_size",
            )

        try:
            items = await knowledge.ingest_uploads(
                knowledge_base_id=knowledge_base_id,
                uploads=files,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        except LookupError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error

        return {
            "success": True,
            "chunks_added": sum(
                item["chunks"]
                for item in items
                if item["status"] == "completed"
            ),
            "chunks_total": rag.document_count,
            "completed": sum(
                item["status"] == "completed"
                for item in items
            ),
            "skipped": sum(
                item["status"] == "skipped"
                for item in items
            ),
            "failed": sum(
                item["status"] == "failed"
                for item in items
            ),
            "items": items,
        }

    @app.post(
        "/v1/knowledge/ingest"
    )
    def ingest(
        body: IngestRequest,
    ):
        try:
            items = knowledge.ingest(
                body.knowledge_base_id,
                body.paths,
                body.chunk_size,
                body.chunk_overlap,
            )

        except LookupError as error:
            raise HTTPException(
                404,
                str(error),
            ) from error

        except (
            ValueError,
            FileNotFoundError,
        ) as error:
            raise HTTPException(
                400,
                str(error),
            ) from error

        completed = sum(
            item["status"]
            == "completed"
            for item in items
        )

        skipped = sum(
            item["status"]
            == "skipped"
            for item in items
        )

        failed = sum(
            item["status"]
            == "failed"
            for item in items
        )

        chunks_added = sum(
            item["chunks"]
            for item in items
            if (
                item["status"]
                == "completed"
            )
        )

        return {
            "success": True,
            "chunks_added": (
                chunks_added
            ),
            "chunks_total": (
                rag.document_count
            ),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "items": items,
        }

    @app.post(
        "/v1/knowledge/search"
    )
    def search(
        body: SearchRequest,
    ):
        result = rag.retrieve(
            body.query
        )

        return {
            "query": body.query,
            "items": [
                {
                    "content": (
                        item
                        .chunk
                        .content
                    ),
                    "source": (
                        item
                        .chunk
                        .source
                    ),
                    "metadata": (
                        item
                        .chunk
                        .metadata
                    ),
                    "score": (
                        item.score
                    ),
                }
                for item
                in result.items
            ],
        }

    @app.post(
        "/v1/agent/chat"
    )
    @app.post(
        "/agent/chat",
        include_in_schema=False,
    )
    def chat(
        body: ChatRequest,
    ):
        session_id = (
            body.session_id
            or str(
                uuid.uuid4()
            )
        )

        agent, lock = (
            sessions.get(
                body.user_id,
                session_id,
            )
        )

        events = queue.Queue()

        events.put(
            {
                "type": "session",
                "session_id": (
                    session_id
                ),
            }
        )

        def run():
            try:
                with lock:
                    agent.chat(
                        body.message,
                        on_token=(
                            lambda token:
                            events.put(
                                {
                                    "type": (
                                        "token"
                                    ),
                                    "content": (
                                        token
                                    ),
                                }
                            )
                        ),
                        on_tool=(
                            lambda name, arguments:
                            events.put(
                                {
                                    "type": (
                                        "tool"
                                    ),
                                    "name": (
                                        name
                                    ),
                                    "arguments": (
                                        arguments
                                    ),
                                }
                            )
                        ),
                    )

                events.put(
                    {
                        "type": "done"
                    }
                )

            except Exception:
                log.exception(
                    "Agent execution failed"
                )

                events.put(
                    {
                        "type": "error",
                        "code": (
                            "AGENT_ERROR"
                        ),
                        "content": (
                            "Agent execution "
                            "failed"
                        ),
                    }
                )

            finally:
                events.put(
                    None
                )

        thread = threading.Thread(
            target=run,
            daemon=True,
        )

        thread.start()

        def stream():
            while True:
                try:
                    event = events.get(
                        timeout=15
                    )

                except queue.Empty:
                    yield (
                        ": heartbeat\n\n"
                    )
                    continue

                if event is None:
                    break

                data = json.dumps(
                    event,
                    ensure_ascii=False,
                )

                yield (
                    f"data: {data}\n\n"
                )

        return StreamingResponse(
            stream(),
            media_type=(
                "text/event-stream"
            ),
            headers={
                "Cache-Control": (
                    "no-cache"
                ),
                "X-Accel-Buffering": (
                    "no"
                ),
            },
        )

    @app.post(
        "/v1/agent/reset"
    )
    @app.post(
        "/agent/reset",
        include_in_schema=False,
    )
    def reset(
        body: SessionRequest,
    ):
        agent, lock = (
            sessions.get(
                body.user_id,
                body.session_id,
            )
        )

        with lock:
            agent.reset()

        return {
            "success": True,
            "user_id": (
                body.user_id
            ),
            "session_id": (
                body.session_id
            ),
        }

    @app.delete(
        "/v1/agent/session/"
        "{session_id}"
    )
    @app.delete(
        "/agent/session/"
        "{session_id}",
        include_in_schema=False,
    )
    def delete(
        session_id: str,
        user_id: str,
    ):
        deleted = sessions.delete(
            user_id,
            session_id,
        )

        if not deleted:
            raise HTTPException(
                404,
                "Session not found",
            )

        return {
            "success": True,
            "user_id": user_id,
            "session_id": session_id,
        }

    @app.delete(
        "/v1/knowledge/bases/"
        "{knowledge_base_id}/documents/"
        "{document_id}"
    )
    def delete_knowledge_document(
            knowledge_base_id: int,
            document_id: int,
    ):
        try:
            deleted = (
                knowledge
                .delete_document(
                    knowledge_base_id,
                    document_id,
                )
            )

        except LookupError as error:
            raise HTTPException(
                404,
                str(error),
            ) from error

        return {
            "success": deleted,
            "knowledge_base_id": (
                knowledge_base_id
            ),
            "document_id": (
                document_id
            ),
        }

    @app.delete(
        "/v1/knowledge/bases/"
        "{knowledge_base_id}"
    )
    def delete_knowledge_base(
            knowledge_base_id: int,
    ):
        try:
            deleted = (
                knowledge
                .delete_base(
                    knowledge_base_id
                )
            )

        except LookupError as error:
            raise HTTPException(
                404,
                str(error),
            ) from error

        return {
            "success": deleted,
            "knowledge_base_id": (
                knowledge_base_id
            ),
        }

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        schemas = (
            openapi_schema
            .get("components", {})
            .get("schemas", {})
        )

        upload_schema_name = (
            "Body_upload_knowledge_documents_"
            "v1_knowledge_bases__knowledge_base_id__"
            "documents_upload_post"
        )

        upload_schema = schemas.get(upload_schema_name)

        if upload_schema:
            files_schema = (
                upload_schema
                .get("properties", {})
                .get("files", {})
            )

            file_items = files_schema.get("items", {})

            # Swagger UI 使用 format=binary 显示文件选择按钮。
            file_items.pop("contentMediaType", None)
            file_items["type"] = "string"
            file_items["format"] = "binary"

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app

