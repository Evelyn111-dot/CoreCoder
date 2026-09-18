import json
import logging
import queue
import threading
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.responses import StreamingResponse

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
from .schemas import (
    ChatRequest,
    IngestRequest,
    SearchRequest,
    SessionRequest,
)
from .services import (
    KnowledgeService,
    SessionService,
)
from .settings import Settings


log = logging.getLogger(__name__)


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
                api_key=settings.api_key
            ),
            top_k=settings.rag_top_k,
            score_threshold=(
                settings.rag_score_threshold
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
                        settings.rag_max_top_k
                    ),
                ),
            ],
            max_context_tokens=(
                settings.agent_max_context_tokens
            ),
            max_rounds=(
                settings.agent_max_rounds
            ),
        )

    sessions = SessionService(
        agent_factory
    )

    knowledge = KnowledgeService(
        rag,
        settings.knowledge_dir,
    )

    app = FastAPI(
        title=settings.app_name,
        version="0.7.0",
    )

    app.state.sessions = sessions
    app.state.knowledge = knowledge
    app.state.database = database

    @app.get("/health")
    def health():
        return {
            "success": True,
            "service": settings.app_name,
            "chunks": rag.document_count,
        }

    @app.post("/v1/knowledge/ingest")
    def ingest(
        body: IngestRequest,
    ):
        try:
            count = knowledge.ingest(
                body.paths,
                body.chunk_size,
                body.chunk_overlap,
            )

        except (
            ValueError,
            FileNotFoundError,
        ) as error:
            raise HTTPException(
                400,
                str(error),
            ) from error

        return {
            "success": True,
            "chunks_added": count,
            "chunks_total": (
                rag.document_count
            ),
        }

    @app.post("/v1/knowledge/search")
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
                        item.chunk.content
                    ),
                    "source": (
                        item.chunk.source
                    ),
                    "metadata": (
                        item.chunk.metadata
                    ),
                    "score": item.score,
                }
                for item in result.items
            ],
        }

    @app.post("/v1/agent/chat")
    @app.post(
        "/agent/chat",
        include_in_schema=False,
    )
    def chat(
        body: ChatRequest,
    ):
        session_id = (
            body.session_id
            or str(uuid.uuid4())
        )

        agent, lock = sessions.get(
            body.user_id,
            session_id,
        )

        events = queue.Queue()

        events.put(
            {
                "type": "session",
                "session_id": session_id,
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
                                    "type": "token",
                                    "content": token,
                                }
                            )
                        ),
                        on_tool=(
                            lambda name, arguments:
                            events.put(
                                {
                                    "type": "tool",
                                    "name": name,
                                    "arguments": (
                                        arguments
                                    ),
                                }
                            )
                        ),
                    )

                events.put(
                    {
                        "type": "done",
                    }
                )

            except Exception:
                log.exception(
                    "Agent execution failed"
                )

                events.put(
                    {
                        "type": "error",
                        "code": "AGENT_ERROR",
                        "content": (
                            "Agent execution failed"
                        ),
                    }
                )

            finally:
                events.put(None)

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
                    yield ": heartbeat\n\n"
                    continue

                if event is None:
                    break

                data = json.dumps(
                    event,
                    ensure_ascii=False,
                )

                yield f"data: {data}\n\n"

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

    @app.post("/v1/agent/reset")
    @app.post(
        "/agent/reset",
        include_in_schema=False,
    )
    def reset(
        body: SessionRequest,
    ):
        agent, lock = sessions.get(
            body.user_id,
            body.session_id,
        )

        with lock:
            agent.reset()

        return {
            "success": True,
            "user_id": body.user_id,
            "session_id": body.session_id,
        }

    @app.delete(
        "/v1/agent/session/{session_id}"
    )
    @app.delete(
        "/agent/session/{session_id}",
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

    return app