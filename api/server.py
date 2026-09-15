"""HTTP API for CoreCoder.

The API layer keeps CoreCoder's Agent unchanged and adds:
- SSE streaming
- user-isolated session management
- per-session concurrency control
- request validation
- SSE heartbeat
- safe error handling
"""

import json
import logging
import os
import queue
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from corecoder import Agent, LLM
from corecoder.tools import GlobTool, GrepTool, ReadFileTool

# Logging
log = logging.getLogger(__name__)

# FastAPI

app = FastAPI()

# LLM

llm = LLM(
    model="qwen3.7-plus",
    api_key=os.getenv("ALI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Session

# session belongs to one user.
# (user_id, session_id) uniquely identifies an Agent.
# locks:
#     one lock for each session
#     prevents the same Agent from being used concurrently
#
# sessions_lock:
#     protects the agents / locks dictionaries themselves

agents: dict[tuple[str, str], Agent] = {}
locks: dict[tuple[str, str], threading.Lock] = {}

sessions_lock = threading.Lock()

# SSE

HEARTBEAT_INTERVAL = 15

def create_agent() -> Agent:
    """Create an Agent with the tools exposed by this API."""
    return Agent(
        llm=llm,
        tools=[
            ReadFileTool(),
            GrepTool(),
            GlobTool(),
        ],
        max_rounds=15,
    )


def get_agent(
    user_id: str,
    session_id: str,
) -> tuple[Agent, threading.Lock]:
    """Get or create an Agent for a user's session."""

    key = (user_id, session_id)

    with sessions_lock:
        if key not in agents:
            agents[key] = create_agent()

        if key not in locks:
            locks[key] = threading.Lock()

        return agents[key], locks[key]


class ChatRequest(BaseModel):
    """Request body for chat."""

    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    message: str = Field(
        min_length=1,
        max_length=20_000,
    )

    session_id: str | None = Field(
        default=None,
        max_length=128,
    )


class SessionRequest(BaseModel):
    """Request body for session operations."""

    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    session_id: str = Field(
        min_length=1,
        max_length=128,
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "success": True,
        "service": "CoreCoder",
    }


@app.post("/agent/chat")
def chat(request: ChatRequest):
    """Chat with CoreCoder through SSE."""

    # If the client does not provide a session ID,
    # create a new session.
    session_id = request.session_id or str(uuid.uuid4())

    agent, lock = get_agent(
        request.user_id,
        session_id,
    )

    events = queue.Queue()

    events.put({
        "type": "session",
        "session_id": session_id,
    })

    def on_token(token: str):
        """Receive streamed LLM tokens."""
        events.put({
            "type": "token",
            "content": token,
        })

    def on_tool(name: str, arguments: dict):
        """Receive tool-call events."""
        events.put({
            "type": "tool",
            "name": name,
            "arguments": arguments,
        })

    # Run Agent in background
    #
    # The HTTP request thread does not directly block on agent.chat().
    # The Agent writes events into the queue.

    def run_agent():
        try:
            # Only one request may operate on the same session
            # at the same time.
            with lock:
                agent.chat(
                    request.message,
                    on_token=on_token,
                    on_tool=on_tool,
                )

            events.put({
                "type": "done",
            })

        except Exception:
            # Keep detailed error information in server logs.
            log.exception(
                "Agent execution failed",
                extra={
                    "user_id": request.user_id,
                    "session_id": session_id,
                },
            )

            # Do not expose internal exception details to clients.
            events.put({
                "type": "error",
                "code": "AGENT_ERROR",
                "content": "Agent execution failed",
            })

        finally:
            # Tell the SSE stream that the Agent execution is finished.
            events.put(None)

    thread = threading.Thread(
        target=run_agent,
        daemon=True,
    )

    thread.start()

    def event_stream():
        while True:
            try:
                # Wait for an Agent event.
                #
                # If no event arrives within HEARTBEAT_INTERVAL,
                # send a heartbeat instead.
                event = events.get(
                    timeout=HEARTBEAT_INTERVAL,
                )

            except queue.Empty:
                # SSE comment.
                #
                # It is ignored by the SSE client but keeps
                # the HTTP connection alive.
                yield ": heartbeat\n\n"
                continue

            # None means the Agent has completely finished.
            if event is None:
                break

            yield (
                f"data: "
                f"{json.dumps(event, ensure_ascii=False)}"
                f"\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/reset")
def reset_session(request: SessionRequest):
    """Clear the conversation history of a user's session."""

    agent, lock = get_agent(
        request.user_id,
        request.session_id,
    )

    # Wait for an ongoing chat request to finish before resetting.
    with lock:
        agent.reset()

    return {
        "success": True,
        "user_id": request.user_id,
        "session_id": request.session_id,
    }


@app.delete("/agent/session/{session_id}")
def delete_session(
    session_id: str,
    user_id: str,
):
    """Delete a user's session completely."""

    key = (user_id, session_id)

    with sessions_lock:
        agent = agents.get(key)
        lock = locks.get(key)

    if agent is None or lock is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    # Wait for any ongoing chat/reset operation to finish.
    #
    # This prevents deleting an Agent while another thread is using it.

    with lock:
        # Re-check the session after acquiring the lock.
        #
        # Another thread may have deleted/recreated the session
        # while we were waiting.

        with sessions_lock:
            current_agent = agents.get(key)

            if current_agent is not agent:
                raise HTTPException(
                    status_code=404,
                    detail="Session not found",
                )

            agents.pop(key, None)
            locks.pop(key, None)

    return {
        "success": True,
        "user_id": user_id,
        "session_id": session_id,
    }