"""ASGI 入口：uvicorn api.server:app --reload"""

from corecoder.backend import create_app

app = create_app()
get_agent = app.state.sessions.get