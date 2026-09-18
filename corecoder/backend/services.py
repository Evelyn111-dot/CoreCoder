import threading
from pathlib import Path

from corecoder.rag import RagPipeline


class SessionService:
    """管理用户会话及会话级互斥锁。"""

    def __init__(self, agent_factory):
        self._agent_factory = agent_factory
        self._agents, self._locks = {}, {}
        self._guard = threading.Lock()

    def get(self, user_id, session_id):
        key = (user_id, session_id)
        with self._guard:
            if key not in self._agents:
                self._agents[key] = self._agent_factory()
                self._locks[key] = threading.Lock()
            return self._agents[key], self._locks[key]

    def delete(self, user_id, session_id):
        key = (user_id, session_id)
        with self._guard:
            agent, lock = self._agents.get(key), self._locks.get(key)
        if agent is None or lock is None:
            return False
        with lock, self._guard:
            if self._agents.get(key) is not agent:
                return False
            self._agents.pop(key, None)
            self._locks.pop(key, None)
        return True


class KnowledgeService:
    def __init__(self, pipeline: RagPipeline, knowledge_dir: Path):
        self.pipeline, self.knowledge_dir = pipeline, knowledge_dir.resolve()

    def ingest(self, paths, chunk_size, chunk_overlap):
        return self.pipeline.ingest_paths(
            [self._safe_path(p) for p in paths],
            chunk_size,
            chunk_overlap,
        )

    def _safe_path(self, value):
        candidate = (self.knowledge_dir / value).resolve()
        if candidate != self.knowledge_dir and self.knowledge_dir not in candidate.parents:
            raise ValueError("知识文件必须位于 KNOWLEDGE_DIR 中")
        if not candidate.is_file():
            raise FileNotFoundError(value)
        return candidate