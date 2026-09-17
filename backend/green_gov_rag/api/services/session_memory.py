"""Session memory for multi-turn conversations.

GreenGovRAG previously had no conversation memory (each query was independent;
``query_history`` was analytics-only). This stores the recent turns of a session so
the query pipeline can (a) resolve coreferences ("what about *its* requirements?")
via LLM rewriting and (b) keep answers coherent across turns.

Backends:
- Redis (distributed, survives multiple API workers / restarts) when
  ``enable_redis_cache`` is on and Redis is reachable.
- In-process dict fallback otherwise (fine for single-worker dev).

Only the last ``session_memory_max_turns`` turns are retained per session.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict, deque
from typing import Optional

from green_gov_rag.config import settings

logger = logging.getLogger(__name__)


class SessionMemory:
    """Store and retrieve recent (user, assistant) turns per session id."""

    def __init__(
        self,
        max_turns: Optional[int] = None,
        ttl: Optional[int] = None,
        enable_redis: Optional[bool] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
    ) -> None:
        self.max_turns = max_turns or settings.session_memory_max_turns
        self.ttl = ttl or settings.session_memory_ttl
        self.enable_redis = (
            settings.enable_redis_cache if enable_redis is None else enable_redis
        )

        # In-memory fallback: session_id -> deque[turn dict]; bounded LRU of sessions.
        self._memory: "OrderedDict[str, deque]" = OrderedDict()
        self._max_sessions = 5000

        self.redis_client = None
        if self.enable_redis:
            try:
                import redis

                self.redis_client = redis.Redis(
                    host=redis_host or settings.redis_host,
                    port=redis_port or settings.redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                self.redis_client.ping()
                logger.info("Session memory using Redis")
            except Exception as exc:  # noqa: BLE001 - fall back to memory
                logger.warning("Session memory Redis unavailable (%s); using memory", exc)
                self.redis_client = None

    def _redis_key(self, session_id: str) -> str:
        return f"session_memory:{session_id}"

    def get_history(self, session_id: Optional[str]) -> list[dict]:
        """Return recent turns as ``[{"user": ..., "assistant": ...}, ...]`` (oldest first)."""
        if not session_id:
            return []

        if self.redis_client:
            try:
                raw = self.redis_client.get(self._redis_key(session_id))
                if raw:
                    return json.loads(raw)[-self.max_turns :]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session memory get failed: %s", exc)
            return []

        turns = self._memory.get(session_id)
        return list(turns) if turns else []

    def add_turn(self, session_id: Optional[str], user: str, assistant: str) -> None:
        """Append a (user, assistant) turn, trimming to the retention window."""
        if not session_id:
            return

        turn = {"user": user, "assistant": assistant}

        if self.redis_client:
            try:
                history = self.get_history(session_id)
                history.append(turn)
                history = history[-self.max_turns :]
                self.redis_client.setex(
                    self._redis_key(session_id), self.ttl, json.dumps(history)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session memory set failed: %s", exc)
            return

        turns = self._memory.get(session_id)
        if turns is None:
            turns = deque(maxlen=self.max_turns)
            self._memory[session_id] = turns
        turns.append(turn)
        self._memory.move_to_end(session_id)
        # Evict oldest sessions if over capacity.
        while len(self._memory) > self._max_sessions:
            self._memory.popitem(last=False)

    def format_history(self, session_id: Optional[str]) -> str:
        """Render history as a plain-text transcript for prompting."""
        history = self.get_history(session_id)
        lines = []
        for turn in history:
            lines.append(f"User: {turn.get('user', '')}")
            lines.append(f"Assistant: {turn.get('assistant', '')}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        if self.redis_client:
            try:
                self.redis_client.delete(self._redis_key(session_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session memory clear failed: %s", exc)
        else:
            self._memory.pop(session_id, None)
