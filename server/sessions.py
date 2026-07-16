"""
Session management for concurrent episodes
==========================================

Production deployments serve more than one analyst / training run at a time.
Each session bundles a full, isolated copy of the environment plus the v3
theme modules (NPC actors, policy drift, ticketing, expert rotation, reward
blend) behind its own lock, so two clients stepping different episodes never
contend or corrupt each other's state.

Clients select a session with the ``X-Session-ID`` header (or the optional
``session_id`` field on ``POST /reset``). Requests without a session ID land
in the ``default`` session, which preserves the original single-tenant
behaviour exactly — existing OpenEnv clients keep working unchanged.

Sessions are evicted after ``SOC_GYM_SESSION_TTL`` seconds of inactivity
(default 3600) and capped at ``SOC_GYM_MAX_SESSIONS`` concurrent sessions
(default 64, LRU-evicted; the default session is never evicted).
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field

from actors import build_default_registry
from actors.registry import ActorRegistry
from graders.expert_panel import ExpertPanel
from models import ExpertProfile, RewardBlendConfig
from scenarios.policy_drift import PolicyDriftEngine
from server.environment import SOCEnvironment
from tools.ticketing import TicketingSystem

DEFAULT_SESSION_ID = "default"

# Session IDs are client-supplied — constrain them so they are safe to log,
# store as dict keys, and echo back in JSON.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class SessionState:
    """Everything one tenant needs for an isolated episode lifecycle."""

    session_id: str
    env: SOCEnvironment = field(default_factory=SOCEnvironment)
    actor_registry: ActorRegistry = field(default_factory=lambda: build_default_registry(seed=0))
    policy_drift: PolicyDriftEngine = field(default_factory=lambda: PolicyDriftEngine(seed=0))
    expert_panel: ExpertPanel = field(default_factory=ExpertPanel)
    ticketing: TicketingSystem = field(default_factory=TicketingSystem)
    reward_blend: RewardBlendConfig = field(default_factory=RewardBlendConfig)
    current_expert: ExpertProfile | None = None
    curriculum_round: int = 0
    actor_step: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.current_expert is None:
            self.current_expert = self.expert_panel.for_round(self.curriculum_round)

    def touch(self) -> None:
        self.last_used = time.time()

    def on_reset(self, seed: int) -> None:
        """Re-seed the v3 theme modules for a fresh episode (call under lock, after env.reset)."""
        self.actor_registry = build_default_registry(seed=seed)
        self.actor_registry.reset(seed=seed)
        self.policy_drift = PolicyDriftEngine(seed=seed)
        max_steps = self.env._config.max_steps if self.env._config else 60
        self.policy_drift.plan(max_steps=max_steps, drift_count=2)
        self.ticketing = TicketingSystem()
        self.current_expert = self.expert_panel.for_round(self.curriculum_round)
        self.actor_step = 0

    def on_step(self) -> None:
        """Advance actors, policy drift, and ticketing SLA clocks (call under lock, after env.step)."""
        self.actor_step += 1
        self.actor_registry.tick(
            step=self.actor_step,
            ctx={"policy_version": self.policy_drift.current().version},
        )
        self.policy_drift.maybe_drift(step=self.actor_step)
        self.ticketing.tick()

    def ensure_episode(self) -> None:
        """Auto-start a default episode if none is active (call under lock)."""
        if self.env._config is None:
            self.env.reset(task_id="phishing", seed=42)
            self.on_reset(seed=42)

    def summary(self) -> dict:
        st = None
        if self.env._config is not None:
            try:
                st = self.env.state().model_dump()
            except Exception:
                st = None
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "active_episode": self.env._config is not None,
            "state": st,
        }


class SessionManager:
    """Thread-safe registry of per-tenant SessionState objects."""

    def __init__(self, max_sessions: int | None = None, ttl_seconds: int | None = None) -> None:
        self.max_sessions = max_sessions if max_sessions is not None else _env_int("SOC_GYM_MAX_SESSIONS", 64)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else _env_int("SOC_GYM_SESSION_TTL", 3600)
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def validate_id(session_id: str) -> bool:
        return bool(_SESSION_ID_RE.match(session_id))

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        sid = session_id or DEFAULT_SESSION_ID
        if not self.validate_id(sid):
            raise ValueError("Invalid session ID: must be 1-64 characters of [A-Za-z0-9._-].")
        with self._lock:
            self._evict_expired_locked()
            sess = self._sessions.get(sid)
            if sess is None:
                if len(self._sessions) >= self.max_sessions:
                    self._evict_lru_locked()
                sess = SessionState(session_id=sid)
                self._sessions[sid] = sess
            sess.touch()
            return sess

    def peek(self, session_id: str | None = None) -> SessionState | None:
        """Return the session if it exists, without creating it."""
        with self._lock:
            return self._sessions.get(session_id or DEFAULT_SESSION_ID)

    def drop(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        """Remove every session (used by tests for isolation)."""
        with self._lock:
            self._sessions.clear()

    def list_summaries(self) -> list[dict]:
        with self._lock:
            self._evict_expired_locked()
            return [s.summary() for s in self._sessions.values()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- internal (call with self._lock held) --------------------------------

    def _evict_expired_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if sid != DEFAULT_SESSION_ID and now - s.last_used > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def _evict_lru_locked(self) -> None:
        candidates = [(s.last_used, sid) for sid, s in self._sessions.items() if sid != DEFAULT_SESSION_ID]
        if not candidates:
            return
        candidates.sort()
        del self._sessions[candidates[0][1]]
