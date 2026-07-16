"""
Episode audit trail
===================

Every SOC decision must be defensible after the fact — that applies to AI
agents too. This module records a per-episode trace of every reset and step
(action taken, reward received, running totals, timestamps) and serves it
back for replay, compliance review, and offline analysis:

  * ``GET /episodes``                      — list recorded episodes
  * ``GET /episodes/{episode_id}/trace``   — full trace as JSON
  * ``GET /episodes/{episode_id}/trace?format=jsonl`` — newline-delimited
    export suitable for piping into SIEM / data-lake ingestion.

The store is in-memory and bounded (``SOC_GYM_AUDIT_MAX_EPISODES``, default
200 episodes; oldest evicted first), so it cannot grow without limit. Set
``SOC_GYM_AUDIT_DIR`` to additionally append each event to a JSONL file per
episode for durable storage.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _max_episodes() -> int:
    try:
        return max(1, int(os.environ.get("SOC_GYM_AUDIT_MAX_EPISODES", "200")))
    except (TypeError, ValueError):
        return 200


@dataclass
class EpisodeTrace:
    episode_id: str
    session_id: str
    task_id: str
    seed: int
    mode: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    final_reward: float | None = None
    events: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "final_reward": self.final_reward,
            "event_count": len(self.events),
            "done": self.finished_at is not None,
        }


class AuditTrail:
    """Bounded, thread-safe store of episode traces."""

    def __init__(self, max_episodes: int | None = None) -> None:
        self.max_episodes = max_episodes if max_episodes is not None else _max_episodes()
        self._traces: OrderedDict[str, EpisodeTrace] = OrderedDict()
        self._lock = threading.Lock()

    # -- recording ------------------------------------------------------------

    def start_episode(self, episode_id: str, session_id: str, task_id: str, seed: int, mode: str) -> None:
        with self._lock:
            trace = EpisodeTrace(
                episode_id=episode_id,
                session_id=session_id,
                task_id=task_id,
                seed=seed,
                mode=mode,
            )
            trace.events.append(
                {
                    "type": "reset",
                    "ts": trace.started_at,
                    "task_id": task_id,
                    "seed": seed,
                    "mode": mode,
                }
            )
            self._traces[episode_id] = trace
            self._traces.move_to_end(episode_id)
            while len(self._traces) > self.max_episodes:
                self._traces.popitem(last=False)
        self._persist(trace, trace.events[-1])

    def record_step(
        self,
        episode_id: str,
        step: int,
        action: dict,
        reward: float,
        cumulative_reward: float,
        done: bool,
        role: str | None = None,
    ) -> None:
        event = {
            "type": "step",
            "ts": time.time(),
            "step": step,
            "role": role,
            "action": action,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "done": done,
        }
        trace = None
        with self._lock:
            trace = self._traces.get(episode_id)
            if trace is None:
                return
            trace.events.append(event)
            if done:
                trace.finished_at = event["ts"]
                trace.final_reward = cumulative_reward
        self._persist(trace, event)

    # -- reading --------------------------------------------------------------

    def list_episodes(self, session_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            traces = list(self._traces.values())
        if session_id is not None:
            traces = [t for t in traces if t.session_id == session_id]
        traces.sort(key=lambda t: t.started_at, reverse=True)
        return [t.summary() for t in traces[:limit]]

    def get(self, episode_id: str) -> EpisodeTrace | None:
        with self._lock:
            return self._traces.get(episode_id)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    # -- durable export (optional) ---------------------------------------------

    def _persist(self, trace: EpisodeTrace | None, event: dict) -> None:
        audit_dir = os.environ.get("SOC_GYM_AUDIT_DIR", "").strip()
        if not audit_dir or trace is None:
            return
        try:
            os.makedirs(audit_dir, exist_ok=True)
            path = os.path.join(audit_dir, f"{trace.episode_id}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"episode_id": trace.episode_id, **event}) + "\n")
        except OSError:
            logger.warning("Failed writing audit event to %s", audit_dir, exc_info=True)


AUDIT = AuditTrail()


def trace_to_jsonl(trace: EpisodeTrace) -> str:
    """Render a trace as newline-delimited JSON (one event per line)."""
    header = {"type": "episode", **trace.summary()}
    lines = [json.dumps(header)]
    lines.extend(json.dumps({"episode_id": trace.episode_id, **e}) for e in trace.events)
    return "\n".join(lines) + "\n"
