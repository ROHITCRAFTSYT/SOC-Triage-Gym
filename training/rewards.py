"""
Reward computation for GRPO training
====================================

Three layers:

1. **Parsing** — ``classify_parse_quality`` / ``parse_action_from_text``
   turn raw model completions into env actions and tag how cleanly they
   parsed (the strongest early-training learning signal for small models).
2. **Scoring** — ``score_completion`` replays the environment to the
   completion's exact state, applies the action, and blends the env's
   step reward with the JSON-validity shaping bonus.
3. **Throughput** — ``ParallelRewardEvaluator`` scores a whole GRPO
   completion group concurrently, one isolated server session per worker
   (the v0.2.0 multi-session server makes this safe). Reward evaluation is
   the wall-clock bottleneck of env-in-the-loop GRPO — with group size 8,
   parallel scoring cuts reward latency roughly by ``min(workers, 8)×``.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Shaping bonuses applied on top of the env's per-step reward. Tuned so that
# strict JSON over a noop-fallback is worth ~0.15 — enough to dominate small
# noisy env rewards early in training but small enough that a strict-JSON
# malicious action still loses to a loose-JSON correct action.
PARSE_BONUS = {"strict": 0.05, "loose": 0.01, "fallback": -0.10}


def classify_parse_quality(text: str) -> str:
    """Tag a model completion by how cleanly it parses as a JSON action.

    Returns one of:
      "strict"   — entire completion is one valid JSON object with action_type
      "loose"    — JSON object with action_type recovered via fenced/regex extraction
      "fallback" — no parseable JSON; intent inferred from keywords (or noop)
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return "strict"
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```", r"(\{[\s\S]*\})"):
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "action_type" in obj:
                return "loose"
        except json.JSONDecodeError:
            continue
    return "fallback"


def parse_action_from_text(text: str, role: str) -> dict | None:
    """Extract a JSON action from model output text (keyword fallback → noop)."""
    patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
        r"(\{[\s\S]*?\})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                action = json.loads(m.group(1))
                if "action_type" in action:
                    action["role"] = role
                    return action
            except json.JSONDecodeError:
                continue
    text_lower = (text or "").lower()
    for keyword, action_type in [
        ("escalate_to_tier2", "escalate_to_tier2"),
        ("classify", "classify_alert"),
        ("enrich", "enrich_indicator"),
        ("isolate", "isolate_host"),
        ("block_ioc", "block_ioc"),
        ("review", "review_decision"),
        ("explain", "explain_team_behavior"),
        ("phase_complete", "phase_complete"),
    ]:
        if keyword in text_lower:
            return {"action_type": action_type, "role": role}
    return {"action_type": "noop", "role": role}


def score_completion(
    client,
    *,
    text: str,
    role: str,
    task_id: str,
    seed: int,
    step_index: int,
    replay_fn: Callable,
) -> tuple[float, str]:
    """Score one completion: replay env to the prompt's state, apply the action.

    Args:
        client: httpx-compatible client bound to the env server (its session
            header determines which server session the replay runs in).
        text: raw model completion.
        role: role being trained (tier1/tier2/manager).
        task_id / seed / step_index: identify the state the prompt was drawn from.
        replay_fn: ``fn(client, task_id, seed, step_index) -> obs`` that resets
            and replays oracle actions to the target step.

    Returns:
        (reward, parse_quality) — env step reward + shaping bonus, and the
        parse-quality tag for telemetry.
    """
    quality = classify_parse_quality(text)
    shaping = PARSE_BONUS[quality]

    try:
        action = parse_action_from_text(text, role)
        if action is None:
            return (-0.05 + shaping, quality)

        obs = replay_fn(client, task_id, seed, step_index)
        if obs.get("done", False):
            return (0.0 + shaping, quality)

        # Guard against role mismatch: replay may have advanced past the
        # target role if the oracle completed the phase.
        acting_role = obs.get("current_role") or "tier1"
        if acting_role != role:
            return (-0.02 + shaping, quality)

        step_resp = client.post(
            "/step",
            content=json.dumps(action),
            headers={"Content-Type": "application/json"},
        )
        if step_resp.status_code != 200:
            return (-0.05 + shaping, quality)
        stepped = step_resp.json()
        return (float(stepped.get("reward", 0.0)) + shaping, quality)
    except Exception:
        return (0.0 + shaping, quality)


@dataclass
class CompletionItem:
    """One completion to score: the text plus the state it was sampled from."""

    text: str
    task_id: str
    seed: int
    step_index: int


class ParallelRewardEvaluator:
    """Scores completion groups concurrently across isolated server sessions.

    Each worker thread owns a dedicated client whose ``X-Session-ID`` maps to
    an isolated environment on the server, so concurrent replays never
    corrupt each other. Requires a server >= 0.2.0 (multi-session).

    Args:
        client_factory: ``fn(session_id) -> httpx-compatible client``. Called
            once per worker; the returned client must route requests into the
            given session (e.g. via an ``X-Session-ID`` default header).
        workers: number of concurrent scoring threads (1 = sequential).
        session_prefix: worker sessions are named ``{prefix}-{i}``.
    """

    def __init__(
        self,
        client_factory: Callable[[str], object],
        workers: int = 4,
        session_prefix: str = "grpo-worker",
    ) -> None:
        self.workers = max(1, int(workers))
        self._local = threading.local()
        self._factory = client_factory
        self._prefix = session_prefix
        self._counter = 0
        self._counter_lock = threading.Lock()

    def _worker_client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            with self._counter_lock:
                worker_id = self._counter
                self._counter += 1
            client = self._factory(f"{self._prefix}-{worker_id}")
            self._local.client = client
        return client

    def score_batch(
        self,
        items: list[CompletionItem],
        role: str,
        replay_fn: Callable,
    ) -> tuple[list[float], dict[str, int]]:
        """Score every item; returns (rewards in input order, parse-quality counts)."""
        quality_counts = {"strict": 0, "loose": 0, "fallback": 0}
        counts_lock = threading.Lock()

        def _score(item: CompletionItem) -> float:
            reward, quality = score_completion(
                self._worker_client(),
                text=item.text,
                role=role,
                task_id=item.task_id,
                seed=item.seed,
                step_index=item.step_index,
                replay_fn=replay_fn,
            )
            with counts_lock:
                quality_counts[quality] += 1
            return reward

        if self.workers == 1:
            rewards = [_score(item) for item in items]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                rewards = list(pool.map(_score, items))
        return rewards, quality_counts
