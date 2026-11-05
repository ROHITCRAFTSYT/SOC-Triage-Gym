"""
Held-out policy evaluation
==========================

Runs full team episodes on **disjoint seeds** (never seen in training) with
the candidate policy driving the trained role and the oracle driving the
other roles — the same protocol as ``benchmark.py``, packaged as a callable
so the training loop can evaluate periodically, gate best-checkpoint saves,
and stop early on plateau.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable

# Seeds 100+ are the project's held-out band (training uses 42..91).
DEFAULT_EVAL_SEEDS = list(range(100, 110))


def run_policy_episode(
    client,
    task_id: str,
    seed: int,
    role: str,
    policy_fn: Callable[[dict], dict],
    oracle_fn: Callable[[dict], dict],
    max_steps: int = 80,
) -> dict:
    """One team episode: ``policy_fn`` acts for ``role``, oracle for the rest.

    Returns {"task_id", "seed", "score", "steps", "policy_steps"}.
    """
    reset_resp = client.post("/reset", json={"task_id": task_id, "seed": seed, "mode": "team"})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    step = 0
    policy_steps = 0
    while not obs.get("done", False) and step < max_steps:
        step += 1
        acting_role = obs.get("current_role") or "tier1"
        if acting_role == role:
            action = policy_fn(obs)
            policy_steps += 1
        else:
            action = oracle_fn(obs)
        step_resp = client.post(
            "/step",
            content=json.dumps(action),
            headers={"Content-Type": "application/json"},
        )
        if step_resp.status_code != 200:
            break
        obs = step_resp.json()

    score = float(obs.get("task_score") or obs.get("cumulative_reward", 0.0))
    return {"task_id": task_id, "seed": seed, "score": score, "steps": step, "policy_steps": policy_steps}


def run_text_policy_episode(
    client,
    task_id: str,
    seed: int,
    role: str,
    generate_fn: Callable[[dict, int], str],
    oracle_fn: Callable[[dict], dict],
    max_steps: int = 80,
    max_text_chars: int = 2000,
) -> dict:
    """One team episode driven by a *text* policy, with a per-step trace.

    ``generate_fn(obs, step) -> completion`` is the model (prompt formatting
    lives inside it). The completion is parsed exactly as during training, and
    every policy step is recorded so a bad score can be explained afterwards.
    The first held-out eval scored 0.001 on all 15 episodes and kept nothing
    but the score, and at least five different failure modes produce exactly
    that number.

    A rejected action (non-200, e.g. an invalid enum value) is recorded and the
    episode continues. Before, it aborted the episode, scoring it 0.0 for one
    bad completion.
    """
    from training.rewards import classify_parse_quality, parse_action_from_text

    obs = client.post("/reset", json={"task_id": task_id, "seed": seed, "mode": "team"}).json()
    trace: list[dict] = []
    step = 0
    while not obs.get("done") and step < max_steps:
        step += 1
        acting_role = obs.get("current_role") or "tier1"
        if acting_role == role:
            text = generate_fn(obs, step)
            action = parse_action_from_text(text, role) or {"action_type": "noop", "role": role}
            record = {
                "step": step,
                "parse": classify_parse_quality(text),
                "action_type": action.get("action_type"),
                "text": (text or "")[:max_text_chars],
            }
        else:
            action, record = oracle_fn(obs), None
        resp = client.post("/step", content=json.dumps(action), headers={"Content-Type": "application/json"})
        if record is not None:
            record["status"] = resp.status_code
            if resp.status_code == 200:
                body = resp.json()
                record["reward"] = body.get("reward")
                record["message"] = (body.get("message") or "")[:200]
            trace.append(record)
        if resp.status_code == 200:
            obs = resp.json()

    score = float(obs.get("task_score") or obs.get("cumulative_reward", 0.0))
    return {
        "task_id": task_id,
        "seed": seed,
        "score": score,
        "steps": step,
        "policy_steps": len(trace),
        "rejected": sum(1 for r in trace if r["status"] != 200),
        "escalated": any(r["action_type"] == "escalate_to_tier2" and r["status"] == 200 for r in trace),
        "trace": trace,
    }


def diagnose_text_policy(episodes: list[dict]) -> dict:
    """Aggregate per-step traces into the numbers that explain a score.

    ``parse`` shows whether the model emits usable JSON at all,
    ``action_types`` shows what it chooses, and ``episodes_escalated`` is what
    team tasks hinge on: no escalation means the 0.001 score floor.
    """
    parse: dict[str, int] = {}
    actions: dict[str, int] = {}
    for ep in episodes:
        for rec in ep.get("trace", []):
            parse[rec["parse"]] = parse.get(rec["parse"], 0) + 1
            actions[rec["action_type"]] = actions.get(rec["action_type"], 0) + 1
    return {
        "policy_steps": sum(parse.values()),
        "parse": dict(sorted(parse.items())),
        "action_types": dict(sorted(actions.items(), key=lambda kv: -kv[1])),
        "rejected_actions": sum(ep.get("rejected", 0) for ep in episodes),
        "episodes_escalated": sum(1 for ep in episodes if ep.get("escalated")),
        "n_episodes": len(episodes),
    }


def evaluate_policy(
    client,
    role: str,
    policy_fn: Callable[[dict], dict],
    oracle_fn: Callable[[dict], dict],
    tasks: list[str],
    seeds: list[int] | None = None,
    max_steps: int = 80,
) -> dict:
    """Evaluate a policy across tasks × held-out seeds.

    Returns a report::

        {
          "role", "tasks", "seeds", "n_episodes",
          "mean_score", "std_score", "min_score", "max_score",
          "per_task": {task: mean},
          "episodes": [per-episode records],
        }
    """
    seeds = seeds if seeds is not None else DEFAULT_EVAL_SEEDS
    episodes = []
    for task_id in tasks:
        for seed in seeds:
            try:
                episodes.append(run_policy_episode(client, task_id, seed, role, policy_fn, oracle_fn, max_steps))
            except Exception as exc:  # network hiccup on one episode shouldn't kill eval
                episodes.append(
                    {"task_id": task_id, "seed": seed, "score": 0.0, "steps": 0, "policy_steps": 0, "error": str(exc)}
                )

    scores = [e["score"] for e in episodes]
    per_task: dict[str, float] = {}
    for task_id in tasks:
        task_scores = [e["score"] for e in episodes if e["task_id"] == task_id]
        if task_scores:
            per_task[task_id] = sum(task_scores) / len(task_scores)

    return {
        "role": role,
        "tasks": tasks,
        "seeds": seeds,
        "n_episodes": len(episodes),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "per_task": per_task,
        "episodes": episodes,
    }
