"""
Baseline agents and an in-process episode runner.

These are the reference policies the benchmark (``benchmark.py``) reports and
the test suite uses to check that the reward signal actually separates good
play from bad. They run against ``SOCEnvironment`` directly, so no server is
needed and results are a pure function of ``(agent, task_id, seed)``.

Agents, from floor to ceiling:

* ``submit``    — submits immediately. The "do nothing" floor.
* ``random``    — uniform over role-valid action types with parameters drawn
                  from the observation. Seeded, so it is reproducible.
* ``heuristic`` — the rule-based analyst in ``baseline_agent.py`` (solo) and
                  the scripted team policy from ``train_grpo.oracle_action``
                  (team mode). Uses only what the agent can observe.
* ``oracle``    — **privileged**: reads the scenario's ground truth. It is an
                  upper-bound reference for the reward/grader design, not a
                  fair competitor. Solo mode only.

Every agent implements the same two-method protocol::

    agent.reset(task_id, seed, mode, config)   # config only used by oracle
    action_dict = agent.act(observation_dict)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from baseline_agent import HeuristicBaselineAgent
from models import (
    ActionType,
    AlertClassification,
    EpisodeMode,
    LogSource,
    ResponseActionType,
    ScenarioConfig,
    SOCAction,
)
from server.environment import SOCEnvironment

SOLO_TASKS: tuple[str, ...] = (
    "phishing",
    "ransomware",
    "cloud_compromise",
    "lateral_movement",
    "queue_management",
    "insider_threat",
    "apt_campaign",
)
TEAM_TASKS: tuple[str, ...] = ("team_phishing_escalation", "team_lateral_team")

# Hard ceiling on steps the runner will take, independent of task budgets, so
# a buggy agent can never spin forever.
_RUNNER_STEP_CAP = 1000


class Agent(Protocol):
    name: str

    def reset(self, task_id: str, seed: int, mode: str, config: ScenarioConfig | None) -> None: ...

    def act(self, obs: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class SubmitAgent:
    """Submits on the first step: the zero-effort floor."""

    name = "submit"

    def reset(self, task_id: str, seed: int, mode: str, config: ScenarioConfig | None) -> None:
        pass

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {"action_type": ActionType.SUBMIT_INVESTIGATION.value}


_ROLE_ACTIONS: dict[str | None, list[ActionType]] = {
    None: [
        ActionType.ENRICH_INDICATOR,
        ActionType.QUERY_LOGS,
        ActionType.CORRELATE_ALERTS,
        ActionType.CHECK_ASSET,
        ActionType.CHECK_USER,
        ActionType.CLASSIFY_ALERT,
        ActionType.MAP_TECHNIQUE,
        ActionType.RECOMMEND_ACTION,
        ActionType.ESCALATE,
        ActionType.SUBMIT_INVESTIGATION,
        ActionType.NOOP,
    ],
    "tier1": [
        ActionType.ENRICH_INDICATOR,
        ActionType.QUERY_LOGS,
        ActionType.CLASSIFY_ALERT,
        ActionType.MAP_TECHNIQUE,
        ActionType.ESCALATE_TO_TIER2,
        ActionType.PHASE_COMPLETE,
        ActionType.NOOP,
    ],
    "tier2": [
        ActionType.FORENSIC_TIMELINE,
        ActionType.ISOLATE_HOST,
        ActionType.BLOCK_IOC,
        ActionType.CLOSE_CASE,
        ActionType.PHASE_COMPLETE,
        ActionType.NOOP,
    ],
    "manager": [
        ActionType.REVIEW_DECISION,
        ActionType.FLAG_INCONSISTENCY,
        ActionType.EXPLAIN_TEAM_BEHAVIOR,
        ActionType.PHASE_COMPLETE,
        ActionType.SUBMIT_INVESTIGATION,
    ],
}

_TECHNIQUE_POOL = ["T1566.001", "T1059.001", "T1071.001", "T1021.002", "T1003.001", "T1041", "T1486", "T1078"]


class RandomAgent:
    """Uniform-random over role-valid action types, parameters drawn from the observation.

    The RNG is reseeded from ``(agent_seed, task_id, seed)`` on every reset, so
    a benchmark run is reproducible end to end.
    """

    name = "random"

    def __init__(self, agent_seed: int = 0) -> None:
        self._agent_seed = agent_seed
        self._rng = random.Random(agent_seed)

    def reset(self, task_id: str, seed: int, mode: str, config: ScenarioConfig | None) -> None:
        self._rng = random.Random(f"{self._agent_seed}:{task_id}:{seed}:{mode}")

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        rng = self._rng
        role = obs.get("current_role")
        alerts = obs.get("alert_queue") or [{"alert_id": "none", "indicators": {}}]
        alert = rng.choice(alerts)
        aid = alert["alert_id"]
        indicators = [(k, v) for k, vals in (alert.get("indicators") or {}).items() for v in vals]
        action_type = rng.choice(_ROLE_ACTIONS.get(role, _ROLE_ACTIONS[None]))
        action: dict[str, Any] = {"action_type": action_type.value}
        if role:
            action["role"] = role

        if action_type == ActionType.ENRICH_INDICATOR and indicators:
            kind, value = rng.choice(indicators)
            action.update(indicator=value, query_alert_id=aid)
            if kind in {"ip", "domain", "file_hash", "email", "url", "user"}:
                action["indicator_type"] = kind
        elif action_type == ActionType.QUERY_LOGS:
            action.update(log_source=rng.choice(list(LogSource)).value, query_alert_id=aid)
        elif action_type == ActionType.CORRELATE_ALERTS:
            other = rng.choice(alerts)["alert_id"]
            action.update(alert_id_a=aid, alert_id_b=other)
        elif action_type == ActionType.CHECK_ASSET:
            hosts = (alert.get("indicators") or {}).get("hostname") or ["unknown-host"]
            action["hostname"] = rng.choice(hosts)
        elif action_type == ActionType.CHECK_USER:
            users = (alert.get("indicators") or {}).get("user") or ["unknown-user"]
            action["username"] = rng.choice(users)
        elif action_type == ActionType.CLASSIFY_ALERT:
            action.update(
                alert_id=aid,
                classification=rng.choice(list(AlertClassification)).value,
                confidence=round(rng.random(), 2),
            )
        elif action_type == ActionType.MAP_TECHNIQUE:
            action.update(alert_id=aid, technique_id=rng.choice(_TECHNIQUE_POOL))
        elif action_type == ActionType.RECOMMEND_ACTION:
            action.update(alert_id=aid, response_action=rng.choice(list(ResponseActionType)).value)
        elif action_type in (ActionType.ESCALATE, ActionType.ESCALATE_TO_TIER2):
            action.update(alert_id=aid, escalation_severity="high", justification="random")
        elif action_type in (ActionType.FORENSIC_TIMELINE, ActionType.ISOLATE_HOST):
            hosts = (alert.get("indicators") or {}).get("hostname") or ["unknown-host"]
            action.update(alert_id=aid, target_host=rng.choice(hosts))
        elif action_type == ActionType.BLOCK_IOC and indicators:
            action.update(alert_id=aid, target_ioc=rng.choice(indicators)[1])
        elif action_type == ActionType.CLOSE_CASE:
            action.update(alert_id=aid, justification="random")
        elif action_type == ActionType.REVIEW_DECISION:
            tickets = obs.get("tickets") or []
            if tickets:
                action["ticket_id"] = rng.choice(tickets)["ticket_id"]
            action["alert_id"] = aid
        elif action_type == ActionType.FLAG_INCONSISTENCY:
            action.update(alert_id=aid, flag_reason="random")
        elif action_type == ActionType.EXPLAIN_TEAM_BEHAVIOR:
            action["explanation_text"] = "Random policy explanation."
        return action


class HeuristicAgent:
    """Observation-only rule-based policy (no ground truth)."""

    name = "heuristic"

    def __init__(self) -> None:
        self._solo = HeuristicBaselineAgent()
        self._mode = EpisodeMode.TIER1_SOLO.value

    def reset(self, task_id: str, seed: int, mode: str, config: ScenarioConfig | None) -> None:
        self._solo.reset()
        self._mode = mode

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._mode == EpisodeMode.TEAM.value:
            # Imported lazily: train_grpo pulls in the training package.
            from train_grpo import oracle_action

            return oracle_action(obs)
        return self._solo.next_action(obs)


class OracleAgent:
    """Privileged ground-truth policy: an upper-bound reference, solo mode only.

    Plays the grader's checklist in priority order so it degrades gracefully on
    tasks whose full checklist exceeds the step budget: classify every alert
    first, then map techniques and recommend responses for true positives,
    then gather evidence. It submits once it has used 75% of the step budget,
    the last point at which the environment's efficiency multiplier is 1.0.
    """

    name = "oracle"

    def __init__(self) -> None:
        self._plan: list[dict[str, Any]] = []
        self._actions_left = 0

    def reset(self, task_id: str, seed: int, mode: str, config: ScenarioConfig | None) -> None:
        if mode != EpisodeMode.TIER1_SOLO.value:
            raise ValueError("OracleAgent supports tier1_solo mode only")
        if config is None:
            raise ValueError("OracleAgent needs the scenario config (it reads ground truth)")
        gt = config.ground_truth
        classify: list[dict[str, Any]] = []
        techniques: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for alert in config.alerts:
            aid = alert.alert_id
            cls = gt.alert_classifications.get(aid, AlertClassification.FALSE_POSITIVE)
            classify.append(
                {"action_type": "classify_alert", "alert_id": aid, "classification": cls.value, "confidence": 0.95}
            )
            for tech in gt.expected_techniques.get(aid, []):
                techniques.append({"action_type": "map_technique", "alert_id": aid, "technique_id": tech})
            for act in gt.expected_response_actions.get(aid, []):
                responses.append({"action_type": "recommend_action", "alert_id": aid, "response_action": act.value})
            for src in gt.relevant_log_sources.get(aid, []):
                evidence.append({"action_type": "query_logs", "log_source": src.value, "query_alert_id": aid})
        # Kill-chain reconstruction is graded on correlating adjacent alerts.
        correlate: list[dict[str, Any]] = []
        chains = ([gt.kill_chain_order] if gt.kill_chain_order else []) + list(gt.attack_chain_ids)
        seen: set[frozenset[str]] = set()
        for chain in chains:
            for a, b in zip(chain, chain[1:], strict=False):
                if frozenset((a, b)) not in seen:
                    seen.add(frozenset((a, b)))
                    correlate.append({"action_type": "correlate_alerts", "alert_id_a": a, "alert_id_b": b})
        self._plan = classify + correlate + techniques + responses + evidence
        # Total steps including the final submit must stay <= 75% of budget.
        self._actions_left = max(0, int(0.75 * config.max_steps) - 1)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._actions_left <= 0 or obs.get("investigation_budget", 0) <= 1 or not self._plan:
            return {"action_type": "submit_investigation"}
        self._actions_left -= 1
        return self._plan.pop(0)


AGENTS: dict[str, type] = {
    "submit": SubmitAgent,
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "oracle": OracleAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return AGENTS[name]()
    except KeyError:
        raise ValueError(f"Unknown agent '{name}'. Valid: {sorted(AGENTS)}") from None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    agent: str
    task_id: str
    seed: int
    mode: str
    score: float
    steps: int
    cumulative_reward: float
    submitted: bool
    step_rewards: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "task_id": self.task_id,
            "seed": self.seed,
            "mode": self.mode,
            "score": self.score,
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward,
            "submitted": self.submitted,
        }


def run_episode(agent: Agent, task_id: str, seed: int, mode: str | None = None) -> EpisodeResult:
    """Play one episode in-process and return its result.

    ``score`` is the terminal task score in ``[0.001, 0.999]``. When the agent
    submits it is the environment's own ``task_score``. When the step budget
    runs out first, the grader score is scaled by the same efficiency
    multiplier a submission at that step would get, so timing out is never
    better than submitting.
    """
    if mode is None:
        mode = EpisodeMode.TEAM.value if task_id in TEAM_TASKS else EpisodeMode.TIER1_SOLO.value
    env = SOCEnvironment()
    obs = env.reset(task_id, seed=seed, mode=mode)
    agent.reset(task_id, seed, mode, env._config.model_copy(deep=True) if env._config else None)

    step_rewards: list[float] = []
    steps = 0
    while not obs.done and steps < _RUNNER_STEP_CAP:
        action = SOCAction.model_validate(agent.act(obs.model_dump(mode="json")))
        obs = env.step(action)
        step_rewards.append(float(obs.reward))
        steps += 1

    submitted = obs.task_score is not None
    if submitted:
        score = float(obs.task_score)
    else:
        score = max(0.001, min(0.999, env.grade() * env._efficiency_multiplier()))
    return EpisodeResult(
        agent=agent.name,
        task_id=task_id,
        seed=seed,
        mode=mode,
        score=round(score, 6),
        steps=steps,
        cumulative_reward=round(float(env._cumulative_reward), 6),
        submitted=submitted,
        step_rewards=step_rewards,
    )
