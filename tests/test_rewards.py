"""Reward-signal properties.

These assert what a reward function for RL must satisfy, independently of the
exact weights: scores stay in range, effort beats inaction, informed play beats
random play, the privileged oracle bounds the observation-only heuristic, and
shaping penalties fire only on genuinely repeated actions.
"""

import pytest

from baselines import SOLO_TASKS, TEAM_TASKS, make_agent, run_episode
from models import SOCAction
from server.environment import SOCEnvironment

ALL_TASKS = SOLO_TASKS + TEAM_TASKS
SEEDS = (1, 2, 42)


def _mean(agent: str, task: str, seeds=SEEDS) -> float:
    return sum(run_episode(make_agent(agent), task, s).score for s in seeds) / len(seeds)


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("agent", ["submit", "random", "heuristic"])
def test_task_score_is_bounded(agent, task):
    for seed in SEEDS:
        result = run_episode(make_agent(agent), task, seed)
        assert 0.0 < result.score < 1.0, (agent, task, seed, result.score)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_empty_submission_earns_nothing(task):
    """Regression: inaction used to collect free efficiency / response / policy credit
    (0.24 on queue_management, 0.40 on false-positive phishing seeds)."""
    for seed in (1, 2, 3, 42, 123):
        assert run_episode(make_agent("submit"), task, seed).score <= 0.001 + 1e-9, (task, seed)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_heuristic_beats_random(task):
    assert _mean("heuristic", task) > _mean("random", task) + 0.1


@pytest.mark.parametrize("task", SOLO_TASKS)
def test_oracle_bounds_heuristic(task):
    """The ground-truth oracle is the reference ceiling; an observation-only
    policy beating it means the grader rewards something other than correctness."""
    assert _mean("oracle", task) >= _mean("heuristic", task)


@pytest.mark.parametrize("task", SOLO_TASKS)
def test_oracle_clears_the_bar(task):
    assert _mean("oracle", task) >= 0.7


def _env(task="phishing", seed=42):
    env = SOCEnvironment()
    env.reset(task, seed=seed)
    return env, env._config.alerts[0].alert_id


def test_correct_classification_outscores_wrong():
    def classify(correct: bool) -> float:
        env, aid = _env()
        src = env._config.ground_truth.relevant_log_sources[aid][0]
        env.step(SOCAction(action_type="query_logs", log_source=src, query_alert_id=aid))
        truth = env._config.ground_truth.alert_classifications[aid]
        label = truth if correct else next(c for c in type(truth) if c != truth)
        return env.step(
            SOCAction(action_type="classify_alert", alert_id=aid, classification=label, confidence=0.9)
        ).reward

    assert classify(True) > 0 > classify(False)


def test_classifying_without_evidence_is_penalised_regardless_of_label():
    """Anti-guessing: a lucky blind guess must not be rewarded."""
    env, aid = _env()
    truth = env._config.ground_truth.alert_classifications[aid]
    obs = env.step(SOCAction(action_type="classify_alert", alert_id=aid, classification=truth, confidence=0.9))
    assert obs.reward < 0


def test_distinct_techniques_are_not_loop_penalised():
    """Regression: the loop signature ignored technique_id, so mapping distinct
    techniques to one alert was treated as a repeated loop."""
    env, aid = _env("phishing", 42)
    techniques = env._config.ground_truth.expected_techniques[aid]
    assert len(techniques) >= 3
    for tech in techniques:
        obs = env.step(SOCAction(action_type="map_technique", alert_id=aid, technique_id=tech))
        assert "Loop penalty" not in obs.message, (tech, obs.message)
        assert obs.reward > 0, (tech, obs.reward)


def test_identical_action_repeats_are_penalised():
    env, aid = _env("phishing", 42)
    action = SOCAction(action_type="query_logs", log_source="firewall", query_alert_id=aid)
    rewards = [env.step(action).reward for _ in range(4)]
    assert rewards[2] < rewards[0]
    assert rewards[3] < rewards[2], "loop penalty should escalate with repeats"


def test_step_after_done_is_inert():
    env, _ = _env()
    env.step(SOCAction(action_type="submit_investigation"))
    before = env._cumulative_reward
    obs = env.step(SOCAction(action_type="noop"))
    assert obs.done and obs.reward == 0.0
    assert env._cumulative_reward == before


def test_budget_exhaustion_terminates_episode():
    env, _ = _env()
    obs = None
    for _ in range(env._config.max_steps):
        obs = env.step(SOCAction(action_type="noop"))
    assert obs.done
    assert "Budget exhausted" in obs.message
