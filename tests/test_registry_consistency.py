"""Registry consistency — every advertised task must be playable and gradable.

Regression guard: ``ransomware`` and ``cloud_compromise`` were registered as
scenarios and advertised in ``openenv.yaml`` / the README, but had no grader,
so every episode on them silently scored 0.0 with "No grader registered".
"""

import os

import pytest
import yaml

from graders import GRADER_REGISTRY
from scenarios import SCENARIO_REGISTRY
from server.app import TASKS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _openenv_task_ids() -> list[str]:
    with open(os.path.join(ROOT, "openenv.yaml"), encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return [t["id"] for t in spec["tasks"]]


@pytest.mark.parametrize("task_id", sorted(SCENARIO_REGISTRY))
def test_every_scenario_has_a_grader(task_id):
    assert task_id in GRADER_REGISTRY, f"scenario '{task_id}' has no grader: it can never score above 0"


def test_openenv_tasks_are_gradable():
    missing = [t for t in _openenv_task_ids() if t not in GRADER_REGISTRY]
    assert not missing, f"openenv.yaml advertises ungradable tasks: {missing}"


def test_http_task_list_matches_openenv_spec():
    assert {t["id"] for t in TASKS} == set(_openenv_task_ids())


@pytest.mark.parametrize("task_id", ["ransomware", "cloud_compromise"])
def test_single_alert_tasks_reach_full_score_with_ground_truth(task_id):
    """A ground-truth investigation on the single-alert tasks must max every grader component."""
    from models import SOCAction
    from server.environment import SOCEnvironment

    for seed in (1, 2, 3, 42):
        env = SOCEnvironment()
        env.reset(task_id, seed=seed)
        cfg = env._config
        gt = cfg.ground_truth
        alert = cfg.alerts[0]
        aid = alert.alert_id
        for src in gt.relevant_log_sources.get(aid, []):
            env.step(SOCAction(action_type="query_logs", log_source=src, query_alert_id=aid))
        env.step(
            SOCAction(
                action_type="classify_alert",
                alert_id=aid,
                classification=gt.alert_classifications[aid],
                confidence=0.9,
            )
        )
        for tech in gt.expected_techniques.get(aid, []):
            env.step(SOCAction(action_type="map_technique", alert_id=aid, technique_id=tech))
        for act in gt.expected_response_actions.get(aid, []):
            env.step(SOCAction(action_type="recommend_action", alert_id=aid, response_action=act))
        # Grader score before submit: every component must be fully achievable.
        score, breakdown, feedback = env.grade_with_breakdown()
        assert score >= 0.99, (task_id, seed, breakdown, feedback)
        assert all(v == 1.0 for v in breakdown.values()), (task_id, seed, breakdown)
        obs = env.step(SOCAction(action_type="submit_investigation"))
        assert obs.done
        assert obs.task_score is not None, obs.message
