"""Input validation on the environment API.

Malformed requests must fail fast with a 4xx and a useful message, never a 500
and never a silently coerced value that changes the episode.
"""

import pytest
from pydantic import ValidationError

from models import SOCAction
from server.environment import SOCEnvironment

# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"task_id": "phishing", "seed": -1},
        {"task_id": "phishing", "seed": 2**32},
        {"task_id": "phishing", "seed": "42"},  # strict: no str -> int coercion
        {"task_id": "phishing", "seed": 1.5},
        {"task_id": "phishing", "seed": True},
        {"task_id": "", "seed": 1},
        {"task_id": "x" * 65, "seed": 1},
        {"task_id": "phishing", "mode": "solo"},
    ],
)
def test_reset_rejects_bad_input_with_422(test_client, body):
    r = test_client.post("/reset", json=body)
    assert r.status_code == 422, r.text


def test_reset_unknown_task_is_400_with_valid_list(test_client):
    r = test_client.post("/reset", json={"task_id": "nope", "seed": 1})
    assert r.status_code == 400
    assert "phishing" in r.json()["detail"]


def test_reset_accepts_seed_bounds(test_client):
    for seed in (0, 2**32 - 1):
        assert test_client.post("/reset", json={"task_id": "phishing", "seed": seed}).status_code == 200


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "not_an_action"},
        {"action_type": "classify_alert", "alert_id": "A", "classification": "maybe"},
        {"action_type": "classify_alert", "alert_id": "A", "classification": "true_positive", "confidence": 1.5},
        {"action_type": "query_logs", "log_source": "firewall", "time_window_hours": 0},
        {"action_type": "query_logs", "log_source": "firewall", "time_window_hours": 10_000},
        {"action_type": "enrich_indicator", "indicator": "a" * 5000},
        {"action_type": "classify_alert", "alert_id": "A" * 500},
    ],
)
def test_step_rejects_bad_actions_with_422(test_client, action):
    test_client.post("/reset", json={"task_id": "phishing", "seed": 1})
    r = test_client.post("/step", json=action)
    assert r.status_code == 422, r.text


def test_step_ignores_unknown_keys_from_llm_policies(test_client):
    """LLM agents often add 'reasoning'; that must not turn a valid action into a 422."""
    test_client.post("/reset", json={"task_id": "phishing", "seed": 1})
    r = test_client.post("/step", json={"action_type": "noop", "reasoning": "thinking out loud"})
    assert r.status_code == 200


def test_generate_scenario_rejects_out_of_range(test_client):
    r = test_client.post("/generate_scenario", json={"difficulty_floor": 1.5})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [-1, "42", 1.0, True, None])
def test_env_reset_rejects_bad_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        SOCEnvironment().reset("phishing", seed=seed)


def test_env_reset_rejects_bad_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        SOCEnvironment().reset("phishing", seed=1, mode="solo")


def test_failed_reset_leaves_previous_episode_intact():
    env = SOCEnvironment()
    env.reset("phishing", seed=1)
    before = env.state()
    with pytest.raises(ValueError):
        env.reset("does_not_exist", seed=1, mode="team")
    after = env.state()
    assert after == before
    assert env._mode.value == "tier1_solo"


def test_env_reset_accepts_enum_mode():
    from models import EpisodeMode

    obs = SOCEnvironment().reset("team_phishing_escalation", seed=1, mode=EpisodeMode.TEAM)
    assert obs.episode_mode == EpisodeMode.TEAM
    assert "Mode: team." in obs.message


def test_socaction_bounds():
    with pytest.raises(ValidationError):
        SOCAction(action_type="explain_team_behavior", explanation_text="x" * 40_000)
    SOCAction(action_type="explain_team_behavior", explanation_text="x" * 30_000)
