"""Held-out eval diagnostics and the do-nothing attractor in team mode.

The published GRPO adapter scored 0.001 on all 15 held-out episodes. These
tests pin down why that number alone explains nothing, and make sure the eval
now records what the model actually did.
"""

import json

import pytest

from models import SOCAction
from server.environment import SOCEnvironment
from train_grpo import oracle_action
from training.evaluation import diagnose_text_policy, run_text_policy_episode

SEEDS = [100, 101, 102]


def _first_alert(obs):
    return obs["alert_queue"][0]["alert_id"]


TEXT_POLICIES = {
    "prose": lambda obs, step: "I will analyse the alert carefully before deciding.",
    "noop": lambda obs, step: '{"action_type": "noop"}',
    "phase_complete": lambda obs, step: '{"action_type": "phase_complete"}',
    "bad_enum": lambda obs, step: json.dumps(
        {"action_type": "classify_alert", "alert_id": _first_alert(obs), "classification": "malicious"}
    ),
    "oracle_as_text": lambda obs, step: "```json\n" + json.dumps(oracle_action(obs)) + "\n```",
}


def _run(test_client, name):
    return [
        run_text_policy_episode(test_client, "team_phishing_escalation", s, "tier1", TEXT_POLICIES[name], oracle_action)
        for s in SEEDS
    ]


def test_text_policy_eval_matches_oracle_when_text_is_good(test_client):
    eps = _run(test_client, "oracle_as_text")
    assert all(ep["score"] >= 0.99 for ep in eps)
    diag = diagnose_text_policy(eps)
    assert diag["episodes_escalated"] == len(SEEDS)
    assert set(diag["parse"]) == {"loose"}


@pytest.mark.parametrize("name", ["prose", "noop", "phase_complete"])
def test_failure_modes_share_a_score_but_not_a_diagnosis(test_client, name):
    eps = _run(test_client, name)
    assert all(ep["score"] <= 0.001 + 1e-9 for ep in eps), "all of these hit the score floor"
    diag = diagnose_text_policy(eps)
    assert diag["episodes_escalated"] == 0
    # ...but the trace tells them apart.
    if name == "prose":
        assert diag["parse"] == {"fallback": diag["policy_steps"]}
        assert diag["action_types"] == {"noop": diag["policy_steps"]}
    else:
        assert set(diag["parse"]) == {"strict"}
        assert set(diag["action_types"]) == {name}


def test_rejected_action_is_recorded_not_fatal(test_client):
    """A 422 used to abort the episode (score 0.0). Now it's counted and play continues."""
    eps = _run(test_client, "bad_enum")
    for ep in eps:
        assert ep["rejected"] > 1, "episode should continue past the first rejected action"
        assert ep["trace"][0]["status"] == 422
    assert diagnose_text_policy(eps)["rejected_actions"] == sum(ep["rejected"] for ep in eps)


def test_trace_keeps_raw_completion(test_client):
    ep = _run(test_client, "prose")[0]
    assert ep["trace"][0]["text"].startswith("I will analyse")
    assert "message" in ep["trace"][0]


def test_zero_work_phase_complete_is_penalised_at_the_step():
    """Regression: the -0.10 penalty went only into the running total, so the
    step reward a per-step trainer sees was +0.001, a do-nothing attractor."""
    env = SOCEnvironment()
    env.reset("team_phishing_escalation", seed=100, mode="team")
    obs = env.step(SOCAction(action_type="phase_complete", role="tier1"))
    assert obs.done
    assert obs.reward < 0
    noop_env = SOCEnvironment()
    noop_env.reset("team_phishing_escalation", seed=100, mode="team")
    assert obs.reward < noop_env.step(SOCAction(action_type="noop", role="tier1")).reward
    assert env._cumulative_reward == pytest.approx(obs.reward)


def test_no_penalty_when_no_escalation_was_required():
    env = SOCEnvironment()
    env.reset("team_phishing_escalation", seed=100, mode="team")
    env._config.ground_truth.required_escalations = []
    obs = env.step(SOCAction(action_type="phase_complete", role="tier1"))
    assert obs.reward > 0
    assert "penalty" not in obs.message.split("|")[0]
