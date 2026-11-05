"""LLM-policy prompts (training/prompts.py).

The first GRPO adapter was trained on a prompt that hid the agent's own
progress and the environment's feedback, listed no action fields, and showed
only IP indicators. These tests pin down that the prompt now carries what a
policy needs, stays inside the trainer's prompt budget, and can't drift from
the API.
"""

import json
import re

import pytest

from models import AlertClassification, IndicatorType, LogSource, SOCAction
from server.environment import _MANAGER_ACTIONS, _TIER1_ACTIONS, _TIER2_ACTIONS, SOCEnvironment
from train_grpo import oracle_action
from training.prompts import (
    ROLE_SYSTEM_PROMPTS,
    PromptBudgetError,
    check_prompt_budget,
    format_obs_prompt,
)

ROLE_ACTIONS = {"tier1": _TIER1_ACTIONS, "tier2": _TIER2_ACTIONS, "manager": _MANAGER_ACTIONS}
ALL_ACTION_NAMES = {a.value for actions in ROLE_ACTIONS.values() for a in actions}


def _team_obs(task="team_lateral_team", seed=100):
    env = SOCEnvironment()
    return env, env.reset(task, seed=seed, mode="team").model_dump(mode="json")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", sorted(ROLE_SYSTEM_PROMPTS))
def test_system_prompt_only_offers_role_valid_actions(role):
    named = {w for w in re.findall(r"[a-z_]+", ROLE_SYSTEM_PROMPTS[role]) if w in ALL_ACTION_NAMES}
    allowed = {a.value for a in ROLE_ACTIONS[role]}
    assert named, "system prompt names no actions"
    assert named <= allowed, f"{role} prompt offers actions the env rejects for it: {named - allowed}"


@pytest.mark.parametrize("role", sorted(ROLE_SYSTEM_PROMPTS))
def test_system_prompt_example_is_a_valid_action(role):
    example = ROLE_SYSTEM_PROMPTS[role].split("Example:", 1)[1].strip()
    action = SOCAction.model_validate(json.loads(example))
    assert action.action_type in ROLE_ACTIONS[role]


def test_tier1_prompt_lists_exact_enum_values():
    p = ROLE_SYSTEM_PROMPTS["tier1"]
    for enum_cls in (IndicatorType, LogSource):
        assert "|".join(m.value for m in enum_cls) in p
    classes = [m.value for m in AlertClassification if m.value != "unclassified"]
    assert "|".join(classes) in p
    assert "unclassified" not in p, "agents must not be invited to classify as unclassified"


def test_tier1_prompt_does_not_offer_submit():
    """In team mode tier-1 submit_investigation ends the whole episode (score 0.001)."""
    assert "submit_investigation" not in ROLE_SYSTEM_PROMPTS["tier1"]
    assert "escalate_to_tier2" in ROLE_SYSTEM_PROMPTS["tier1"]


# ---------------------------------------------------------------------------
# Observation prompt
# ---------------------------------------------------------------------------


def test_prompt_shows_progress_and_feedback():
    env, obs = _team_obs("team_phishing_escalation")
    alert = obs["alert_queue"][0]
    before = format_obs_prompt(obs, "tier1")
    assert "new" in before and "1 need action" in before

    ip = alert["indicators"]["ip"][0]
    env.step(SOCAction(action_type="enrich_indicator", indicator=ip, indicator_type="ip", role="tier1"))
    truth = env._config.ground_truth.alert_classifications[alert["alert_id"]]
    obs = env.step(
        SOCAction(
            action_type="classify_alert", alert_id=alert["alert_id"], classification=truth, confidence=0.9, role="tier1"
        )
    ).model_dump(mode="json")
    after = format_obs_prompt(obs, "tier1")
    assert f"class={truth.value}" in after
    assert "enriched=1" in after
    assert "Last result: Classified" in after
    assert "[Blended reward" not in after, "internal reward breakdown should be stripped"


def test_escalated_true_positive_moves_to_done_for_tier1():
    env, obs = _team_obs("team_phishing_escalation")
    aid = obs["alert_queue"][0]["alert_id"]
    env.step(
        SOCAction(action_type="enrich_indicator", indicator=obs["alert_queue"][0]["indicators"]["ip"][0], role="tier1")
    )
    env.step(SOCAction(action_type="classify_alert", alert_id=aid, classification="true_positive", role="tier1"))
    obs = env.step(SOCAction(action_type="escalate_to_tier2", alert_id=aid, role="tier1")).model_dump(mode="json")
    prompt = format_obs_prompt(obs, "tier1")
    assert "0 need action" in prompt
    assert f"Done: {aid}=true_positive+escalated" in prompt
    # Tier-2 sees the ticket the escalation created, and the alert as its work.
    tier2_obs = env.step(SOCAction(action_type="phase_complete", role="tier1")).model_dump(mode="json")
    tier2 = format_obs_prompt(tier2_obs, "tier2")
    assert "TKT-EP0001 escalation" in tier2
    assert "1 need action" in tier2


def test_non_ip_indicators_are_visible():
    obs = {
        "alert_queue": [
            {
                "alert_id": "A1",
                "severity": "high",
                "title": "t",
                "indicators": {"hostname": ["WS-042"], "user": ["jdoe"], "file_hash": ["f" * 64]},
            }
        ],
        "investigations": {"A1": {"alert_id": "A1", "escalated": True}},
    }
    tier1 = format_obs_prompt(obs, "tier1")
    assert "hostname=WS-042" in tier1 and "user=jdoe" in tier1
    assert "file_hash" not in tier1, "hashes cost ~40 tokens and only tier-2 acts on them"
    assert "file_hash=" in format_obs_prompt(obs, "tier2")


def test_step_comes_from_the_environment_not_the_caller():
    """Training passed 0-based steps and eval 1-based; the prompt now ignores both."""
    _, obs = _team_obs()
    assert format_obs_prompt(obs, "tier1", 0) == format_obs_prompt(obs, "tier1", 7) == format_obs_prompt(obs, "tier1")


def test_pending_alerts_are_capped_and_overflow_is_named():
    _, obs = _team_obs("team_lateral_team")
    prompt = format_obs_prompt(obs, "tier1")
    assert "8 need action" in prompt
    assert prompt.count("\n- ALT-") == 5
    assert "(+3 more pending:" in prompt


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

# Measured with the Qwen2.5 BPE vocab over 2,113 prompts (oracle + random
# rollouts, both team tasks, seeds 100-129): longest 676 tokens / 1,986 chars,
# against max_prompt_length=768. A character cap can't *guarantee* the token
# budget (chars/token ranged 2.58-3.9), so this is a growth tripwire; the hard
# guarantee is check_prompt_budget() with the real tokenizer at dataset build.
CHAR_TRIPWIRE = 2100


def _rollout_prompts(task, seed):
    env, obs = _team_obs(task, seed)
    for _ in range(80):
        if obs["done"]:
            return
        role = obs.get("current_role") or "tier1"
        yield ROLE_SYSTEM_PROMPTS[role], format_obs_prompt(obs, role)
        obs = env.step(SOCAction.model_validate(oracle_action(obs))).model_dump(mode="json")


@pytest.mark.parametrize("task", ["team_phishing_escalation", "team_lateral_team"])
def test_prompt_size_tripwire(task):
    longest = max(len(s) + len(u) for seed in range(100, 106) for s, u in _rollout_prompts(task, seed))
    assert longest <= CHAR_TRIPWIRE, f"prompt grew to {longest} chars; re-measure tokens before raising the cap"


class _FakeTokenizer:
    """~3 chars per token, enough to exercise the budget check."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(m["content"] for m in messages)
        return list(range(len(text) // 3 + 10))


def test_check_prompt_budget_passes_and_reports_longest():
    rows = [{"prompt": [{"role": "system", "content": "x" * 300}, {"role": "user", "content": "y" * 300}]}]
    assert check_prompt_budget(rows, _FakeTokenizer(), 768) == 210


def test_check_prompt_budget_fails_loudly():
    rows = [{"prompt": [{"role": "user", "content": "y" * 3000}]}]
    with pytest.raises(PromptBudgetError, match="exceed max_prompt_length=768"):
        check_prompt_budget(rows, _FakeTokenizer(), 768)


# ---------------------------------------------------------------------------
# Environment feedback in team mode
# ---------------------------------------------------------------------------


def test_team_mode_hint_does_not_tell_tier1_to_submit():
    """Regression: the hint said 'Call submit_investigation', which for tier-1 in
    team mode ends the whole episode and scores 0.001."""
    env, obs = _team_obs("team_phishing_escalation")
    aid = obs["alert_queue"][0]["alert_id"]
    env.step(
        SOCAction(action_type="enrich_indicator", indicator=obs["alert_queue"][0]["indicators"]["ip"][0], role="tier1")
    )
    msg = env.step(
        SOCAction(action_type="classify_alert", alert_id=aid, classification="true_positive", role="tier1")
    ).message
    assert "submit_investigation" not in msg
    assert "escalate_to_tier2" in msg and "phase_complete" in msg


def test_solo_mode_hint_still_says_submit():
    env = SOCEnvironment()
    obs = env.reset("phishing", seed=42)
    aid = obs.alert_queue[0].alert_id
    env.step(SOCAction(action_type="query_logs", log_source="firewall", query_alert_id=aid))
    msg = env.step(SOCAction(action_type="classify_alert", alert_id=aid, classification="true_positive")).message
    assert "Call submit_investigation to finalize" in msg


# ---------------------------------------------------------------------------
# Sufficiency: the prompt alone carries enough to play the task
# ---------------------------------------------------------------------------

_ALERT_LINE = re.compile(r'^- (\S+) (\S+) "[^"]*" \| (.*?)(?: \| iocs: (.*))?$')


def _prompt_only_policy(prompt: str) -> str:
    """Rule-based tier-1 that reads nothing but the rendered prompt text."""
    if " 0 need action" in prompt:
        return json.dumps({"action_type": "phase_complete"})
    for line in prompt.splitlines():
        m = _ALERT_LINE.match(line)
        if not m:
            continue
        aid, severity, status, iocs = m.groups()
        if status == "new":
            kind, value = (iocs or "ip=").split()[0].split("=", 1)
            return json.dumps(
                {"action_type": "enrich_indicator", "query_alert_id": aid, "indicator": value, "indicator_type": kind}
            )
        if "class=" not in status:
            cls = "true_positive" if severity in ("high", "critical") else "false_positive"
            return json.dumps({"action_type": "classify_alert", "alert_id": aid, "classification": cls})
        if "class=true_positive" in status and "escalated" not in status:
            return json.dumps({"action_type": "escalate_to_tier2", "alert_id": aid, "justification": "confirmed"})
    return json.dumps({"action_type": "phase_complete"})


@pytest.mark.parametrize(("task", "floor"), [("team_phishing_escalation", 0.95), ("team_lateral_team", 0.75)])
def test_prompt_alone_is_enough_to_play(test_client, task, floor):
    """A text-only policy matches the observation-reading heuristic (0.999 / 0.800
    on the leaderboard). Under the old prompt this was impossible: nothing in it
    said whether an alert had been classified or escalated."""
    from training.evaluation import run_text_policy_episode

    eps = [
        run_text_policy_episode(
            test_client,
            task,
            seed,
            "tier1",
            lambda obs, step: _prompt_only_policy(format_obs_prompt(obs, "tier1")),
            oracle_action,
        )
        for seed in range(100, 105)
    ]
    assert all(ep["escalated"] for ep in eps)
    assert sum(ep["score"] for ep in eps) / len(eps) >= floor
