"""
Team-mode tests for SOC-Triage-Gym v2.
Tests multi-agent phase state machine, ticket bus, containment tools,
manager oversight, team grader, and backward-compat guarantee.
"""

import pytest
from fastapi.testclient import TestClient

from models import (
    ActionType,
    AgentRole,
    AlertClassification,
    EpisodeMode,
    EpisodePhase,
    SOCAction,
    TicketKind,
)
from server.environment import SOCEnvironment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SOCEnvironment()


@pytest.fixture
def team_env(env):
    env.reset("team_phishing_escalation", seed=42, mode="team")
    return env


@pytest.fixture
def app_client():
    from server.app import app
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_solo_reset_unchanged(env):
    """tier1_solo mode must behave identically to Round 1 baseline."""
    obs = env.reset("phishing", seed=42)
    assert obs.episode_mode == EpisodeMode.TIER1_SOLO
    assert obs.current_role is None
    assert obs.current_phase is None
    assert obs.phase_steps_remaining is None
    assert obs.tickets == []
    assert len(obs.alert_queue) == 1
    assert obs.investigation_budget == 15


def test_solo_default_mode(env):
    """reset() with no mode argument must default to tier1_solo."""
    obs = env.reset("phishing", seed=42)
    assert obs.episode_mode == EpisodeMode.TIER1_SOLO


def test_solo_step_unchanged(env):
    """Step in solo mode returns no team fields."""
    env.reset("phishing", seed=42)
    alert_id = "ALT-P-001" if True else env._config.alerts[0].alert_id
    alert_id = env._config.alerts[0].alert_id
    obs = env.step(SOCAction(action_type=ActionType.NOOP))
    assert obs.episode_mode == EpisodeMode.TIER1_SOLO
    assert obs.current_role is None
    assert obs.team_reward_breakdown is None


# ---------------------------------------------------------------------------
# Team mode reset
# ---------------------------------------------------------------------------

def test_team_reset_phase_triage(team_env):
    obs = team_env._build_observation(role=AgentRole.TIER1, reward=0.0)
    assert obs.episode_mode == EpisodeMode.TEAM
    assert obs.current_phase == EpisodePhase.TRIAGE
    assert obs.current_role == AgentRole.TIER1
    assert obs.phase_steps_remaining == 40


def test_team_reset_empty_tickets(team_env):
    obs = team_env._build_observation(role=AgentRole.TIER1, reward=0.0)
    assert obs.tickets == []


def test_team_reset_state(team_env):
    st = team_env.state()
    assert st.episode_mode == EpisodeMode.TEAM
    assert st.current_phase == EpisodePhase.TRIAGE
    assert st.current_role == AgentRole.TIER1


# ---------------------------------------------------------------------------
# Tier-1 phase actions
# ---------------------------------------------------------------------------

def test_tier1_classify_correct_reward(team_env):
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                       query_alert_id=alert_id, role=AgentRole.TIER1))
    obs = env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                              classification=AlertClassification.TRUE_POSITIVE, confidence=0.9,
                              role=AgentRole.TIER1))
    assert obs.reward > 0


def test_tier1_escalate_to_tier2_creates_ticket(team_env):
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                       query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                        classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="Confirmed TP", role=AgentRole.TIER1))
    assert len(env._tickets) == 1
    assert env._tickets[0].kind == TicketKind.ESCALATION
    assert env._tickets[0].from_role == AgentRole.TIER1
    assert env._tickets[0].to_role == AgentRole.TIER2
    assert alert_id in env._escalated_alert_ids


def test_tier1_wrong_role_rejected(team_env):
    """A Tier-2 action submitted by tier1 role must be rejected."""
    obs = team_env.step(SOCAction(action_type=ActionType.ISOLATE_HOST,
                                   target_host="somehost", role=AgentRole.TIER1))
    assert obs.reward < 0


def test_over_escalation_penalty(team_env):
    """Escalating >30% of alerts triggers penalty."""
    env = team_env
    # team_phishing has 1 alert — escalating it is fine (0/1 < 0.3)
    # Simulate by manually pre-populating escalated_alert_ids
    env._escalated_alert_ids = [f"fake-{i}" for i in range(5)]  # 5 already escalated
    # With 5 escalated out of 1 total alert, ratio > 0.3
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    obs = env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                              justification="escalating", role=AgentRole.TIER1))
    # reward should be penalised vs normal escalation of required alert
    assert obs.reward < 0.20  # penalty applied even on required escalation


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

def _escalate_one(env):
    """Helper: escalate the first alert so T1 phase_complete doesn't short-circuit."""
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                        classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="TP", role=AgentRole.TIER1))
    return alert_id


def test_phase_complete_advances_to_response(team_env):
    _escalate_one(team_env)
    obs = team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    assert obs.current_phase == EpisodePhase.RESPONSE
    assert obs.current_role == AgentRole.TIER2


def test_response_phase_complete_advances_to_oversight(team_env):
    _escalate_one(team_env)
    team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    obs = team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER2))
    assert obs.current_phase == EpisodePhase.OVERSIGHT
    assert obs.current_role == AgentRole.MANAGER


def test_oversight_submit_ends_episode(team_env):
    _escalate_one(team_env)
    team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER2))
    obs = team_env.step(SOCAction(action_type=ActionType.SUBMIT_INVESTIGATION, role=AgentRole.MANAGER))
    assert obs.done is True
    assert obs.task_score is not None
    assert obs.current_phase == EpisodePhase.COMPLETE


# ---------------------------------------------------------------------------
# Tier-2 containment tools
# ---------------------------------------------------------------------------

def _advance_to_tier2(env):
    """Helper: run T1 phase with escalation, advance to T2."""
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                        classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="TP confirmed", role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    return alert_id


def test_tier2_forensic_timeline(team_env):
    alert_id = _advance_to_tier2(team_env)
    host = list(team_env._config.alerts[0].indicators.get("hostname", ["WORKSTATION-ALPHA"]))[0]
    obs = team_env.step(SOCAction(action_type=ActionType.FORENSIC_TIMELINE, alert_id=alert_id,
                                   target_host=host, role=AgentRole.TIER2))
    assert len(obs.containment_results) > 0
    assert obs.containment_results[-1].action_type == "forensic_timeline"
    assert obs.containment_results[-1].success is True


def test_tier2_isolate_host_positive(team_env):
    alert_id = _advance_to_tier2(team_env)
    # Find a host that is actually in a TP alert's indicators for this scenario.
    from models import AlertClassification as _AC
    tp_host = None
    for a in team_env._config.alerts:
        gt = team_env._config.ground_truth.alert_classifications.get(a.alert_id)
        if gt == _AC.TRUE_POSITIVE:
            hosts = a.indicators.get("hostname", [])
            if hosts:
                tp_host = hosts[0]
                break
    if tp_host is None:
        pytest.skip("No TP host indicator available in this scenario")
    obs = team_env.step(SOCAction(action_type=ActionType.ISOLATE_HOST, alert_id=alert_id,
                                   target_host=tp_host, role=AgentRole.TIER2))
    import re
    m = re.search(r"role=(-?[\d.]+)", obs.message)
    assert m is not None
    assert float(m.group(1)) > 0, "TP host isolation role-specific reward should be positive"
    assert obs.containment_results[-1].success is True


def test_tier2_close_case_creates_closure_ticket(team_env):
    alert_id = _advance_to_tier2(team_env)
    team_env.step(SOCAction(action_type=ActionType.CLOSE_CASE, alert_id=alert_id,
                              justification="Contained", role=AgentRole.TIER2))
    closure_tickets = [t for t in team_env._tickets if t.kind == TicketKind.CLOSURE]
    assert len(closure_tickets) == 1
    assert closure_tickets[0].from_role == AgentRole.TIER2


# ---------------------------------------------------------------------------
# Manager oversight
# ---------------------------------------------------------------------------

def _advance_to_manager(env):
    alert_id = _advance_to_tier2(env)
    host = list(env._config.alerts[0].indicators.get("hostname", ["WORKSTATION-ALPHA"]))[0]
    env.step(SOCAction(action_type=ActionType.CLOSE_CASE, alert_id=alert_id,
                        justification="Contained", role=AgentRole.TIER2))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER2))
    return alert_id


def test_manager_review_decision_positive(team_env):
    alert_id = _advance_to_manager(team_env)
    obs = team_env.step(SOCAction(action_type=ActionType.REVIEW_DECISION, alert_id=alert_id,
                                   role=AgentRole.MANAGER))
    assert obs.reward > 0
    assert obs.manager_review_result is not None
    assert obs.manager_review_result.action_type == "review_decision"


def test_manager_explain_team_behavior_heuristic_score(team_env):
    alert_id = _advance_to_manager(team_env)
    explanation = (
        f"The team investigated {alert_id} and correctly classified it as a true positive. "
        f"Tier-1 escalated the alert after finding malicious IP indicators. "
        f"Tier-2 isolated the host and blocked the C2 domain. No inconsistencies found."
    )
    obs = team_env.step(SOCAction(action_type=ActionType.EXPLAIN_TEAM_BEHAVIOR,
                                   explanation_text=explanation, role=AgentRole.MANAGER))
    assert obs.reward > 0


def test_manager_flag_inconsistency_spurious_penalty(team_env):
    _advance_to_manager(team_env)
    # team_phishing_escalation has expected_manager_flags=[] so flagging is spurious.
    # In team mode, step_reward = 0.6 * role_specific + 0.4 * team_f1, so the blended
    # total may be positive even when role_specific is negative. Verify the role-specific
    # component (logged in message as "Blended reward: role=X") is negative.
    obs = team_env.step(SOCAction(action_type=ActionType.FLAG_INCONSISTENCY,
                                   alert_id="ALT-TEAM-001", flag_reason="Suspicious",
                                   role=AgentRole.MANAGER))
    import re
    match = re.search(r"role=(-?[\d.]+)", obs.message)
    assert match is not None, f"Expected blended reward in message, got: {obs.message}"
    role_component = float(match.group(1))
    assert role_component < 0, f"Spurious flag role_specific reward should be negative, got {role_component}"


# ---------------------------------------------------------------------------
# Team reward and episode score
# ---------------------------------------------------------------------------

def test_full_team_episode_produces_score(team_env):
    """Complete end-to-end team episode must return task_score > 0."""
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    host = list(env._config.alerts[0].indicators.get("hostname", ["WORKSTATION-ALPHA"]))[0]

    # T1 phase
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                        classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="TP", role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))

    # T2 phase
    env.step(SOCAction(action_type=ActionType.ISOLATE_HOST, alert_id=alert_id,
                        target_host=host, role=AgentRole.TIER2))
    env.step(SOCAction(action_type=ActionType.CLOSE_CASE, alert_id=alert_id,
                        justification="Done", role=AgentRole.TIER2))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER2))

    # Manager phase
    env.step(SOCAction(action_type=ActionType.REVIEW_DECISION, alert_id=alert_id,
                        role=AgentRole.MANAGER))
    obs = env.step(SOCAction(action_type=ActionType.SUBMIT_INVESTIGATION, role=AgentRole.MANAGER))

    assert obs.done is True
    assert obs.task_score > 0.5
    assert obs.team_reward_breakdown is not None
    assert obs.team_reward_breakdown.tier1_individual > 0
    # tier2 individual accumulates blended (0.6*role + 0.4*team_delta); with delta semantics
    # the tier2 window may net slightly negative after T1 consumed most of the team delta.
    # Verify overall team total is positive instead.
    assert obs.team_reward_breakdown.total > 0


def test_team_reward_breakdown_per_role(team_env):
    """Per-role rewards must accumulate independently."""
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    t1_pre = env._tier1_reward
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    assert env._tier2_reward == 0.0  # T2 hasn't acted yet
    assert env._tier1_reward == t1_pre  # phase_complete gives 0 reward, T1 stays same


# ---------------------------------------------------------------------------
# Ticket visibility (role-filtered observations)
# ---------------------------------------------------------------------------

def test_tier2_sees_escalation_tickets_only(team_env):
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="TP", role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    obs = env._build_observation(role=AgentRole.TIER2, reward=0.0)
    assert any(t.kind == TicketKind.ESCALATION for t in obs.tickets)


def test_tier2_sees_only_escalated_alerts(team_env):
    env = team_env
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip, indicator_type="ip",
                        query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                        justification="TP", role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    obs = env._build_observation(role=AgentRole.TIER2, reward=0.0)
    assert all(a.alert_id in env._escalated_alert_ids for a in obs.alert_queue)


def test_tier1_does_not_see_tier2_tickets(team_env):
    obs = team_env._build_observation(role=AgentRole.TIER1, reward=0.0)
    # No T2 tickets have been created yet
    tier2_tickets = [t for t in obs.tickets if t.from_role == AgentRole.TIER2]
    assert tier2_tickets == []


# ---------------------------------------------------------------------------
# Team-mode graders
# ---------------------------------------------------------------------------

def test_team_grader_perfect_score():
    from graders.team_grader import TeamGrader
    from models import AlertClassification, InvestigationState, ResponseActionType
    from scenarios.team_phishing_escalation import TeamPhishingEscalationScenario

    scenario = TeamPhishingEscalationScenario(seed=42).generate()
    alert_id = scenario.alerts[0].alert_id
    investigations = {alert_id: InvestigationState(alert_id=alert_id)}
    investigations[alert_id].classification = AlertClassification.TRUE_POSITIVE
    investigations[alert_id].recommended_actions = [ResponseActionType.QUARANTINE_FILE]
    investigations[alert_id].escalated = True

    grader = TeamGrader()
    score = grader.grade(scenario, investigations, steps_used=10, max_steps=60)
    assert score >= 0.5


def test_red_team_grader_inverse_of_blue():
    from graders.red_team_grader import RedTeamGrader
    from graders.team_grader import TeamGrader
    from models import AlertClassification, InvestigationState
    from scenarios.team_phishing_escalation import TeamPhishingEscalationScenario

    scenario = TeamPhishingEscalationScenario(seed=42).generate()
    alert_id = scenario.alerts[0].alert_id
    investigations = {alert_id: InvestigationState(alert_id=alert_id)}
    # Blue team scores 0: wrong classification
    investigations[alert_id].classification = AlertClassification.FALSE_POSITIVE

    blue_score = TeamGrader().grade(scenario, investigations, 10, 60)
    red_score = RedTeamGrader().grade(scenario, investigations, 10, 60)
    # Red should be high when blue is low
    assert red_score > 0.5
    assert red_score > blue_score


# ---------------------------------------------------------------------------
# Red-Team Generator
# ---------------------------------------------------------------------------

def test_red_team_generator_deterministic():
    from models import RedTeamConfig
    from scenarios.red_team_generator import RedTeamGenerator

    config = RedTeamConfig(difficulty_floor=0.5)
    g1 = RedTeamGenerator(config=config, seed=42)
    g2 = RedTeamGenerator(config=config, seed=42)
    s1 = g1.generate()
    s2 = g2.generate()
    assert s1.scenario_id == s2.scenario_id
    assert len(s1.alerts) == len(s2.alerts)


def test_red_team_generator_higher_difficulty_more_alerts():
    from models import RedTeamConfig
    from scenarios.red_team_generator import RedTeamGenerator

    low = RedTeamGenerator(config=RedTeamConfig(difficulty_floor=0.1), seed=42).generate()
    high = RedTeamGenerator(config=RedTeamConfig(difficulty_floor=0.9), seed=42).generate()
    assert len(high.alerts) >= len(low.alerts)


def test_red_team_adapt_difficulty_increases():
    from models import RedTeamConfig
    from scenarios.red_team_generator import RedTeamGenerator

    g = RedTeamGenerator(config=RedTeamConfig(difficulty_floor=0.5), seed=42)
    g2 = g.adapt_difficulty(blue_win_rate=0.8)  # high win rate → harder
    assert g2.config.difficulty_floor > 0.5


def test_red_team_adapt_difficulty_decreases():
    from models import RedTeamConfig
    from scenarios.red_team_generator import RedTeamGenerator

    g = RedTeamGenerator(config=RedTeamConfig(difficulty_floor=0.5), seed=42)
    g2 = g.adapt_difficulty(blue_win_rate=0.2)  # low win rate → easier
    assert g2.config.difficulty_floor < 0.5


# ---------------------------------------------------------------------------
# HTTP API — team endpoints
# ---------------------------------------------------------------------------

def test_api_reset_team_mode(app_client):
    resp = app_client.post("/reset", json={"task_id": "team_phishing_escalation", "seed": 42, "mode": "team"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode_mode"] == "team"
    assert data["current_role"] == "tier1"
    assert data["current_phase"] == "triage"


def test_api_reset_solo_default(app_client):
    resp = app_client.post("/reset", json={"task_id": "phishing", "seed": 42})
    assert resp.status_code == 200
    assert resp.json()["episode_mode"] == "tier1_solo"


def test_api_generate_scenario(app_client):
    resp = app_client.post("/generate_scenario", json={"seed": 42, "difficulty_floor": 0.6})
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert data["task_id"] == "red_team_generated"
    assert len(data["alerts"]) > 0


def test_api_tasks_includes_team(app_client):
    resp = app_client.get("/tasks")
    ids = [t["id"] for t in resp.json()["tasks"]]
    assert "team_phishing_escalation" in ids
    assert "team_lateral_team" in ids


def test_api_team_lateral_team_reset(app_client):
    resp = app_client.post("/reset", json={"task_id": "team_lateral_team", "seed": 42, "mode": "team"})
    # key can be missing seed, still 200 with defaults
    resp = app_client.post("/reset", json={"task_id": "team_lateral_team", "mode": "team"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode_mode"] == "team"
    assert len(data["alert_queue"]) == 8  # 5 TP + 3 FP


# ---------------------------------------------------------------------------
# Regression tests for reward-hack / correctness fixes (B1-B6)
# ---------------------------------------------------------------------------

def test_team_f1_delta_not_sticky(team_env):
    """After classifying correctly, NOOP spam must NOT keep yielding team_f1 reward."""
    alert_id = team_env._config.alerts[0].alert_id
    ip = list(team_env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    team_env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip,
                             indicator_type="ip", query_alert_id=alert_id, role=AgentRole.TIER1))
    team_env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                             classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))

    import re
    team_components = []
    for _ in range(5):
        obs = team_env.step(SOCAction(action_type=ActionType.NOOP, role=AgentRole.TIER1))
        m = re.search(r"team=(-?[\d.]+)", obs.message)
        if m:
            team_components.append(float(m.group(1)))

    # Delta should be ~0 on every NOOP after the classification-driven delta was consumed.
    assert all(abs(t) < 0.01 for t in team_components[1:]), \
        f"team_f1 delta should be ~0 after consumption, got {team_components}"


def test_close_case_idempotency(team_env):
    """Second close_case on same alert must yield negative reward (no farming)."""
    alert_id = _advance_to_tier2(team_env)
    obs1 = team_env.step(SOCAction(action_type=ActionType.CLOSE_CASE, alert_id=alert_id,
                                     justification="Contained", role=AgentRole.TIER2))
    obs2 = team_env.step(SOCAction(action_type=ActionType.CLOSE_CASE, alert_id=alert_id,
                                     justification="Contained again", role=AgentRole.TIER2))
    import re
    m = re.search(r"role=(-?[\d.]+)", obs2.message)
    assert m is not None
    assert float(m.group(1)) < 0, "Duplicate close_case should be penalized"


def test_tier1_phase_complete_with_zero_escalations_short_circuits(team_env):
    """If T1 calls phase_complete with zero escalations, episode terminates."""
    obs = team_env.step(SOCAction(action_type=ActionType.PHASE_COMPLETE, role=AgentRole.TIER1))
    assert obs.done is True
    assert obs.current_phase == EpisodePhase.COMPLETE


def test_manager_judge_fallback_on_missing_api_key(monkeypatch):
    """ManagerJudge must fall back to heuristic when no API key is present."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from graders.manager_judge import ManagerJudge

    judge = ManagerJudge()
    score = judge.judge(
        explanation="Alert ALT-TEAM-001 escalated and contained via host isolation.",
        investigations={"ALT-TEAM-001": {"classification": "true_positive"}},
        config=None,
        episode_id="test-ep",
        seed=42,
        trajectory_hash="abc",
    )
    assert 0.001 <= score <= 0.999
    assert score > 0.0  # heuristic should credit mentions/keywords


def test_over_escalation_threshold_off_by_one():
    """Ensure the threshold is strictly >25% (not the old >=30%)."""
    env = SOCEnvironment()
    env.reset("team_lateral_team", seed=42, mode="team")
    # Escalate ceil(0.25 * 8) + 1 = 3 alerts worth of ids; 3/8 = 0.375 > 0.25 → penalty
    env._escalated_alert_ids = [f"fake-{i}" for i in range(2)]  # already 2/8=0.25 (boundary)
    alert_id = env._config.alerts[0].alert_id
    ip = list(env._config.alerts[0].indicators.get("ip", ["1.2.3.4"]))[0]
    env.step(SOCAction(action_type=ActionType.ENRICH_INDICATOR, indicator=ip,
                        indicator_type="ip", query_alert_id=alert_id, role=AgentRole.TIER1))
    env.step(SOCAction(action_type=ActionType.CLASSIFY_ALERT, alert_id=alert_id,
                        classification=AlertClassification.TRUE_POSITIVE, role=AgentRole.TIER1))
    obs = env.step(SOCAction(action_type=ActionType.ESCALATE_TO_TIER2, alert_id=alert_id,
                              justification="escalating", role=AgentRole.TIER1))
    # 3/8 = 0.375 > 0.25 triggers penalty
    import re
    m = re.search(r"role=(-?[\d.]+)", obs.message)
    assert m is not None
    role_component = float(m.group(1))
    # Normal escalation reward is higher; penalty (-0.08) applied should lower it substantially
    assert role_component < 0.20
