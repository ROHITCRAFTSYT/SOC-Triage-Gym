"""
Theme-coverage regression tests (SOC-Triage-Gym v3).

Each test asserts presence + correctness of one OpenEnv hackathon sub-theme
mechanic so that judges — and CI — can confirm the env still claims what
it says it claims in the README.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from actors import build_default_registry
from graders.expert_panel import ExpertPanel, build_standard_panel
from graders.token_scaled_reward import token_scaled_bonus
from models import (
    ActorKind,
    AgentRole,
    PolicyVersion,
    RewardBlendConfig,
)
from scenarios.policy_drift import PolicyDriftEngine
from server.app import app
from tools.ticketing import TicketingSystem

# ---------------------------------------------------------------------------
# Halluminate — multi-actor inbox
# ---------------------------------------------------------------------------

def test_actors_registry_deterministic_and_routes_by_role():
    reg_a = build_default_registry(seed=42)
    reg_b = build_default_registry(seed=42)
    for step in range(1, 60):
        msgs_a = reg_a.tick(step=step, ctx={"policy_version": 1})
        msgs_b = reg_b.tick(step=step, ctx={"policy_version": 1})
        assert len(msgs_a) == len(msgs_b), f"non-deterministic at step {step}"

    tier1_msgs = reg_a.inbox_for(AgentRole.TIER1)
    manager_msgs = reg_a.inbox_for(AgentRole.MANAGER)
    assert any(m.actor == ActorKind.THREAT_INTEL for m in tier1_msgs)
    assert any(m.actor == ActorKind.END_USER for m in tier1_msgs)
    assert any(m.actor == ActorKind.COMPLIANCE for m in manager_msgs)


def test_actors_produce_both_relevant_and_noise_messages():
    # At least some end-user messages must be real (relevant) and some noise.
    reg = build_default_registry(seed=7)
    for step in range(1, 120):
        reg.tick(step=step)
    end_user = [m for m in reg.all_messages() if m.actor == ActorKind.END_USER]
    assert any(m.ground_truth_relevant for m in end_user)
    assert any(not m.ground_truth_relevant for m in end_user)


# ---------------------------------------------------------------------------
# Patronus — policy / schema drift
# ---------------------------------------------------------------------------

def test_policy_drift_respects_schedule_and_activates_in_order():
    engine = PolicyDriftEngine(seed=99)
    engine.plan(max_steps=100, drift_count=2)
    seen = []
    for step in range(1, 101):
        v = engine.maybe_drift(step)
        if v is not None:
            seen.append(v.version)
    assert len(seen) == 2
    assert seen == [2, 3]
    assert engine.current().version == 3


def test_policy_active_at_returns_correct_version():
    engine = PolicyDriftEngine(seed=1)
    engine.plan(max_steps=60, drift_count=1)
    drift_step = engine._schedule[0]
    # Before the drift, v1 should be active.
    assert engine.active_at(drift_step - 1).version == 1
    engine.maybe_drift(drift_step)
    assert engine.active_at(drift_step).version == 2
    assert engine.active_at(drift_step + 5).version == 2


def test_policy_compliance_flags_admin_escalation_violations():
    engine = PolicyDriftEngine(seed=1)
    engine._versions.append(
        PolicyVersion(version=2, step_activated=5, admin_must_escalate=True)
    )
    log = [
        {"step": 3, "is_admin": True, "escalated": False},   # before drift — no violation
        {"step": 6, "is_admin": True, "escalated": True},    # after drift, compliant
        {"step": 7, "is_admin": True, "escalated": False},   # after drift, violation
    ]
    result = engine.policy_compliance(log)
    assert result["total"] == 2
    assert result["violations"] == 1
    assert result["compliance_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Mercor — token-scaled reward
# ---------------------------------------------------------------------------

def test_token_bonus_zero_below_floor():
    cfg = RewardBlendConfig()
    assert token_scaled_bonus("short text", 1.0, cfg) == 0.0


def test_token_bonus_saturates_at_cap():
    cfg = RewardBlendConfig()
    long_text = "word " * (cfg.token_scale_cap + 200)
    bonus = token_scaled_bonus(long_text, content_quality=1.0, config=cfg)
    assert bonus == pytest.approx(cfg.token_scale_max_bonus)


def test_token_bonus_zero_when_quality_zero():
    cfg = RewardBlendConfig()
    long_text = "word " * 500
    assert token_scaled_bonus(long_text, 0.0, cfg) == 0.0


def test_token_bonus_monotonic_in_length():
    cfg = RewardBlendConfig()
    texts = ["word " * n for n in (50, 150, 300, 500)]
    bonuses = [token_scaled_bonus(t, 1.0, cfg) for t in texts]
    assert bonuses == sorted(bonuses)


# ---------------------------------------------------------------------------
# Snorkel — experts-in-the-loop
# ---------------------------------------------------------------------------

def test_expert_panel_rotation_is_cyclic():
    panel = ExpertPanel(build_standard_panel())
    ids = [panel.for_round(i).expert_id for i in range(6)]
    assert ids[0] == ids[3]  # cycle length 3
    assert len({ids[0], ids[1], ids[2]}) == 3


def test_expert_weights_shift_scoring():
    panel = ExpertPanel()
    signals = {
        "accuracy": 1.0, "reasoning": 0.0, "actionability": 0.0,
        "speed": 1.0, "thoroughness": 0.0,
    }
    scored = [panel.score(signals, expert=e) for e in panel.all_profiles()]
    # Accuracy-weighted expert should score this highest, speedy still high, thorough lowest.
    ids = [s["expert_id"] for s in scored]
    totals = {s["expert_id"]: s["total"] for s in scored}
    assert "dr_accuracy" in ids
    assert totals["dr_accuracy"] > totals["thorough_thea"]


# ---------------------------------------------------------------------------
# Scaler AI Labs — multi-app ticketing + cross-app rule
# ---------------------------------------------------------------------------

def test_ticketing_open_resolve_and_audit():
    ts = TicketingSystem()
    t = ts.open(alert_id="ALT-001", priority="P2", note="suspicious login")
    assert t.ticket_id.startswith("TKT-")
    assert ts.open_count() == 1
    ts.touch(t.ticket_id, app="EDR", note="isolated host")
    assert "EDR" in ts.get(t.ticket_id).app_chain
    ts.resolve(t.ticket_id, note="contained")
    assert ts.open_count() == 0
    audit = ts.audit_summary()
    assert audit["resolved"] == 1
    assert "EDR" in audit["apps_used"]


def test_iam_disable_user_blocked_without_priority_ticket():
    ts = TicketingSystem()
    assert ts.can_disable_user("ALT-xyz") is False
    ts.open("ALT-xyz", priority="P3")
    assert ts.can_disable_user("ALT-xyz") is False  # P3 not high enough
    ts.open("ALT-xyz", priority="P1")
    assert ts.can_disable_user("ALT-xyz") is True


def test_ticketing_sla_tick_decrements():
    ts = TicketingSystem()
    t = ts.open("ALT-1", priority="P1")
    initial = t.sla_steps_remaining
    for _ in range(3):
        ts.tick()
    assert ts.get(t.ticket_id).sla_steps_remaining == initial - 3


# ---------------------------------------------------------------------------
# Long-horizon — apt_campaign task registered and grader wired
# ---------------------------------------------------------------------------

def test_apt_campaign_registered():
    from graders import GRADER_REGISTRY
    from scenarios import SCENARIO_REGISTRY
    assert "apt_campaign" in SCENARIO_REGISTRY
    assert "apt_campaign" in GRADER_REGISTRY


def test_apt_campaign_generates_long_scenario():
    from scenarios.apt_campaign import APTCampaignScenario
    cfg = APTCampaignScenario(seed=42).generate()
    assert cfg.max_steps == 250
    assert len(cfg.alerts) >= 30  # composite — typically 60+
    assert cfg.task_id == "apt_campaign"
    assert cfg.ground_truth.true_positive_ids


def test_apt_campaign_grader_rewards_narrative_length():
    from graders.apt_campaign_grader import APTCampaignGrader
    from scenarios.apt_campaign import APTCampaignScenario

    grader = APTCampaignGrader()
    cfg = APTCampaignScenario(seed=42).generate()
    # Fabricate a minimal investigations dict that classifies all TPs correctly.
    from models import AlertClassification, InvestigationState
    invs = {}
    for alert in cfg.alerts:
        gt_cls = cfg.ground_truth.alert_classifications.get(
            alert.alert_id, AlertClassification.FALSE_POSITIVE
        )
        invs[alert.alert_id] = InvestigationState(
            alert_id=alert.alert_id, classification=gt_cls
        )

    grader.set_context(narrative_text="", policy_compliance_rate=1.0)
    short_score = grader.grade(cfg, invs, steps_used=100, max_steps=250)

    long_narrative = "word " * 400
    grader.set_context(narrative_text=long_narrative, policy_compliance_rate=1.0)
    long_score = grader.grade(cfg, invs, steps_used=100, max_steps=250)

    assert long_score > short_score


# ---------------------------------------------------------------------------
# Endpoint-level smoke: themes/coverage + actor/expert/policy endpoints live
# ---------------------------------------------------------------------------

def test_themes_coverage_endpoint():
    with TestClient(app) as client:
        r = client.get("/themes/coverage")
        assert r.status_code == 200
        body = r.json()
        for key in (
            "theme_1_multi_agent",
            "fleet_ai_oversight",
            "halluminate_multi_actor",
            "theme_2_long_horizon",
            "mercor_token_scaled_rewards",
            "patronus_schema_drift",
            "theme_4_self_improvement",
            "snorkel_experts_in_loop",
            "scaler_ai_multi_app_enterprise",
        ):
            assert body["coverage"][key] is True


def test_expert_rotation_endpoint_cycles():
    with TestClient(app) as client:
        r0 = client.post("/experts/rotate", json={"round_index": 0}).json()
        r1 = client.post("/experts/rotate", json={"round_index": 1}).json()
        r2 = client.post("/experts/rotate", json={"round_index": 2}).json()
        r3 = client.post("/experts/rotate", json={"round_index": 3}).json()
        assert r0["expert"]["expert_id"] == r3["expert"]["expert_id"]
        assert r0["expert"]["expert_id"] != r1["expert"]["expert_id"]
        assert r1["expert"]["expert_id"] != r2["expert"]["expert_id"]


def test_reward_token_bonus_endpoint_scales_with_length():
    with TestClient(app) as client:
        short = client.post("/reward/token_bonus", json={"text": "hi", "content_quality": 1.0}).json()
        long = client.post(
            "/reward/token_bonus",
            json={"text": "word " * 500, "content_quality": 1.0},
        ).json()
        assert short["bonus"] == 0.0
        assert long["bonus"] > 0.0


def test_actors_and_policy_endpoints_after_reset():
    with TestClient(app) as client:
        reset = client.post("/reset", json={"task_id": "phishing", "seed": 5}).json()
        assert reset["task_id"] == "phishing"
        # Advance a few steps so actors tick.
        for _ in range(6):
            client.post("/step", json={"action_type": "noop"})
        msgs = client.get("/actors/messages").json()
        assert "messages" in msgs
        pol = client.get("/policy/current").json()
        assert pol["version"] >= 1


def test_ticketing_endpoint_cross_app_rule():
    with TestClient(app) as client:
        client.post("/reset", json={"task_id": "phishing", "seed": 1})
        resp = client.get("/tickets/can_disable_user", params={"alert_id": "ALT-Z"}).json()
        assert resp["allowed"] is False
        client.post("/tickets/open", json={"alert_id": "ALT-Z", "priority": "P1"})
        resp = client.get("/tickets/can_disable_user", params={"alert_id": "ALT-Z"}).json()
        assert resp["allowed"] is True
