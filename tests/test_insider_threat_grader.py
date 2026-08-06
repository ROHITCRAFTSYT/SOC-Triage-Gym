"""Insider Threat grader (Task 4) — previously untested.

Graders are the reward core, so this pins the component scorers (F1 over TP/FP,
attack-chain detection, missed-TP penalty) in isolation with a synthetic ground
truth, plus an end-to-end grade() on a real generated scenario: perfect play
scores near the top of the clamped range, doing nothing scores near the bottom.
"""

from dataclasses import dataclass, field

from graders.insider_threat_grader import InsiderThreatGrader
from models import AlertClassification, InvestigationState
from scenarios.insider_threat import InsiderThreatScenario

TP = AlertClassification.TRUE_POSITIVE
FP = AlertClassification.FALSE_POSITIVE
BTP = AlertClassification.BENIGN_TRUE_POSITIVE


@dataclass
class FakeGT:
    true_positive_ids: list = field(default_factory=list)
    false_positive_ids: list = field(default_factory=list)
    benign_tp_ids: list = field(default_factory=list)
    attack_chain_ids: list = field(default_factory=list)


def _inv(mapping: dict) -> dict:
    return {aid: InvestigationState(alert_id=aid, classification=cls) for aid, cls in mapping.items()}


# --------------------------------------------------------------- F1
def test_f1_is_one_for_perfect_classification():
    g = InsiderThreatGrader()
    gt = FakeGT(true_positive_ids=["a", "b"], false_positive_ids=["c"])
    inv = _inv({"a": TP, "b": TP, "c": FP})
    assert g._compute_f1(inv, gt) == 1.0


def test_f1_punishes_missed_and_false_positives():
    g = InsiderThreatGrader()
    gt = FakeGT(true_positive_ids=["a", "b"], false_positive_ids=["c"])
    # 'a' missed (unclassified), 'c' wrongly called TP
    inv = _inv({"b": TP, "c": TP})
    assert g._compute_f1(inv, gt) < 1.0


# --------------------------------------------------------------- attack chains
def test_attack_chain_found_only_when_all_alerts_are_tp():
    g = InsiderThreatGrader()
    gt = FakeGT(attack_chain_ids=[["a", "b"], ["c", "d"]])
    full = _inv({"a": TP, "b": TP, "c": TP, "d": TP})
    assert g._attack_chain_score(full, gt) == 1.0
    half = _inv({"a": TP, "b": TP, "c": TP})  # 'd' unclassified -> chain 2 not found
    assert g._attack_chain_score(half, gt) == 0.5


def test_no_chains_scores_full():
    assert InsiderThreatGrader()._attack_chain_score({}, FakeGT()) == 1.0


# --------------------------------------------------------------- missed TP
def test_missed_tp_penalty_reduces_score():
    g = InsiderThreatGrader()
    gt = FakeGT(true_positive_ids=["a", "b", "c"])
    assert g._missed_tp_score(_inv({"a": TP, "b": TP, "c": TP}), gt) == 1.0
    partial = g._missed_tp_score(_inv({"a": TP}), gt)  # two TPs missed
    assert partial < 1.0
    assert g._missed_tp_score({}, gt) < partial  # all missed, worse (floored at 0)


# --------------------------------------------------------------- end to end
def _perfect_investigations(config):
    gt = config.ground_truth
    invs = {}
    for alert in config.alerts:
        aid = alert.alert_id
        invs[aid] = InvestigationState(
            alert_id=aid,
            classification=gt.alert_classifications.get(aid),
            recommended_actions=list(gt.expected_response_actions.get(aid, [])),
        )
    return invs


def test_perfect_play_beats_doing_nothing():
    config = InsiderThreatScenario(seed=42).generate()
    g = InsiderThreatGrader()
    perfect = g.grade(config, _perfect_investigations(config), steps_used=10, max_steps=80)
    nothing = g.grade(config, {}, steps_used=80, max_steps=80)
    assert perfect > nothing
    assert 0.001 <= nothing and perfect <= 0.999  # clamped range


def test_grade_with_breakdown_shape():
    config = InsiderThreatScenario(seed=7).generate()
    score, breakdown, feedback = InsiderThreatGrader().grade_with_breakdown(
        config, _perfect_investigations(config), 10, 80)
    assert 0.0 <= score <= 1.0
    assert {"f1_score", "attack_chains_found", "true_positive_coverage",
            "efficiency", "response_quality"} <= set(breakdown)
    assert isinstance(feedback, str)
