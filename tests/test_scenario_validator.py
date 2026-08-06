"""Scenario integrity validator.

A malformed scenario silently corrupts every reward graded against it, so the
validator must (a) never false-positive on a real generated scenario and (b)
catch the concrete ways an answer key can go wrong: dangling references,
overlapping TP/FP/benign partitions, unclassified alerts, duplicates, and
structural basics.
"""

import copy

import pytest

from models import AlertClassification
from scenarios.apt_campaign import APTCampaignScenario
from scenarios.insider_threat import InsiderThreatScenario
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.phishing import PhishingScenario
from scenarios.queue_management import QueueManagementScenario
from scenarios.validate import assert_valid, validate_scenario

ALL_SCENARIOS = [
    PhishingScenario,
    LateralMovementScenario,
    QueueManagementScenario,
    InsiderThreatScenario,
    APTCampaignScenario,
]


@pytest.mark.parametrize("scenario_cls", ALL_SCENARIOS)
@pytest.mark.parametrize("seed", [1, 42, 77])
def test_real_scenarios_are_valid(scenario_cls, seed):
    config = scenario_cls(seed=seed).generate()
    assert validate_scenario(config) == []
    assert_valid(config)  # must not raise


def _phishing():
    return PhishingScenario(seed=42).generate()


def test_detects_dangling_ground_truth_reference():
    config = _phishing()
    config.ground_truth.true_positive_ids.append("ALERT-DOESNOTEXIST")
    issues = validate_scenario(config)
    assert any("unknown alert_id" in i for i in issues)


def test_detects_overlapping_partitions():
    config = _phishing()
    real_id = config.alerts[0].alert_id
    config.ground_truth.true_positive_ids = [real_id]
    config.ground_truth.false_positive_ids = [real_id]
    issues = validate_scenario(config)
    assert any("both true_positive and false_positive" in i for i in issues)


def test_detects_unclassified_alert():
    config = _phishing()
    config.ground_truth.alert_classifications = {}
    issues = validate_scenario(config)
    assert any("missing a ground-truth classification" in i for i in issues)


def test_detects_duplicate_alert_ids():
    config = _phishing()
    config.alerts.append(copy.deepcopy(config.alerts[0]))
    issues = validate_scenario(config)
    assert any("duplicate alert_id" in i for i in issues)


def test_detects_nonpositive_max_steps():
    config = _phishing()
    config.max_steps = 0
    assert any("max_steps must be positive" in i for i in validate_scenario(config))


def test_detects_empty_scenario():
    config = _phishing()
    config.alerts = []
    config.ground_truth.alert_classifications = {}
    assert any("no alerts" in i for i in validate_scenario(config))


def test_assert_valid_raises_with_details():
    config = _phishing()
    config.ground_truth.expected_manager_flags.append("NOPE")
    with pytest.raises(ValueError, match="integrity validation"):
        assert_valid(config)


def test_valid_after_fixing_classification_roundtrip():
    # Sanity: clearing then restoring a classification returns to valid.
    config = _phishing()
    aid = config.alerts[0].alert_id
    saved = config.ground_truth.alert_classifications.get(aid, AlertClassification.TRUE_POSITIVE)
    config.ground_truth.alert_classifications = {}
    assert validate_scenario(config)  # broken
    config.ground_truth.alert_classifications = {aid: saved}
    assert validate_scenario(config) == []
