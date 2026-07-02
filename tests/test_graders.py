"""
Tests for all graders: PhishingGrader, LateralMovementGrader, QueueManagementGrader.
"""


from models import (
    AlertClassification,
    CorrelatedEvent,
    CorrelationType,
    InvestigationState,
)
from graders.phishing_grader import PhishingGrader
from graders.lateral_movement_grader import LateralMovementGrader
from graders.queue_management_grader import QueueManagementGrader


# ===================================================================
# Phishing Grader Tests
# ===================================================================


class TestPhishingGrader:
    """Tests for PhishingGrader (Task 1)."""

    def test_phishing_grader_perfect_tp(self, phishing_config):
        """Classify correctly, map technique, query relevant logs, recommend action -> score near 1.0."""
        grader = PhishingGrader()
        config = phishing_config
        alert_id = config.alerts[0].alert_id
        gt = config.ground_truth

        inv = InvestigationState(alert_id=alert_id)

        # Classify correctly as TP
        inv.classification = AlertClassification.TRUE_POSITIVE

        # Map all expected techniques
        inv.mapped_techniques = list(gt.expected_techniques.get(alert_id, []))

        # Query all relevant log sources
        relevant_sources = gt.relevant_log_sources.get(alert_id, [])
        for source in relevant_sources:
            source_key = source.value
            entries = config.log_db.get(source_key, {}).get(alert_id, [])
            inv.queried_sources[source_key] = entries

        # Recommend all expected actions
        inv.recommended_actions = list(gt.expected_response_actions.get(alert_id, []))

        investigations = {alert_id: inv}
        score = grader.grade(config, investigations, steps_used=5, max_steps=15)

        assert score >= 0.95, f"Perfect phishing TP should score >= 0.95, got {score}"

    def test_phishing_grader_wrong_classification(self, phishing_config):
        """Classify TP as FP -> low score."""
        grader = PhishingGrader()
        config = phishing_config
        alert_id = config.alerts[0].alert_id

        inv = InvestigationState(alert_id=alert_id)
        inv.classification = AlertClassification.FALSE_POSITIVE  # Wrong!

        investigations = {alert_id: inv}
        score = grader.grade(config, investigations, steps_used=5, max_steps=15)

        # Classification weight is 0.4, and it's 0 for wrong classification
        # Other components may also be 0 since no evidence gathered
        assert score < 0.4, f"Wrong classification should give low score, got {score}"

    def test_phishing_grader_unclassified(self, phishing_config):
        """Submit without classifying -> 0.0 classification score."""
        grader = PhishingGrader()
        config = phishing_config
        alert_id = config.alerts[0].alert_id

        inv = InvestigationState(alert_id=alert_id)
        # classification remains None

        investigations = {alert_id: inv}
        score = grader.grade(config, investigations, steps_used=5, max_steps=15)

        # Classification=0, technique=0, evidence=0, response=0 -> score = 0.0
        assert score == 0.0, f"Unclassified should give 0.0, got {score}"

    def test_phishing_grader_grade_matches_breakdown(self, phishing_config):
        """grade() and grade_with_breakdown() produce the same score."""
        grader = PhishingGrader()
        config = phishing_config
        alert_id = config.alerts[0].alert_id

        inv = InvestigationState(alert_id=alert_id)
        inv.classification = AlertClassification.TRUE_POSITIVE
        inv.mapped_techniques = ["T1566.001"]

        investigations = {alert_id: inv}

        score = grader.grade(config, investigations, steps_used=5, max_steps=15)
        breakdown_score, breakdown, feedback = grader.grade_with_breakdown(
            config, investigations, steps_used=5, max_steps=15
        )

        assert abs(score - breakdown_score) < 1e-6, (
            f"grade() returned {score} but grade_with_breakdown() returned {breakdown_score}"
        )
        assert isinstance(breakdown, dict)
        assert "classification" in breakdown


# ===================================================================
# Lateral Movement Grader Tests
# ===================================================================


class TestLateralMovementGrader:
    """Tests for LateralMovementGrader (Task 2)."""

    def test_lateral_movement_all_tp_classified(self, lateral_movement_config):
        """Correctly classify all 5 alerts as TP -> high classification component."""
        grader = LateralMovementGrader()
        config = lateral_movement_config
        gt = config.ground_truth

        investigations = {}
        for alert in config.alerts:
            inv = InvestigationState(alert_id=alert.alert_id)
            inv.classification = AlertClassification.TRUE_POSITIVE
            investigations[alert.alert_id] = inv

        score = grader.grade(config, investigations, steps_used=10, max_steps=30)

        # Classification is 0.30 weight * 1.0 = 0.30
        # Efficiency at 10/30 ~= 0.33 ratio -> 1.0 efficiency * 0.10 = 0.10
        # Other components may be 0 since we didn't map techniques/response
        # But classification + efficiency should be at least 0.30 + 0.10 = 0.40
        assert score >= 0.35, f"All correct TP classification should give decent score, got {score}"

    def test_lateral_movement_chain_reconstruction(self, lateral_movement_config):
        """Correlating all adjacent kill chain pairs should give chain_score = 1.0."""
        grader = LateralMovementGrader()
        config = lateral_movement_config
        gt = config.ground_truth
        kill_chain = gt.kill_chain_order

        investigations = {}
        for alert in config.alerts:
            investigations[alert.alert_id] = InvestigationState(alert_id=alert.alert_id)

        # Create correlations for each adjacent pair
        for i in range(len(kill_chain) - 1):
            pair = [kill_chain[i], kill_chain[i + 1]]
            corr = CorrelatedEvent(
                alert_ids=pair,
                correlation_type=CorrelationType.USER,
                shared_indicator="shared_value",
                description=f"Correlation between {pair[0]} and {pair[1]}",
                confidence=0.9,
            )
            investigations[kill_chain[i]].correlations_found.append(corr)

        # Get the chain score via the internal method
        chain_score = grader._chain_reconstruction_score(investigations, kill_chain)
        assert chain_score == 1.0, f"Full chain reconstruction should give 1.0, got {chain_score}"


# ===================================================================
# Queue Management Grader Tests
# ===================================================================


class TestQueueManagementGrader:
    """Tests for QueueManagementGrader (Task 3)."""

    def test_queue_management_f1_all_correct(self, queue_management_config):
        """Correctly classify all 20 alerts -> F1 = 1.0 and high overall score."""
        grader = QueueManagementGrader()
        config = queue_management_config
        gt = config.ground_truth

        investigations = {}
        for alert in config.alerts:
            inv = InvestigationState(alert_id=alert.alert_id)
            inv.classification = gt.alert_classifications.get(alert.alert_id)
            investigations[alert.alert_id] = inv

        score = grader.grade(config, investigations, steps_used=20, max_steps=60)

        # With all correct classifications:
        # F1 = 1.0 (0.30), attack_chain = 1.0 (0.20), missed_tp = 1.0 (0.20)
        # efficiency at 20/60 = 0.33 -> 1.0 (0.15)
        # response may be partial since we didn't recommend actions
        # At minimum: 0.30 + 0.20 + 0.20 + 0.15 = 0.85
        assert score >= 0.70, f"All correct classifications should give high score, got {score}"

    def test_queue_management_missed_tp_penalty(self, queue_management_config):
        """Leaving all TPs unclassified should heavily penalize the score."""
        grader = QueueManagementGrader()
        config = queue_management_config
        gt = config.ground_truth

        investigations = {}
        for alert in config.alerts:
            inv = InvestigationState(alert_id=alert.alert_id)
            # Only classify FPs correctly, leave TPs and BTPs unclassified
            if alert.alert_id in gt.false_positive_ids:
                inv.classification = AlertClassification.FALSE_POSITIVE
            # TPs and BTPs remain None (unclassified)
            investigations[alert.alert_id] = inv

        score = grader.grade(config, investigations, steps_used=20, max_steps=60)

        # Missed all TPs -> missed_tp_score should be 0.0 (5 TPs * -0.2 = -1.0, clamped to 0)
        # F1 will be low since recall = 0
        # Attack chains not found
        assert score < 0.40, f"Missed all TPs should give low score, got {score}"
