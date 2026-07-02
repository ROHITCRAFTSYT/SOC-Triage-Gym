"""
Tests for scenario generation: alert counts, determinism, ground truth.
"""


from models import AlertClassification
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.phishing import PhishingScenario
from scenarios.queue_management import QueueManagementScenario


class TestPhishingScenario:
    def test_phishing_generates_1_alert(self):
        """Phishing scenario should generate exactly 1 alert."""
        config = PhishingScenario(seed=42).generate()
        assert len(config.alerts) == 1
        assert config.task_id == "phishing"
        assert config.max_steps == 15

    def test_phishing_ground_truth(self):
        """Ground truth should have a classification for the single alert."""
        config = PhishingScenario(seed=42).generate()
        alert_id = config.alerts[0].alert_id
        gt = config.ground_truth
        assert alert_id in gt.alert_classifications
        classification = gt.alert_classifications[alert_id]
        assert classification in {
            AlertClassification.TRUE_POSITIVE,
            AlertClassification.FALSE_POSITIVE,
        }


class TestLateralMovementScenario:
    def test_lateral_movement_generates_5_alerts(self):
        """Lateral movement scenario should generate exactly 5 alerts."""
        config = LateralMovementScenario(seed=42).generate()
        assert len(config.alerts) == 5
        assert config.task_id == "lateral_movement"
        assert config.max_steps == 30

    def test_lateral_movement_kill_chain(self):
        """Kill chain order should contain all 5 alert IDs."""
        config = LateralMovementScenario(seed=42).generate()
        gt = config.ground_truth
        assert gt.kill_chain_order is not None
        assert len(gt.kill_chain_order) == 5
        alert_ids = {a.alert_id for a in config.alerts}
        for chain_id in gt.kill_chain_order:
            assert chain_id in alert_ids


class TestQueueManagementScenario:
    def test_queue_management_generates_20_alerts(self):
        """Queue management scenario should generate exactly 20 alerts."""
        config = QueueManagementScenario(seed=42).generate()
        assert len(config.alerts) == 20
        assert config.task_id == "queue_management"
        assert config.max_steps == 60

    def test_queue_management_alert_composition(self):
        """Should have 5 TPs, 3 BTPs, 12 FPs."""
        config = QueueManagementScenario(seed=42).generate()
        gt = config.ground_truth
        assert len(gt.true_positive_ids) == 5
        assert len(gt.benign_tp_ids) == 3
        assert len(gt.false_positive_ids) == 12


class TestScenarioDeterminism:
    def test_scenario_determinism(self):
        """Same seed should produce identical scenario configs."""
        config1 = PhishingScenario(seed=77).generate()
        config2 = PhishingScenario(seed=77).generate()

        assert config1.scenario_id == config2.scenario_id
        assert len(config1.alerts) == len(config2.alerts)
        assert config1.alerts[0].alert_id == config2.alerts[0].alert_id
        assert config1.alerts[0].title == config2.alerts[0].title
        assert config1.alerts[0].indicators == config2.alerts[0].indicators

        # Different seed should give different results
        config3 = PhishingScenario(seed=78).generate()
        assert config3.alerts[0].alert_id != config1.alerts[0].alert_id

    def test_lateral_movement_determinism(self):
        """Same seed produces identical lateral movement scenarios."""
        config1 = LateralMovementScenario(seed=99).generate()
        config2 = LateralMovementScenario(seed=99).generate()

        assert config1.scenario_id == config2.scenario_id
        for i in range(5):
            assert config1.alerts[i].alert_id == config2.alerts[i].alert_id


class TestGroundTruthCompleteness:
    def test_ground_truth_has_all_alerts(self):
        """Ground truth alert_classifications should cover every alert."""
        for ScenarioCls, task_id in [
            (PhishingScenario, "phishing"),
            (LateralMovementScenario, "lateral_movement"),
            (QueueManagementScenario, "queue_management"),
        ]:
            config = ScenarioCls(seed=42).generate()
            gt = config.ground_truth
            alert_ids = {a.alert_id for a in config.alerts}
            classified_ids = set(gt.alert_classifications.keys())
            assert alert_ids == classified_ids, (
                f"{task_id}: alert IDs {alert_ids} != classified IDs {classified_ids}"
            )
