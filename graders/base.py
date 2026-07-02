"""
Base Grader
============
Abstract base class for all SOC-Triage-Gym graders.
All graders receive the complete ScenarioConfig (containing ground truth)
and the episode's InvestigationState dict, returning a float in [0.0, 1.0].
"""

from abc import ABC, abstractmethod

from models import AlertClassification, InvestigationState, ScenarioConfig


class BaseGrader(ABC):
    """Abstract base for task graders. All graders return float in [0.0, 1.0]."""

    @abstractmethod
    def grade(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """
        Grade the agent's performance on an episode.

        Args:
            config: Full scenario config including ground_truth.
            investigations: Dict of alert_id → InvestigationState.
            steps_used: Number of steps the agent took.
            max_steps: Maximum allowed steps.

        Returns:
            Score in [0.0, 1.0].
        """
        ...

    def grade_with_breakdown(
        self,
        config: "ScenarioConfig",
        investigations: "dict[str, InvestigationState]",
        steps_used: int,
        max_steps: int,
    ) -> tuple:
        """
        Grade and return (score, breakdown_dict, feedback_str).
        Subclasses override this for per-component breakdowns.
        Default delegates to grade() with a minimal breakdown.
        """
        score = self.grade(config, investigations, steps_used, max_steps)
        return score, {"total": score}, ""

    # ------------------------------------------------------------------
    # Shared helper methods
    # ------------------------------------------------------------------

    def _classification_accuracy(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
    ) -> float:
        """Fraction of alerts correctly classified (unclassified = wrong)."""
        gt = config.ground_truth.alert_classifications
        if not gt:
            return 1.0
        correct = sum(
            1
            for alert_id, expected in gt.items()
            if investigations.get(alert_id, InvestigationState(alert_id=alert_id)).classification == expected
        )
        return correct / len(gt)

    def _technique_accuracy(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        only_tps: bool = True,
    ) -> float:
        """
        Fraction of expected MITRE ATT&CK techniques that were mapped.
        If only_tps=True, only scores technique mapping for true positive alerts.
        """
        expected_map = config.ground_truth.expected_techniques
        if not expected_map:
            return 1.0

        total_score = 0.0
        total_alerts = 0

        for alert_id, expected_techniques in expected_map.items():
            if not expected_techniques:
                continue

            # Skip FP alerts for technique scoring if only_tps
            if only_tps:
                gt_class = config.ground_truth.alert_classifications.get(alert_id)
                if gt_class == AlertClassification.FALSE_POSITIVE:
                    continue

            inv = investigations.get(alert_id)
            if inv is None:
                total_alerts += 1
                continue

            mapped = set(inv.mapped_techniques)
            expected = set(expected_techniques)

            # Give partial credit: exact match + parent technique credit
            exact_matches = mapped & expected
            # Parent technique credit (e.g., T1566 for T1566.001)
            parent_credits = sum(
                1 for t in expected
                if t not in mapped and t.split(".")[0] in mapped
            ) * 0.5

            alert_score = (len(exact_matches) + parent_credits) / len(expected)
            total_score += min(alert_score, 1.0)
            total_alerts += 1

        return total_score / total_alerts if total_alerts > 0 else 1.0

    def _evidence_completeness(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
    ) -> float:
        """
        Fraction of relevant log sources that were queried across all alerts.
        """
        relevant_sources = config.ground_truth.relevant_log_sources
        if not relevant_sources:
            return 1.0

        total_relevant = 0
        total_queried = 0

        for alert_id, sources in relevant_sources.items():
            if not sources:
                continue
            inv = investigations.get(alert_id)
            queried = set(inv.queried_sources.keys()) if inv else set()
            expected = {s.value for s in sources}
            total_queried += len(queried & expected)
            total_relevant += len(expected)

        return total_queried / total_relevant if total_relevant > 0 else 1.0

    def _response_quality(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
    ) -> float:
        """
        Fraction of expected response actions that were recommended,
        only for true positive alerts.
        """
        expected_map = config.ground_truth.expected_response_actions
        if not expected_map:
            return 1.0

        total_score = 0.0
        total_alerts = 0

        for alert_id, expected_actions in expected_map.items():
            if not expected_actions:
                continue

            gt_class = config.ground_truth.alert_classifications.get(alert_id)

            inv = investigations.get(alert_id)
            recommended = set(inv.recommended_actions) if inv else set()
            expected = set(expected_actions)

            if gt_class == AlertClassification.FALSE_POSITIVE:
                # For FPs, score whether agent correctly recommended no_action
                from models import ResponseActionType
                if ResponseActionType.NO_ACTION in recommended or not recommended:
                    total_score += 1.0
                else:
                    total_score += 0.0
            else:
                # For TPs/BTPs, score overlap with expected actions
                if expected:
                    overlap = recommended & expected
                    total_score += len(overlap) / len(expected)

            total_alerts += 1

        return total_score / total_alerts if total_alerts > 0 else 1.0

    def _efficiency_score(self, steps_used: int, max_steps: int) -> float:
        """Score based on steps used vs budget. Lower is better."""
        if max_steps <= 0:
            return 0.0
        ratio = steps_used / max_steps
        if ratio <= 0.4:
            return 1.0
        if ratio <= 0.6:
            return 0.8
        if ratio <= 0.8:
            return 0.6
        return 0.3

    def _clamp(self, value: float) -> float:
        """Clamp a score to strictly (0.0, 1.0) — exclusive, as required by OpenEnv validator."""
        return max(0.001, min(0.999, value))
