"""
Lateral Movement Grader — Task 2 (Medium)
==========================================
Scores agent performance on the 5-alert kill chain investigation.

Weights:
  classification  0.30  — correct TP classification for each of 5 alerts (averaged)
  technique       0.20  — correct ATT&CK technique for each alert (averaged)
  chain_score     0.20  — kill chain reconstruction (correlated all adjacent pairs)
  response        0.20  — appropriate containment per alert phase
  efficiency      0.10  — steps used vs. budget
"""

from typing import Dict

from models import InvestigationState, ScenarioConfig
from graders.base import BaseGrader


class LateralMovementGrader(BaseGrader):
    """Grader for Task 2: Multi-Alert Lateral Movement Kill Chain."""

    def grade(
        self,
        config: ScenarioConfig,
        investigations: Dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """Grade a lateral movement episode. Returns 0.0–1.0."""

        gt = config.ground_truth

        # ---- Classification Score (0.30) ----
        classification_score = self._classification_accuracy(config, investigations)

        # ---- Technique Score (0.20) ----
        technique_score = self._technique_accuracy(config, investigations, only_tps=True)

        # ---- Kill Chain Reconstruction Score (0.20) ----
        chain_score = self._chain_reconstruction_score(investigations, gt.kill_chain_order or [])

        # ---- Response Score (0.20) ----
        response_score = self._response_quality(config, investigations)

        # ---- Efficiency Score (0.10) ----
        efficiency_score = self._efficiency_score(steps_used, max_steps)

        # ---- Weighted sum ----
        final = (
            0.30 * classification_score
            + 0.20 * technique_score
            + 0.20 * chain_score
            + 0.20 * response_score
            + 0.10 * efficiency_score
        )

        return self._clamp(final)

    def grade_with_breakdown(self, config, investigations, steps_used, max_steps):
        """Grade and return (score, breakdown, feedback)."""
        gt = config.ground_truth
        classification_score = self._classification_accuracy(config, investigations)
        technique_score = self._technique_accuracy(config, investigations, only_tps=True)
        chain_score = self._chain_reconstruction_score(investigations, gt.kill_chain_order or [])
        response_score = self._response_quality(config, investigations)
        efficiency_score = self._efficiency_score(steps_used, max_steps)

        final = self._clamp(
            0.30 * classification_score + 0.20 * technique_score
            + 0.20 * chain_score + 0.20 * response_score + 0.10 * efficiency_score
        )

        feedback_parts = []
        if classification_score < 0.8:
            feedback_parts.append(f"Classification accuracy {int(classification_score*100)}% — some alerts misclassified.")
        if chain_score < 0.8:
            feedback_parts.append(f"Kill chain reconstruction {int(chain_score*100)}% — correlate adjacent alerts.")
        if technique_score < 0.8:
            feedback_parts.append(f"ATT&CK technique mapping {int(technique_score*100)}%.")

        return final, {
            "classification": round(classification_score, 3),
            "technique_mapping": round(technique_score, 3),
            "kill_chain_reconstruction": round(chain_score, 3),
            "response_quality": round(response_score, 3),
            "efficiency": round(efficiency_score, 3),
        }, " ".join(feedback_parts) or "Strong performance across all components."

    def _chain_reconstruction_score(
        self,
        investigations: Dict[str, InvestigationState],
        kill_chain_order: list,
    ) -> float:
        """
        Score how well the agent reconstructed the kill chain.
        For each consecutive pair in kill_chain_order, check if the agent
        found a CorrelatedEvent linking those two alerts.
        """
        if len(kill_chain_order) < 2:
            return 0.0

        # Collect all correlated pairs found by agent
        found_pairs: set = set()
        for inv in investigations.values():
            for corr_event in inv.correlations_found:
                if len(corr_event.alert_ids) == 2:
                    pair = tuple(sorted(corr_event.alert_ids))
                    found_pairs.add(pair)

        # Check each adjacent pair in the kill chain
        chain_pairs = [
            tuple(sorted([kill_chain_order[i], kill_chain_order[i + 1]]))
            for i in range(len(kill_chain_order) - 1)
        ]

        matched = sum(1 for pair in chain_pairs if pair in found_pairs)
        return matched / len(chain_pairs)
