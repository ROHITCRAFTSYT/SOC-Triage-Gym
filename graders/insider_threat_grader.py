"""
Insider Threat Grader -- Task 4 (Expert)
=========================================
Scores agent performance on the 30-alert insider threat investigation.

Weights:
  f1_score            0.25  -- F1 on TP/FP classifications across all 30 alerts
  attack_chain_score  0.25  -- identified the 3 hidden attack chains
  missed_tp_penalty   0.20  -- heavy penalty per missed true positive
  efficiency_score    0.15  -- steps used vs. budget
  response_score      0.15  -- quality of response for actual TPs
"""


from graders.base import BaseGrader
from models import AlertClassification, InvestigationState, ScenarioConfig


class InsiderThreatGrader(BaseGrader):
    """Grader for Task 4: Insider Threat Investigation."""

    def grade(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """Grade an insider threat episode. Returns 0.0-1.0."""

        gt = config.ground_truth

        f1 = self._compute_f1(investigations, gt)
        chain_score = self._attack_chain_score(investigations, gt)
        missed_tp_score = self._missed_tp_score(investigations, gt)
        efficiency_score = self._efficiency_score(steps_used, max_steps)
        response_score = self._response_quality_score(config, investigations)

        final = (
            0.25 * f1
            + 0.25 * chain_score
            + 0.20 * missed_tp_score
            + 0.15 * efficiency_score
            + 0.15 * response_score
        )

        return self._clamp(final)

    def grade_with_breakdown(self, config, investigations, steps_used, max_steps):
        """Grade and return (score, breakdown, feedback)."""
        gt = config.ground_truth
        f1 = self._compute_f1(investigations, gt)
        chain_score = self._attack_chain_score(investigations, gt)
        missed_tp_score = self._missed_tp_score(investigations, gt)
        efficiency_score = self._efficiency_score(steps_used, max_steps)
        response_score = self._response_quality_score(config, investigations)

        final = self._clamp(
            0.25 * f1 + 0.25 * chain_score + 0.20 * missed_tp_score
            + 0.15 * efficiency_score + 0.15 * response_score
        )

        # Count unclassified alerts
        unclassified = sum(1 for inv in investigations.values() if inv.classification is None)
        feedback_parts = []
        if unclassified > 0:
            feedback_parts.append(f"{unclassified} alerts left unclassified.")
        if f1 < 0.8:
            feedback_parts.append(f"F1 score {int(f1*100)}% -- check FP dismissal and TP escalation balance.")
        if chain_score < 1.0:
            chains_pct = int(chain_score * 100)
            feedback_parts.append(f"Only {chains_pct}% of attack chains identified (3 chains expected).")
        if missed_tp_score < 1.0:
            feedback_parts.append("Missed true positives -- insider threats went undetected.")
        if efficiency_score < 0.6:
            feedback_parts.append("Low efficiency -- too many steps used relative to budget.")

        return final, {
            "f1_score": round(f1, 3),
            "attack_chains_found": round(chain_score, 3),
            "true_positive_coverage": round(missed_tp_score, 3),
            "efficiency": round(efficiency_score, 3),
            "response_quality": round(response_score, 3),
        }, " ".join(feedback_parts) or "Excellent insider threat investigation performance."

    def _compute_f1(
        self,
        investigations: dict[str, InvestigationState],
        gt,
    ) -> float:
        """Compute F1 score treating TP+BTP as positive, FP as negative."""
        actual_positives: set[str] = set(gt.true_positive_ids + gt.benign_tp_ids)
        actual_negatives: set[str] = set(gt.false_positive_ids)

        positive_classes = {AlertClassification.TRUE_POSITIVE, AlertClassification.BENIGN_TRUE_POSITIVE}
        negative_class = AlertClassification.FALSE_POSITIVE

        tp_count = 0
        fp_count = 0
        fn_count = 0

        for alert_id in actual_positives:
            inv = investigations.get(alert_id)
            classified = inv.classification if inv else None
            if classified in positive_classes:
                tp_count += 1
            elif classified == negative_class:
                fn_count += 1
            else:
                fn_count += 1  # Unclassified = missed positive

        for alert_id in actual_negatives:
            inv = investigations.get(alert_id)
            classified = inv.classification if inv else None
            if classified in positive_classes:
                fp_count += 1

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return f1

    def _attack_chain_score(
        self,
        investigations: dict[str, InvestigationState],
        gt,
    ) -> float:
        """
        Score whether agent found all 3 hidden attack chains.
        Each chain scores 1/3; combined = 1.0 if all found.
        A chain is "found" if all alerts in it were classified as TP.
        """
        if not gt.attack_chain_ids:
            return 1.0

        chains_found = 0
        positive_classes = {AlertClassification.TRUE_POSITIVE, AlertClassification.BENIGN_TRUE_POSITIVE}

        for chain in gt.attack_chain_ids:
            if not chain:
                continue
            all_found = all(
                investigations.get(aid, InvestigationState(alert_id=aid)).classification
                in positive_classes
                for aid in chain
            )
            if all_found:
                chains_found += 1

        return chains_found / len(gt.attack_chain_ids)

    def _missed_tp_score(
        self,
        investigations: dict[str, InvestigationState],
        gt,
    ) -> float:
        """
        Compute missed TP penalty.
        Start at 1.0, subtract per missed TP.
        9 TPs total: each missed costs ~0.11.
        BTP misses cost 0.05 each (5 BTPs).
        Score floor is 0.0.
        """
        positive_classes = {AlertClassification.TRUE_POSITIVE, AlertClassification.BENIGN_TRUE_POSITIVE}
        score = 1.0

        for alert_id in gt.true_positive_ids:
            inv = investigations.get(alert_id)
            classified = inv.classification if inv else None
            if classified not in positive_classes:
                score -= 0.11  # ~0.11 per missed TP (9 TPs -> can zero out)

        for alert_id in gt.benign_tp_ids:
            inv = investigations.get(alert_id)
            classified = inv.classification if inv else None
            if classified not in positive_classes:
                score -= 0.05  # 0.05 per missed BTP (5 BTPs)

        return max(0.0, score)

    def _response_quality_score(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
    ) -> float:
        """
        Fraction of expected response actions recommended for TPs.
        Delegates to the base class _response_quality method.
        """
        return self._response_quality(config, investigations)
