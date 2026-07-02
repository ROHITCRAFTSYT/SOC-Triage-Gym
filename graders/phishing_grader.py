"""
Phishing Grader — Task 1 (Easy)
================================
Scores agent performance on the single-alert phishing triage task.

Weights:
  classification  0.4  — correct TP/FP decision
  technique       0.2  — correct MITRE ATT&CK technique mapped (if TP)
  evidence        0.2  — relevant log sources queried
  response        0.2  — appropriate containment recommended
"""


from graders.base import BaseGrader
from models import AlertClassification, InvestigationState, ScenarioConfig


class PhishingGrader(BaseGrader):
    """Grader for Task 1: Single-Alert Phishing Triage."""

    def grade(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """Grade a phishing triage episode. Returns 0.0–1.0."""

        if not config.alerts:
            return 0.0

        alert_id = config.alerts[0].alert_id
        inv = investigations.get(alert_id)
        gt = config.ground_truth
        expected_class = gt.alert_classifications.get(alert_id)

        # ---- Classification Score (0.4) ----
        if inv is None or inv.classification is None:
            classification_score = 0.0
        elif inv.classification == expected_class:
            classification_score = 1.0
        else:
            classification_score = 0.0

        # ---- Technique Score (0.2) ----
        # Only scored if TP; if FP, full credit for not mapping a technique
        if expected_class == AlertClassification.FALSE_POSITIVE:
            # For FP: full technique credit if agent mapped nothing (or said N/A)
            technique_score = 1.0 if (inv is None or not inv.mapped_techniques) else 0.5
        else:
            expected_techniques = gt.expected_techniques.get(alert_id, [])
            if not expected_techniques:
                technique_score = 1.0
            elif inv is None or not inv.mapped_techniques:
                technique_score = 0.0
            else:
                mapped = set(inv.mapped_techniques)
                expected = set(expected_techniques)
                exact = mapped & expected
                # Partial credit for parent technique (T1566 instead of T1566.001)
                parent_credit = sum(
                    0.5 for t in expected
                    if t not in mapped and t.split(".")[0] in mapped
                )
                technique_score = min((len(exact) + parent_credit) / len(expected), 1.0)

        # ---- Evidence Score (0.2) ----
        relevant_sources = gt.relevant_log_sources.get(alert_id, [])
        if not relevant_sources:
            evidence_score = 1.0
        elif inv is None:
            evidence_score = 0.0
        else:
            queried = set(inv.queried_sources.keys())
            expected_sources = {s.value for s in relevant_sources}
            evidence_score = len(queried & expected_sources) / len(expected_sources)

        # ---- Response Score (0.2) ----
        expected_actions = gt.expected_response_actions.get(alert_id, [])
        if not expected_actions:
            response_score = 1.0
        elif inv is None:
            response_score = 0.0
        elif expected_class == AlertClassification.FALSE_POSITIVE:
            from models import ResponseActionType
            recommended = set(inv.recommended_actions)
            # For FP: reward no escalation; penalize if recommended aggressive action
            aggressive = {
                ResponseActionType.ISOLATE_ENDPOINT,
                ResponseActionType.BLOCK_IP,
                ResponseActionType.DISABLE_ACCOUNT,
            }
            if not recommended or ResponseActionType.NO_ACTION in recommended:
                response_score = 1.0
            elif recommended & aggressive:
                response_score = 0.0
            else:
                response_score = 0.5
        else:
            recommended = set(inv.recommended_actions)
            expected = set(expected_actions)
            if not expected:
                response_score = 1.0
            else:
                response_score = len(recommended & expected) / len(expected)

        # ---- Compute weighted final score ----
        final = (
            0.4 * classification_score
            + 0.2 * technique_score
            + 0.2 * evidence_score
            + 0.2 * response_score
        )

        if final <= 0.0:
            return 0.0
        return self._clamp(final)

    def grade_with_breakdown(self, config, investigations, steps_used, max_steps):
        """Grade and return (score, breakdown, feedback)."""
        if not config.alerts:
            return 0.0, {}, "No alerts in episode."

        alert_id = config.alerts[0].alert_id
        inv = investigations.get(alert_id)
        gt = config.ground_truth
        expected_class = gt.alert_classifications.get(alert_id)

        from models import AlertClassification
        if inv is None or inv.classification is None:
            classification_score = 0.0
        elif inv.classification == expected_class:
            classification_score = 1.0
        else:
            classification_score = 0.0

        if expected_class == AlertClassification.FALSE_POSITIVE:
            technique_score = 1.0 if (inv is None or not inv.mapped_techniques) else 0.5
        else:
            expected_techniques = gt.expected_techniques.get(alert_id, [])
            if not expected_techniques:
                technique_score = 1.0
            elif inv is None or not inv.mapped_techniques:
                technique_score = 0.0
            else:
                mapped = set(inv.mapped_techniques)
                expected = set(expected_techniques)
                exact = mapped & expected
                parent_credit = sum(0.5 for t in expected if t not in mapped and t.split(".")[0] in mapped)
                technique_score = min((len(exact) + parent_credit) / len(expected), 1.0)

        relevant_sources = gt.relevant_log_sources.get(alert_id, [])
        if not relevant_sources:
            evidence_score = 1.0
        elif inv is None:
            evidence_score = 0.0
        else:
            queried = set(inv.queried_sources.keys())
            expected_sources = {s.value for s in relevant_sources}
            evidence_score = len(queried & expected_sources) / len(expected_sources)

        expected_actions = gt.expected_response_actions.get(alert_id, [])
        if not expected_actions and expected_class != AlertClassification.FALSE_POSITIVE:
            response_score = 1.0
        elif inv is None:
            response_score = 0.0
        elif expected_class == AlertClassification.FALSE_POSITIVE:
            from models import ResponseActionType
            recommended = set(inv.recommended_actions)
            aggressive = {
                ResponseActionType.ISOLATE_ENDPOINT,
                ResponseActionType.BLOCK_IP,
                ResponseActionType.DISABLE_ACCOUNT,
            }
            if not recommended or ResponseActionType.NO_ACTION in recommended:
                response_score = 1.0
            elif recommended & aggressive:
                response_score = 0.0
            else:
                response_score = 0.5
        else:
            recommended = set(inv.recommended_actions)
            expected = set(expected_actions)
            response_score = len(recommended & expected) / len(expected) if expected else 1.0

        raw_final = (
            0.4 * classification_score + 0.2 * technique_score
            + 0.2 * evidence_score + 0.2 * response_score
        )
        final = 0.0 if raw_final <= 0.0 else self._clamp(raw_final)

        feedback_parts = []
        if classification_score < 1.0:
            feedback_parts.append(f"Alert classified as '{inv.classification if inv else 'unclassified'}', expected '{expected_class}'.")
        if technique_score < 1.0:
            feedback_parts.append(f"MITRE technique mapping incomplete ({int(technique_score*100)}%).")
        if evidence_score < 1.0:
            feedback_parts.append(f"Only {int(evidence_score*100)}% of relevant log sources queried.")
        if response_score < 1.0:
            feedback_parts.append(f"Response actions {int(response_score*100)}% complete.")

        return final, {
            "classification": round(classification_score, 3),
            "technique_mapping": round(technique_score, 3),
            "evidence_gathered": round(evidence_score, 3),
            "response_quality": round(response_score, 3),
        }, " ".join(feedback_parts) or "All components scored correctly."
