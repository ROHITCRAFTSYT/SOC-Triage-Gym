"""
Team Grader
===========
Grades team-mode episodes on collective outcome using the documented
team-shaped outcome:

  team_score = TP-containment-rate * FP-dismissal-rate

TP-containment-rate: fraction of TP alerts correctly classified AND
having at least one containment action recommended (escalated=True
also counts as a containment signal for Tier-2 cases).

FP-dismissal-rate: fraction of FP alerts correctly classified as
false_positive.
"""


from graders.base import BaseGrader
from models import AlertClassification, InvestigationState, ScenarioConfig


def compute_team_metrics(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
) -> tuple[float, float, float]:
    """Return (team_score, tp_containment_rate, fp_dismissal_rate)."""
    gt = config.ground_truth

    # ---- TP-containment-rate ----
    tp_ids = [
        alert_id
        for alert_id, cls in gt.alert_classifications.items()
        if cls != AlertClassification.FALSE_POSITIVE
    ]
    if tp_ids:
        tp_contained = 0
        for alert_id in tp_ids:
            inv = investigations.get(alert_id)
            if inv is None:
                continue
            correct_class = inv.classification == gt.alert_classifications[alert_id]
            has_containment = bool(inv.recommended_actions) or inv.escalated
            if correct_class and has_containment:
                tp_contained += 1
        tp_containment_rate = tp_contained / len(tp_ids)
    else:
        tp_containment_rate = 1.0  # vacuously true — no TPs to miss

    # ---- FP-dismissal-rate ----
    fp_ids = gt.false_positive_ids
    if fp_ids:
        fp_dismissed = sum(
            1
            for alert_id in fp_ids
            if investigations.get(alert_id, InvestigationState(alert_id=alert_id))
            .classification == AlertClassification.FALSE_POSITIVE
        )
        fp_dismissal_rate = fp_dismissed / len(fp_ids)
    else:
        fp_dismissal_rate = 1.0  # vacuously true — no FPs to miss

    team_score = tp_containment_rate * fp_dismissal_rate
    return team_score, tp_containment_rate, fp_dismissal_rate


def compute_team_f1(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
) -> float:
    """
    Backward-compatible helper retained for callers.
    Despite the name, this returns the documented team-shaped score.
    """
    team_score, _, _ = compute_team_metrics(config, investigations)
    return team_score


class TeamGrader(BaseGrader):
    """Grader for team-mode episodes — scores collective TP containment and FP dismissal."""

    def grade(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """
        Compute team F1 and return a score in (0.001, 0.999).

        Args:
            config: Full scenario config including ground_truth.
            investigations: Dict of alert_id → InvestigationState.
            steps_used: Number of steps taken (unused — no efficiency penalty here).
            max_steps: Maximum allowed steps (unused).

        Returns:
            Score in (0.001, 0.999).
        """
        raw, _, _ = compute_team_metrics(config, investigations)
        return self._clamp(raw)

    def grade_with_breakdown(
        self,
        config: ScenarioConfig,
        investigations: dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> tuple:
        """
        Grade and return (score, breakdown_dict, feedback_str).

        Returns:
            (score, {"tp_containment": float, "fp_dismissal": float,
                     "team_f1": float, "total": float}, feedback_str)
        """
        gt = config.ground_truth
        team_score, tp_containment_rate, fp_dismissal_rate = compute_team_metrics(config, investigations)
        total_tps = len([
            alert_id
            for alert_id, cls in gt.alert_classifications.items()
            if cls != AlertClassification.FALSE_POSITIVE
        ])
        tps_contained = int(round(tp_containment_rate * total_tps)) if total_tps else 0
        total_fps = len(gt.false_positive_ids)
        fps_dismissed = int(round(fp_dismissal_rate * total_fps)) if total_fps else 0
        score = self._clamp(team_score)

        breakdown = {
            "tp_containment": tp_containment_rate,
            "fp_dismissal": fp_dismissal_rate,
            "team_score": team_score,
            "total": score,
        }

        feedback_str = (
            f"Team score: {score:.3f}. "
            f"TPs: {tps_contained}/{total_tps} contained "
            f"(TP-containment-rate={tp_containment_rate:.3f}). "
            f"FPs: {fps_dismissed}/{total_fps} dismissed "
            f"(FP-dismissal-rate={fp_dismissal_rate:.3f})."
        )

        return score, breakdown, feedback_str
