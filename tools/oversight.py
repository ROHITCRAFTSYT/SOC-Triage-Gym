"""
SOC Manager (Oversight) Tools
================================
Pure functions: no side effects beyond explicit mutations to investigations dict.
Each function returns (ManagerReviewResult, float reward, str message).
"""

from __future__ import annotations

from models import (
    ManagerReviewResult,
    ScenarioConfig,
    InvestigationState,
    AlertClassification,
    TicketMessage,
    TicketKind,
)
from graders.manager_judge import ManagerJudge


_MANAGER_JUDGE = ManagerJudge()


def review_decision(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
    ticket: TicketMessage,
) -> tuple[ManagerReviewResult, float, str]:
    """Review a Tier-1 escalation or Tier-2 closure ticket and score the decision."""
    alert_id = ticket.alert_id
    gt = config.ground_truth
    is_tp = alert_id in gt.true_positive_ids

    if ticket.kind == TicketKind.ESCALATION:
        if is_tp:
            reward = 0.12
            finding = f"Escalation of alert '{alert_id}' is correct — confirmed TP."
            msg = f"Escalation review: correct. Alert '{alert_id}' is a true positive."
        else:
            reward = 0.02
            finding = f"Escalation of alert '{alert_id}' appears incorrect — not a confirmed TP."
            msg = f"Escalation review: questionable. Alert '{alert_id}' is not a confirmed TP."
    elif ticket.kind == TicketKind.CLOSURE:
        inv = investigations.get(alert_id)
        agent_classification = inv.classification if inv is not None else None
        expected = gt.alert_classifications.get(alert_id)
        if agent_classification is not None and agent_classification == expected:
            reward = 0.10
            finding = f"Closure of alert '{alert_id}' is correct — classification matches ground truth."
            msg = f"Closure review: correct. Classification '{agent_classification.value}' matches ground truth."
        else:
            reward = 0.03
            expected_str = expected.value if expected is not None else "unknown"
            actual_str = agent_classification.value if agent_classification is not None else "unset"
            finding = (
                f"Closure discrepancy on alert '{alert_id}': "
                f"agent classified as '{actual_str}', expected '{expected_str}'."
            )
            msg = f"Closure review: discrepancy found on alert '{alert_id}'."
    else:
        reward = 0.01
        finding = f"Ticket kind '{ticket.kind.value}' reviewed — no specific check applied."
        msg = f"Ticket '{ticket.ticket_id}' reviewed."

    result = ManagerReviewResult(
        action_type="review_decision",
        ticket_id=ticket.ticket_id,
        alert_id=alert_id,
        finding=finding,
        override_applied=False,
    )
    return result, reward, msg


def override_classification(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
    alert_id: str,
    new_classification: AlertClassification,
) -> tuple[ManagerReviewResult, float, str]:
    """Override the classification of an alert and score the correction."""
    inv = investigations.get(alert_id)
    current = inv.classification if inv is not None else None
    expected = config.ground_truth.alert_classifications.get(alert_id)

    already_correct = current == expected

    if already_correct:
        reward = -0.05
        override_applied = False
        finding = (
            f"Classification of alert '{alert_id}' was already correct ('{current.value if current else 'unset'}')."
            " Override unnecessary."
        )
        msg = f"Override unnecessary — alert '{alert_id}' was already correctly classified."
    else:
        # Apply override regardless of direction
        if inv is not None:
            inv.classification = new_classification
        override_applied = True
        if new_classification == expected:
            reward = 0.25
            finding = (
                f"Override of alert '{alert_id}' to '{new_classification.value}' is correct — matches ground truth."
            )
            msg = f"Override applied: alert '{alert_id}' corrected to '{new_classification.value}'."
        else:
            reward = -0.20
            expected_str = expected.value if expected is not None else "unknown"
            finding = (
                f"Override of alert '{alert_id}' to '{new_classification.value}' is wrong — "
                f"ground truth is '{expected_str}'."
            )
            msg = f"Override applied but incorrect — alert '{alert_id}' overridden to '{new_classification.value}', expected '{expected_str}'."

    result = ManagerReviewResult(
        action_type="override_classification",
        alert_id=alert_id,
        finding=finding,
        override_applied=override_applied,
    )
    return result, reward, msg


def flag_inconsistency(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
    alert_id: str,
    flag_reason: str,
) -> tuple[ManagerReviewResult, float, str]:
    """Flag an inconsistency in Tier-1 or Tier-2 handling of an alert."""
    expected_flags = config.ground_truth.expected_manager_flags

    if alert_id in expected_flags:
        reward = 0.15
        inconsistency_found = True
        finding = f"Flagged: {flag_reason}"
        msg = f"Legitimate inconsistency flagged on alert '{alert_id}'."
    else:
        reward = -0.15
        inconsistency_found = False
        finding = f"Flagged: {flag_reason}"
        msg = f"Spurious flag on alert '{alert_id}' — not in expected manager flags."

    result = ManagerReviewResult(
        action_type="flag_inconsistency",
        alert_id=alert_id,
        finding=finding,
        inconsistency_found=inconsistency_found,
    )
    return result, reward, msg


def explain_team_behavior(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
    tickets: list[TicketMessage],
    explanation_text: str,
    episode_id: str = "unknown",
    seed: int = 42,
    trajectory_hash: str = "",
) -> tuple[ManagerReviewResult, float, str]:
    """Manager provides an explanation scored by the LLM judge with heuristic fallback."""
    judge_score = _MANAGER_JUDGE.judge(
        explanation=explanation_text,
        investigations=investigations,
        config=config,
        episode_id=episode_id,
        seed=seed,
        trajectory_hash=trajectory_hash,
    )
    reward = round(0.15 * judge_score, 4)

    result = ManagerReviewResult(
        action_type="explain_team_behavior",
        finding="Explanation recorded.",
        explanation=explanation_text,
    )
    msg = f"Team behavior explanation recorded (judge score: {judge_score:.2f}, reward: {reward:.2f})."
    return result, reward, msg
