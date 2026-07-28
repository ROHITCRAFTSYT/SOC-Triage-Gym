"""
Scenario Integrity Validator
============================
A generated ``ScenarioConfig`` is the ground truth an episode is graded against.
If its answer key references alert IDs that don't exist, or its true/false/benign
partitions overlap, every reward computed from it is silently wrong — a
correctness bug a training run would never surface on its own.

``validate_scenario`` returns a list of human-readable issue strings (empty means
the scenario is internally consistent). Checks are deliberately conservative: they
only assert invariants a well-formed scenario must satisfy, so they never
false-positive on legitimately hard/noisy scenarios.
"""

from __future__ import annotations

from models import ScenarioConfig


def validate_scenario(config: ScenarioConfig) -> list[str]:
    """Return a list of integrity issues; an empty list means the scenario is valid."""
    issues: list[str] = []
    alert_ids = [a.alert_id for a in config.alerts]
    id_set = set(alert_ids)

    # --- structural basics ------------------------------------------------
    if not config.alerts:
        issues.append("scenario has no alerts")
    if config.max_steps <= 0:
        issues.append(f"max_steps must be positive, got {config.max_steps}")
    if len(alert_ids) != len(id_set):
        dupes = sorted({aid for aid in alert_ids if alert_ids.count(aid) > 1})
        issues.append(f"duplicate alert_id(s): {', '.join(dupes)}")

    gt = config.ground_truth

    # --- every ground-truth reference must resolve to a real alert --------
    def _check_ids(label: str, ids: list[str]) -> None:
        missing = [i for i in ids if i not in id_set]
        if missing:
            issues.append(f"{label} references unknown alert_id(s): {', '.join(sorted(set(missing)))}")

    _check_ids("alert_classifications", list(gt.alert_classifications))
    _check_ids("true_positive_ids", gt.true_positive_ids)
    _check_ids("false_positive_ids", gt.false_positive_ids)
    _check_ids("benign_tp_ids", gt.benign_tp_ids)
    _check_ids("expected_techniques", list(gt.expected_techniques))
    _check_ids("expected_response_actions", list(gt.expected_response_actions))
    _check_ids("relevant_indicators", list(gt.relevant_indicators))
    _check_ids("relevant_log_sources", list(gt.relevant_log_sources))
    _check_ids("required_escalations", gt.required_escalations)
    _check_ids("required_containments", list(gt.required_containments))
    _check_ids("expected_manager_flags", gt.expected_manager_flags)
    if gt.kill_chain_order:
        _check_ids("kill_chain_order", gt.kill_chain_order)
    for chain in gt.attack_chain_ids:
        _check_ids("attack_chain_ids", chain)

    # --- the TP / FP / benign partitions must be mutually exclusive -------
    tp, fp, benign = set(gt.true_positive_ids), set(gt.false_positive_ids), set(gt.benign_tp_ids)
    for a_name, a_set, b_name, b_set in (
        ("true_positive", tp, "false_positive", fp),
        ("true_positive", tp, "benign_tp", benign),
        ("false_positive", fp, "benign_tp", benign),
    ):
        overlap = a_set & b_set
        if overlap:
            issues.append(
                f"alert(s) in both {a_name} and {b_name}: {', '.join(sorted(overlap))}"
            )

    # --- every alert should have a ground-truth classification -----------
    unclassified = [aid for aid in alert_ids if aid not in gt.alert_classifications]
    if unclassified:
        issues.append(
            f"alert(s) missing a ground-truth classification: {', '.join(sorted(unclassified))}"
        )

    return issues


def assert_valid(config: ScenarioConfig) -> None:
    """Raise ``ValueError`` if the scenario has any integrity issues."""
    issues = validate_scenario(config)
    if issues:
        raise ValueError(
            f"Scenario {config.scenario_id!r} failed integrity validation:\n  - "
            + "\n  - ".join(issues)
        )
