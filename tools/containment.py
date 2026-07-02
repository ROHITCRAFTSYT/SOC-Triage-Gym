"""
Tier-2 Responder Containment & Forensic Tools
===============================================
Pure functions: no side effects, all inputs passed explicitly.
Each function returns (ContainmentResult, float reward, str message).
"""

from __future__ import annotations

from models import (
    ContainmentResult,
    InvestigationState,
    ScenarioConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host_in_tp_alert(config: ScenarioConfig, target_host: str) -> bool:
    """Return True if target_host appears in indicators['hostname'] of any TP alert."""
    for alert in config.alerts:
        if alert.alert_id not in config.ground_truth.true_positive_ids:
            continue
        hostnames = alert.indicators.get("hostname", [])
        if target_host in hostnames:
            return True
    return False


def _user_in_tp_alert(config: ScenarioConfig, target_user: str) -> bool:
    """Return True if target_user appears in indicators['user'] of any TP alert."""
    for alert in config.alerts:
        if alert.alert_id not in config.ground_truth.true_positive_ids:
            continue
        users = alert.indicators.get("user", [])
        if target_user in users:
            return True
    return False


# ---------------------------------------------------------------------------
# Forensic tools
# ---------------------------------------------------------------------------

def forensic_timeline(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_host: str,
) -> tuple[ContainmentResult, float, str]:
    """Generate a forensic timeline for a host by scanning all log_db entries."""
    entries: list[str] = []
    for _source_name, alert_map in config.log_db.items():
        for _aid, log_list in alert_map.items():
            for entry in log_list:
                if entry.hostname == target_host:
                    user = entry.user or "—"
                    action = entry.action or entry.event_type
                    entries.append(
                        f"{entry.timestamp} {action} {user} {target_host}"
                    )

    timeline = entries[:10]
    is_relevant = _host_in_tp_alert(config, target_host)
    reward = 0.12 if is_relevant else 0.03

    result = ContainmentResult(
        action_type="forensic_timeline",
        target=target_host,
        success=True,
        details=f"Forensic timeline generated for host '{target_host}'. {len(timeline)} entries found.",
        evidence=[],
        timeline_entries=timeline,
    )
    msg = (
        f"Timeline for '{target_host}': {len(timeline)} log entries retrieved."
        + (" Host is relevant to a TP alert." if is_relevant else "")
    )
    return result, reward, msg


def sandbox_detonate(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_ioc: str,
) -> tuple[ContainmentResult, float, str]:
    """Simulate detonating a file hash or IOC in a sandbox environment."""
    enrichment = config.enrichment_db.get(target_ioc)

    if enrichment is None:
        result = ContainmentResult(
            action_type="sandbox_detonate",
            target=target_ioc,
            success=False,
            details=f"IOC '{target_ioc}' not found in threat intelligence database.",
            evidence=["No malicious behavior detected"],
        )
        return result, 0.02, f"Sandbox: '{target_ioc}' not in database — no malicious behavior detected."

    if enrichment.malicious:
        malware_names = (
            ", ".join(enrichment.associated_malware)
            if enrichment.associated_malware
            else "Unknown"
        )
        evidence = [
            f"Malware family: {malware_names}",
            f"Threat score: {enrichment.threat_score}",
        ]
        reward = 0.10
        msg = f"Sandbox MALICIOUS: '{target_ioc}' — {malware_names}, score {enrichment.threat_score}."
    else:
        evidence = ["No malicious behavior detected"]
        reward = 0.02
        msg = f"Sandbox CLEAN: '{target_ioc}' — no malicious behavior detected."

    result = ContainmentResult(
        action_type="sandbox_detonate",
        target=target_ioc,
        success=True,
        details=f"Sandbox detonation complete for IOC '{target_ioc}'.",
        evidence=evidence,
    )
    return result, reward, msg


def memory_analysis(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_host: str,
) -> tuple[ContainmentResult, float, str]:
    """Simulate memory dump analysis on a target host."""
    host_in_tp = _host_in_tp_alert(config, target_host)

    asset = config.asset_db.get(target_host)
    is_critical = asset is not None and asset.criticality == "critical"

    if is_critical and host_in_tp:
        evidence = [
            "Suspicious process injection detected",
            "C2 beacon in memory",
        ]
        reward = 0.10
        msg = f"Memory analysis of '{target_host}': critical host with suspicious artifacts found."
    else:
        evidence = ["Memory analysis complete — no suspicious artifacts"]
        reward = 0.04
        msg = f"Memory analysis of '{target_host}': no suspicious artifacts detected."

    result = ContainmentResult(
        action_type="memory_analysis",
        target=target_host,
        success=True,
        details=f"Memory analysis completed on host '{target_host}'.",
        evidence=evidence,
    )
    return result, reward, msg


# ---------------------------------------------------------------------------
# Containment tools
# ---------------------------------------------------------------------------

def isolate_host(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_host: str,
) -> tuple[ContainmentResult, float, str]:
    """Simulate host isolation / network quarantine."""
    correct = _host_in_tp_alert(config, target_host)

    if correct:
        reward = 0.20
        details = f"Host {target_host} isolated from network."
        msg = f"Host '{target_host}' correctly isolated — confirmed TP asset."
    else:
        reward = -0.15
        details = f"Host {target_host} isolation — verify threat first."
        msg = f"Host '{target_host}' isolation may be incorrect — not found in any TP alert indicators."

    result = ContainmentResult(
        action_type="isolate_host",
        target=target_host,
        success=True,
        details=details,
        evidence=[],
    )
    return result, reward, msg


def disable_user_account(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_user: str,
) -> tuple[ContainmentResult, float, str]:
    """Simulate disabling a user account."""
    is_tp_user = _user_in_tp_alert(config, target_user)

    if is_tp_user:
        reward = 0.18
        details = f"Account {target_user} disabled."
        msg = f"Account '{target_user}' disabled — confirmed TP user."
    else:
        reward = -0.15
        details = f"Account {target_user} disabled — verify threat before disabling benign accounts."
        msg = f"Account '{target_user}' disabled but not confirmed in any TP alert — possible false action."

    result = ContainmentResult(
        action_type="disable_user_account",
        target=target_user,
        success=True,
        details=details,
        evidence=[],
    )
    return result, reward, msg


def block_ioc(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    target_ioc: str,
    ioc_type: str,
) -> tuple[ContainmentResult, float, str]:
    """Simulate blocking an IOC (IP, domain, file hash) at the perimeter."""
    enrichment = config.enrichment_db.get(target_ioc)

    if enrichment is None:
        reward = 0.02
        msg = f"IOC '{target_ioc}' not found in threat intel — blocked with low confidence."
    elif enrichment.malicious:
        reward = 0.12
        msg = f"IOC '{target_ioc}' is known malicious — correctly blocked at perimeter."
    else:
        reward = -0.08
        msg = f"IOC '{target_ioc}' is known-clean in threat intel — unnecessary block."

    result = ContainmentResult(
        action_type="block_ioc",
        target=target_ioc,
        success=True,
        details=f"Blocked {ioc_type} {target_ioc} at perimeter.",
        evidence=[],
    )
    return result, reward, msg


def close_case(
    config: ScenarioConfig,
    inv: InvestigationState,
    alert_id: str,
    resolution: str,
) -> tuple[ContainmentResult, float, str]:
    """Record case closure with a resolution note."""
    already_closed = any(
        entry.startswith("Case closed:") for entry in inv.evidence_timeline
    )
    inv.evidence_timeline.append(f"Case closed: {resolution}")

    result = ContainmentResult(
        action_type="close_case",
        target=alert_id,
        success=True,
        details=f"Case {alert_id} closed. Resolution: {resolution}",
        evidence=[],
    )
    if already_closed:
        msg = f"Case '{alert_id}' already closed — duplicate close_case penalized."
        return result, -0.02, msg
    msg = f"Case '{alert_id}' closed with resolution: {resolution}"
    return result, 0.05, msg
