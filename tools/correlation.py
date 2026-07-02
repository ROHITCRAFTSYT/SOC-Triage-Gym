"""
Alert Correlation Tool
========================
Pure function: checks if two alerts share any threat indicators.
Returns a CorrelatedEvent if a link is found, along with a step reward.

Reward logic:
  +0.20  if alerts share at least one indicator (correlation found)
  -0.03  if alerts share no indicators
   0.00  if this exact pair was already correlated this episode
"""


from models import AlertMeta, CorrelatedEvent, CorrelationType, InvestigationState, ScenarioConfig

# Map indicator key names to CorrelationType enum values
_INDICATOR_TO_CORRELATION_TYPE: dict[str, CorrelationType] = {
    "ip": CorrelationType.SOURCE_IP,
    "domain": CorrelationType.DOMAIN,
    "file_hash": CorrelationType.FILE_HASH,
    "email": CorrelationType.SOURCE_IP,  # email sender correlates as source
    "url": CorrelationType.DOMAIN,
    "user": CorrelationType.USER,
    "hostname": CorrelationType.HOSTNAME,
}


def correlate_alerts(
    config: ScenarioConfig,
    investigations: dict[str, InvestigationState],
    alert_id_a: str,
    alert_id_b: str,
) -> tuple[CorrelatedEvent | None, float, str]:
    """
    Check whether two alerts share any threat indicators.

    Args:
        config: The current scenario configuration.
        investigations: All per-alert investigation states (to detect duplicate correlations).
        alert_id_a: First alert ID.
        alert_id_b: Second alert ID.

    Returns:
        (CorrelatedEvent or None, step_reward, message)
    """
    if alert_id_a == alert_id_b:
        return None, -0.03, "Cannot correlate an alert with itself."

    # Check if both alerts exist
    alert_a = _find_alert(config, alert_id_a)
    alert_b = _find_alert(config, alert_id_b)

    if alert_a is None:
        return None, -0.03, f"Alert {alert_id_a} not found in this episode."
    if alert_b is None:
        return None, -0.03, f"Alert {alert_id_b} not found in this episode."

    # Check for duplicate correlation
    pair_key = tuple(sorted([alert_id_a, alert_id_b]))
    inv_a = investigations.get(alert_id_a)
    if inv_a:
        for existing in inv_a.correlations_found:
            existing_pair = tuple(sorted(existing.alert_ids))
            if existing_pair == pair_key:
                return existing, 0.0, f"Already correlated {alert_id_a} ↔ {alert_id_b}."

    # Find shared indicators across all indicator types
    shared = _find_shared_indicators(alert_a, alert_b)

    if not shared:
        return None, -0.03, f"No shared indicators between {alert_id_a} and {alert_id_b}."

    # Build correlation event from the strongest shared indicator
    indicator_type_key, shared_value = shared[0]
    corr_type = _INDICATOR_TO_CORRELATION_TYPE.get(indicator_type_key, CorrelationType.TIME_WINDOW)

    # Bonus reward if this correlation was expected (in kill chain)
    bonus = 0.0
    if config.ground_truth.kill_chain_order:
        chain = config.ground_truth.kill_chain_order
        for i in range(len(chain) - 1):
            if {chain[i], chain[i + 1]} == {alert_id_a, alert_id_b}:
                bonus = 0.10  # Extra reward for finding kill chain link
                break

    event = CorrelatedEvent(
        alert_ids=[alert_id_a, alert_id_b],
        correlation_type=corr_type,
        shared_indicator=shared_value,
        description=(
            f"Alerts {alert_id_a} and {alert_id_b} share {corr_type.value}: '{shared_value}'. "
            f"{len(shared)} total shared indicator(s) found."
        ),
        confidence=min(0.5 + 0.1 * len(shared), 1.0),
        relevance_score=0.8 + bonus,
    )

    reward = 0.20 + bonus
    msg = (
        f"Correlation found: {alert_id_a} ↔ {alert_id_b} via {corr_type.value} "
        f"'{shared_value}'. {len(shared)} shared indicator(s)."
    )
    if bonus > 0:
        msg += " [Kill chain link detected!]"

    return event, reward, msg


def _find_alert(config: ScenarioConfig, alert_id: str) -> AlertMeta | None:
    """Find an alert in the config by ID."""
    for alert in config.alerts:
        if alert.alert_id == alert_id:
            return alert
    return None


def _find_shared_indicators(
    alert_a: AlertMeta,
    alert_b: AlertMeta,
) -> list[tuple[str, str]]:
    """
    Return list of (indicator_type, shared_value) pairs that appear in both alerts.
    Sorted by indicator type priority (IPs and domains first).
    """
    shared = []
    priority_order = ["ip", "domain", "file_hash", "user", "hostname", "email", "url"]

    for itype in priority_order:
        values_a = set(alert_a.indicators.get(itype, []))
        values_b = set(alert_b.indicators.get(itype, []))
        overlap = values_a & values_b
        for val in sorted(overlap):
            shared.append((itype, val))

    return shared
