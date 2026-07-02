"""
SIEM Log Query Tool
====================
Pure function: queries a specific log source for events related to an alert.
Returns matching log entries and a step reward signal.

Reward logic:
  +0.10  if this log source is listed in ground_truth.relevant_log_sources for the alert
  -0.05  if this log source is irrelevant to the alert (no evidence there)
  -0.03  if this exact source was already queried for this alert (duplicate)
   0.00  if source is relevant but returns empty results
"""


from models import InvestigationState, LogEntry, LogSource, ScenarioConfig


def query_logs(
    config: ScenarioConfig,
    investigation: InvestigationState,
    log_source: LogSource,
    alert_id: str,
    time_window_hours: int = 24,
) -> tuple[list[LogEntry], float, str]:
    """
    Query a log source for events related to a specific alert.

    Args:
        config: The current scenario configuration (contains log_db and ground_truth).
        investigation: The active alert's investigation state (used to detect duplicates).
        log_source: Which SIEM log source to query.
        alert_id: Which alert's context to query against.
        time_window_hours: Time window for query (cosmetic in simulation).

    Returns:
        (List[LogEntry], step_reward, message)
    """
    source_key = log_source.value

    # Check for duplicate query
    if source_key in investigation.queried_sources:
        msg = f"Already queried {source_key} for alert {alert_id}. No new results."
        return investigation.queried_sources[source_key], -0.03, msg

    # Fetch from log database
    source_db = config.log_db.get(source_key, {})
    entries: list[LogEntry] = source_db.get(alert_id, [])

    # Determine reward
    relevant_sources = config.ground_truth.relevant_log_sources.get(alert_id, [])
    is_relevant = log_source in relevant_sources

    if is_relevant:
        if entries:
            reward = 0.10
            msg = f"Found {len(entries)} relevant log entries in {source_key}."
        else:
            reward = 0.0
            msg = f"{source_key} is relevant to this alert but returned no entries in the queried window."
    else:
        reward = -0.05
        msg = f"No relevant activity found in {source_key} for this alert."

    return entries, reward, msg
