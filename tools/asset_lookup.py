"""
Asset Inventory Lookup Tool
=============================
Pure function: looks up a host in the scenario's asset database.

Reward logic:
  +0.05  if hostname appears in any log entry already seen by the agent
   0.00  for neutral/unknown lookups
  -0.03  if hostname doesn't exist in the asset DB and wasn't seen in any logs
"""


from models import AssetInfo, InvestigationState, ScenarioConfig


def lookup_asset(
    config: ScenarioConfig,
    investigation: InvestigationState,
    hostname: str,
) -> tuple[AssetInfo | None, float, str]:
    """
    Look up a host in the asset inventory.

    Args:
        config: The current scenario configuration (contains asset_db).
        investigation: The active investigation state (used to score relevance).
        hostname: The hostname to look up.

    Returns:
        (AssetInfo or None, step_reward, message)
    """
    # Check for duplicate lookup
    if hostname in investigation.assets_looked_up:
        return investigation.assets_looked_up[hostname], 0.0, f"Already looked up asset '{hostname}'."

    asset = config.asset_db.get(hostname)

    if asset is None:
        # Check if it appeared in any log results (agent may be following a valid lead)
        seen_in_logs = _hostname_in_logs(investigation, hostname)
        if seen_in_logs:
            reward = 0.0
            msg = f"Asset '{hostname}' seen in logs but not found in asset inventory. May be unmanaged."
        else:
            reward = -0.03
            msg = f"Asset '{hostname}' not found in asset inventory."
        # Return minimal stub
        return None, reward, msg

    # Check if this hostname was seen in collected log evidence
    seen_in_logs = _hostname_in_logs(investigation, hostname)
    reward = 0.05 if seen_in_logs else 0.02
    msg = (
        f"Asset found: {hostname} ({asset.asset_type}, criticality={asset.criticality}, "
        f"owner={asset.owner}, dept={asset.department})"
    )

    return asset, reward, msg


def _hostname_in_logs(investigation: InvestigationState, hostname: str) -> bool:
    """Return True if the hostname appears in any log entries already collected."""
    for entries in investigation.queried_sources.values():
        for entry in entries:
            if entry.hostname == hostname:
                return True
            # Also check details dict
            if entry.details.get("hostname") == hostname:
                return True
            if entry.details.get("computer") == hostname:
                return True
    return False
