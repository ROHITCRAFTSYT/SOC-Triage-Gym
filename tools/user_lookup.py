"""
User Directory Lookup Tool
============================
Pure function: looks up a user in the scenario's user database.

Reward logic:
  +0.05  if username appears in any log entry or alert indicator already seen
   0.02  for a valid user lookup (user exists in DB)
   0.00  for duplicate lookups
  -0.03  if user doesn't exist in DB and wasn't seen in any evidence
"""


from models import InvestigationState, ScenarioConfig, UserInfo


def lookup_user(
    config: ScenarioConfig,
    investigation: InvestigationState,
    username: str,
) -> tuple[UserInfo | None, float, str]:
    """
    Look up a user in the directory service.

    Args:
        config: The current scenario configuration (contains user_db).
        investigation: The active investigation state (used to score relevance).
        username: The username to look up (samaccountname).

    Returns:
        (UserInfo or None, step_reward, message)
    """
    # Check for duplicate lookup
    if username in investigation.users_looked_up:
        return investigation.users_looked_up[username], 0.0, f"Already looked up user '{username}'."

    user = config.user_db.get(username)

    if user is None:
        # Check if user appeared in evidence
        seen_in_evidence = _username_in_evidence(investigation, username)
        if seen_in_evidence:
            reward = 0.0
            msg = f"User '{username}' seen in logs but not found in directory. May be external or deleted."
        else:
            reward = -0.03
            msg = f"User '{username}' not found in user directory."
        return None, reward, msg

    # Score based on whether this user appeared in collected evidence
    seen_in_evidence = _username_in_evidence(investigation, username)
    reward = 0.05 if seen_in_evidence else 0.02

    risk_flag = " [HIGH RISK USER]" if user.risk_score > 0.7 else ""
    priv_flag = " [PRIVILEGED]" if user.is_privileged else ""
    msg = (
        f"User found: {username} ({user.role}, {user.department}){priv_flag}{risk_flag}. "
        f"Risk score: {user.risk_score:.2f}, Last login: {user.last_login or 'unknown'}"
    )

    return user, reward, msg


def _username_in_evidence(investigation: InvestigationState, username: str) -> bool:
    """Return True if the username appears in any collected log entries or alert indicators."""
    for entries in investigation.queried_sources.values():
        for entry in entries:
            if entry.user == username:
                return True
            if entry.details.get("user") == username:
                return True
            if entry.details.get("account") == username:
                return True
    return False
