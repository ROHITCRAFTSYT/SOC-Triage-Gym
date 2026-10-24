"""
Threat Intelligence Enrichment Tool
=====================================
Pure function: looks up an indicator in the scenario's enrichment_db
and returns the result along with a step reward signal.

Reward logic (must stay in sync with enrich_indicator below):
  +0.12  relevant indicator that is malicious in the threat intel db
  +0.08  relevant indicator that is benign — still worth checking
  -0.03  duplicate: indicator was already enriched this episode
  -0.03  irrelevant indicator that exists in the db (unproductive lookup)
  -0.02  no db entry AND not a relevant indicator (chasing a dead end)
   0.00  no db entry but the indicator is relevant (missing intel, not the agent's fault)
"""


from models import EnrichmentResult, IndicatorType, InvestigationState, ScenarioConfig


def enrich_indicator(
    config: ScenarioConfig,
    investigation: InvestigationState,
    indicator: str,
    indicator_type: IndicatorType,
) -> tuple[EnrichmentResult, float, str]:
    """
    Look up a threat indicator in the scenario's enrichment database.

    Args:
        config: The current scenario configuration (contains enrichment_db and ground_truth).
        investigation: The active alert's investigation state (used to detect duplicates).
        indicator: The indicator value to enrich (IP, domain, hash, etc.).
        indicator_type: The type of indicator.

    Returns:
        (EnrichmentResult, step_reward, message)
    """
    # Check for duplicate enrichment
    if indicator in investigation.enriched_indicators:
        msg = f"Indicator '{indicator}' already enriched — no new information."
        return investigation.enriched_indicators[indicator], -0.03, msg

    # Look up in scenario's threat intel database
    result = config.enrichment_db.get(indicator)

    if result is None:
        # Indicator not in DB — synthesize a benign/unknown result
        result = EnrichmentResult(
            indicator=indicator,
            indicator_type=indicator_type,
            malicious=False,
            confidence=0.1,
            threat_score=0,
            threat_type=None,
            tags=["unknown"],
            source="threat_intel",
        )
        # Check if this was relevant (should have been in DB if it was)
        relevant = _is_relevant_indicator(config, investigation.alert_id, indicator)
        reward = -0.02 if not relevant else 0.0
        msg = f"No threat intel found for '{indicator}'. Indicator appears clean."
        return result, reward, msg

    # Determine reward based on relevance to ground truth
    relevant = _is_relevant_indicator(config, investigation.alert_id, indicator)

    if relevant:
        if result.malicious:
            reward = 0.12  # relevant AND malicious — high signal
            msg = f"MALICIOUS: '{indicator}' is a known threat indicator. Threat score: {result.threat_score}/100."
        else:
            reward = 0.08  # relevant but benign — still good to check
            msg = f"CLEAN: '{indicator}' has no known malicious activity. Threat score: {result.threat_score}/100."
    else:
        # Irrelevant indicator
        reward = -0.03
        msg = f"Enriched '{indicator}' — not directly relevant to this alert."

    return result, reward, msg


def _is_relevant_indicator(config: ScenarioConfig, alert_id: str, indicator: str) -> bool:
    """Return True if this indicator appears in the ground truth's relevant_indicators for the alert."""
    relevant_list = config.ground_truth.relevant_indicators.get(alert_id, [])
    return indicator in relevant_list
