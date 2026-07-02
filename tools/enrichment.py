"""
Threat Intelligence Enrichment Tool
=====================================
Pure function: looks up an indicator in the scenario's enrichment_db
and returns the result along with a step reward signal.

Reward logic:
  +0.10  if indicator is in ground_truth.relevant_indicators for the alert being investigated
  -0.03  if indicator was already enriched this episode (duplicate penalty)
  -0.03  if indicator has no entry and is unrelated to any ground truth
   0.00  for benign/neutral indicators not in the threat intel db
"""

from typing import Tuple
from models import EnrichmentResult, IndicatorType, ScenarioConfig, InvestigationState


def enrich_indicator(
    config: ScenarioConfig,
    investigation: InvestigationState,
    indicator: str,
    indicator_type: IndicatorType,
) -> Tuple[EnrichmentResult, float, str]:
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
