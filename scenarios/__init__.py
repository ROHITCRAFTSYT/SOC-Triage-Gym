"""SOC-Triage-Gym scenario generators."""
from scenarios.apt_campaign import APTCampaignScenario
from scenarios.insider_threat import InsiderThreatScenario
from scenarios.lateral_movement import LateralMovementScenario
from scenarios.phishing import PhishingScenario
from scenarios.queue_management import QueueManagementScenario
from scenarios.team_lateral_team import TeamLateralTeamScenario
from scenarios.team_phishing_escalation import TeamPhishingEscalationScenario

SCENARIO_REGISTRY = {
    "phishing": PhishingScenario,
    "lateral_movement": LateralMovementScenario,
    "queue_management": QueueManagementScenario,
    "insider_threat": InsiderThreatScenario,
    "team_phishing_escalation": TeamPhishingEscalationScenario,
    "team_lateral_team": TeamLateralTeamScenario,
    "apt_campaign": APTCampaignScenario,
}

__all__ = [
    "PhishingScenario",
    "LateralMovementScenario",
    "QueueManagementScenario",
    "InsiderThreatScenario",
    "TeamPhishingEscalationScenario",
    "TeamLateralTeamScenario",
    "APTCampaignScenario",
    "SCENARIO_REGISTRY",
]
