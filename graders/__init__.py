"""SOC-Triage-Gym graders — deterministic 0.0 to 1.0 scoring."""
from graders.apt_campaign_grader import APTCampaignGrader
from graders.insider_threat_grader import InsiderThreatGrader
from graders.lateral_movement_grader import LateralMovementGrader
from graders.phishing_grader import PhishingGrader
from graders.queue_management_grader import QueueManagementGrader
from graders.team_grader import TeamGrader

GRADER_REGISTRY = {
    "phishing": PhishingGrader,
    "lateral_movement": LateralMovementGrader,
    "queue_management": QueueManagementGrader,
    "insider_threat": InsiderThreatGrader,
    "team_phishing_escalation": TeamGrader,
    "team_lateral_team": TeamGrader,
    "red_team_generated": TeamGrader,
    "apt_campaign": APTCampaignGrader,
}

__all__ = [
    "PhishingGrader",
    "LateralMovementGrader",
    "QueueManagementGrader",
    "InsiderThreatGrader",
    "TeamGrader",
    "APTCampaignGrader",
    "GRADER_REGISTRY",
]
