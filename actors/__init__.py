"""
External NPC actors (Halluminate sub-theme).

Simulated non-learning actors that inject unsolicited messages into the
agent's inbox during an episode, forcing the multi-role SOC team to manage
multiple actors beyond its own three learnable roles.

Actors are deterministic — same (scenario_id, seed) → same message stream.
"""
from actors.compliance import ComplianceOfficerActor
from actors.end_user import EndUserReporterActor
from actors.registry import ActorRegistry, build_default_registry
from actors.threat_intel import ThreatIntelFeedActor

__all__ = [
    "ThreatIntelFeedActor",
    "ComplianceOfficerActor",
    "EndUserReporterActor",
    "ActorRegistry",
    "build_default_registry",
]
