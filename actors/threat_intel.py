"""ThreatIntelFeed actor — pushes unsolicited IOC updates."""
from __future__ import annotations

import uuid

from actors.registry import BaseActor
from models import ActorKind, ActorMessage, AgentRole


class ThreatIntelFeedActor(BaseActor):
    """
    Pushes a threat-intel advisory roughly every 12 steps. Payload sometimes
    includes an IOC that matches alerts already in the queue (ground_truth_relevant=True)
    and sometimes noise (ground_truth_relevant=False).
    """

    kind = "threat_intel"
    _IOC_POOL = [
        ("185.220.101.34", "TOR exit node with active C2 sightings"),
        ("evil-update.example", "newly-registered domain tied to FIN7 infrastructure"),
        ("9f2c3a...e1", "SHA-256 observed in Emotet loader campaign"),
        ("benign-corp.example", "registrar churn flagged by a noisy vendor feed"),
    ]

    def on_step(self, step: int, ctx: dict | None = None) -> ActorMessage | None:
        if step < 3 or step % 12 != 0:
            return None
        self._counter += 1
        ioc, note = self._rng.choice(self._IOC_POOL)
        relevant = self._rng.random() < 0.65
        return ActorMessage(
            message_id=f"TI-{uuid.UUID(int=self._rng.getrandbits(128)).hex[:8]}",
            actor=ActorKind.THREAT_INTEL,
            to_role=AgentRole.TIER1,
            subject=f"[ThreatIntel] New advisory — {ioc}",
            body=(
                f"ThreatIntelFeed advisory {self._counter}: indicator `{ioc}` "
                f"— {note}. Reconcile against current alert queue."
            ),
            step_created=step,
            requires_response=False,
            ground_truth_relevant=relevant,
            metadata={"ioc": ioc, "feed_confidence": round(self._rng.uniform(0.3, 0.95), 2)},
        )
