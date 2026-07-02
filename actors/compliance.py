"""ComplianceOfficer actor — periodic policy / T&C questions for the Manager."""
from __future__ import annotations

import uuid

from actors.registry import BaseActor
from models import ActorKind, ActorMessage, AgentRole


class ComplianceOfficerActor(BaseActor):
    """
    Sends compliance-audit questions roughly every 20 steps, addressed to Manager.
    Ties into Patronus schema-drift: when policy_version changes, this actor's
    question references the new policy.
    """

    kind = "compliance"

    _QUESTIONS = [
        "Can you confirm that all high-severity alerts today were escalated within SLA?",
        "Our policy review flagged a T&C update on session tokens — has the team adopted v{version}?",
        "For audit trail: summarise the override rationale for any Tier-1 classification the Manager overturned.",
        "Please attest that privileged-account alerts received Tier-2 containment within this shift.",
    ]

    def on_step(self, step: int, ctx: dict | None = None) -> ActorMessage | None:
        if step < 10 or step % 20 != 0:
            return None
        self._counter += 1
        idx = self._rng.randrange(len(self._QUESTIONS))
        policy_version = (ctx or {}).get("policy_version", 1)
        body = self._QUESTIONS[idx].format(version=policy_version)
        return ActorMessage(
            message_id=f"CO-{uuid.UUID(int=self._rng.getrandbits(128)).hex[:8]}",
            actor=ActorKind.COMPLIANCE,
            to_role=AgentRole.MANAGER,
            subject=f"[Compliance] Audit question #{self._counter}",
            body=body,
            step_created=step,
            requires_response=True,
            ground_truth_relevant=True,
            metadata={"policy_version": policy_version},
        )
