"""EndUserReporter actor — noisy mailbox of user-submitted phishing reports."""
from __future__ import annotations

import uuid

from actors.registry import BaseActor
from models import ActorKind, ActorMessage, AgentRole


class EndUserReporterActor(BaseActor):
    """
    Employees send 'is this phishing?' reports. Some are real (ground_truth_relevant=True,
    agent earns reward if it classifies/escalates), most are benign (False).
    """

    kind = "end_user"

    _DEPARTMENTS = ["finance", "hr", "engineering", "sales", "legal", "support"]
    _SUBJECTS = [
        "Fwd: Urgent: password reset required",
        "Package delivery failed — please confirm address",
        "Fwd: Your DocuSign is ready",
        "Re: Q3 planning deck",
        "Fwd: I got this weird email — is it safe?",
        "Payroll notification — action required",
    ]

    def on_step(self, step: int, ctx: dict | None = None) -> ActorMessage | None:
        if step % 6 != 3:  # roughly one every 6 steps, offset so it doesn't collide
            return None
        self._counter += 1
        real = self._rng.random() < 0.30  # 30% of user reports are real phishing
        dept = self._rng.choice(self._DEPARTMENTS)
        subj = self._rng.choice(self._SUBJECTS)
        return ActorMessage(
            message_id=f"EU-{uuid.UUID(int=self._rng.getrandbits(128)).hex[:8]}",
            actor=ActorKind.END_USER,
            to_role=AgentRole.TIER1,
            subject=f"[User/{dept}] {subj}",
            body=(
                f"Employee in {dept} forwarded a suspicious message. "
                f"{'Attachment contains a known loader hash.' if real else 'Looks like routine marketing spam, user is cautious.'}"
            ),
            step_created=step,
            requires_response=real,
            ground_truth_relevant=real,
            metadata={"department": dept, "is_real_phish": real},
        )
