"""
Enterprise TicketingSystem (Scaler AI Labs sub-theme, Theme #3.1).

Treats the SOC workflow as a multi-app enterprise pipeline:

    SIEM      — alert ingestion, log queries, correlation
    EDR       — endpoint forensics, sandbox, memory analysis, isolation
    IAM       — user lookup, disable, privilege review
    TICKETING — SLA clocks, priority, assignment, cross-app audit

Cross-app business rule enforced here:
    You cannot `disable_user` in IAM unless an OPEN ticket of priority P2
    (or higher) exists referencing the target user's alert.

This is a thin, in-memory ticket store. Deterministic by construction
because IDs include a monotonic counter rather than uuid4.
"""
from __future__ import annotations

from models import AgentRole, TicketSLA


class TicketingSystem:
    """Lightweight ticketing app with SLA clocks and cross-app rules."""

    PRIORITY_BUDGET = {"P1": 10, "P2": 20, "P3": 40, "P4": 80}

    def __init__(self) -> None:
        self._tickets: dict[str, TicketSLA] = {}
        self._by_alert: dict[str, list[str]] = {}
        self._counter = 0

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def open(
        self,
        alert_id: str,
        priority: str = "P3",
        assignee: AgentRole = AgentRole.TIER2,
        note: str = "",
    ) -> TicketSLA:
        self._counter += 1
        tid = f"TKT-{self._counter:05d}"
        sla = self.PRIORITY_BUDGET.get(priority, 40)
        ticket = TicketSLA(
            ticket_id=tid,
            alert_id=alert_id,
            priority=priority,
            assignee_role=assignee,
            status="open",
            sla_steps_remaining=sla,
            app_chain=["SIEM", "TICKETING"],
            notes=[note] if note else [],
        )
        self._tickets[tid] = ticket
        self._by_alert.setdefault(alert_id, []).append(tid)
        return ticket

    def touch(self, ticket_id: str, app: str, note: str = "") -> TicketSLA | None:
        t = self._tickets.get(ticket_id)
        if t is None:
            return None
        if app not in t.app_chain:
            t.app_chain.append(app)
        if note:
            t.notes.append(note)
        return t

    def resolve(self, ticket_id: str, note: str = "") -> TicketSLA | None:
        t = self._tickets.get(ticket_id)
        if t is None:
            return None
        t.status = "resolved"
        if note:
            t.notes.append(note)
        return t

    def tick(self) -> None:
        """Decrement SLA clocks on all open tickets."""
        for t in self._tickets.values():
            if t.status in ("open", "in_progress") and t.sla_steps_remaining > 0:
                t.sla_steps_remaining -= 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, ticket_id: str) -> TicketSLA | None:
        return self._tickets.get(ticket_id)

    def by_alert(self, alert_id: str) -> list[TicketSLA]:
        return [self._tickets[i] for i in self._by_alert.get(alert_id, []) if i in self._tickets]

    def open_count(self) -> int:
        return sum(1 for t in self._tickets.values() if t.status in ("open", "in_progress"))

    def sla_breaches(self) -> list[TicketSLA]:
        return [t for t in self._tickets.values() if t.sla_steps_remaining <= 0 and t.status != "resolved"]

    def all_tickets(self) -> list[TicketSLA]:
        return list(self._tickets.values())

    # ------------------------------------------------------------------
    # Cross-app business rules
    # ------------------------------------------------------------------
    def can_disable_user(self, alert_id: str) -> bool:
        """
        IAM.disable_user is only permitted if an OPEN P2+ ticket exists for
        the corresponding alert_id. Prevents the agent from taking a drastic
        identity action without an audit trail.
        """
        for t in self.by_alert(alert_id):
            if t.status in ("open", "in_progress") and t.priority in ("P1", "P2"):
                return True
        return False

    def audit_summary(self) -> dict:
        return {
            "total_tickets": len(self._tickets),
            "open": self.open_count(),
            "resolved": sum(1 for t in self._tickets.values() if t.status == "resolved"),
            "sla_breaches": len(self.sla_breaches()),
            "apps_used": sorted({a for t in self._tickets.values() for a in t.app_chain}),
        }
