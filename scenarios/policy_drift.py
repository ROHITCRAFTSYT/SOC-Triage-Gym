"""
PolicyDriftEngine (Patronus sub-theme, Theme #3.2).

Injects mid-episode schema / policy drift. The engine tracks a sequence of
PolicyVersion snapshots; graders must honour whichever version was active
at the step an action was taken.

Drift triggers are deterministic per (seed, schedule).
"""
from __future__ import annotations

import random

from models import PolicyVersion


class PolicyDriftEngine:
    """
    Maintains an append-only list of PolicyVersion entries.

    Each entry records the step at which it became active. `active_at(step)`
    returns the most-recent policy whose step_activated <= step.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._versions: list[PolicyVersion] = [PolicyVersion()]
        self._schedule: list[int] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def plan(self, max_steps: int, drift_count: int = 2) -> None:
        """
        Pre-compute drift steps (roughly evenly spaced in middle of episode).
        Deterministic given the seed passed at construction.
        """
        self._schedule.clear()
        if drift_count <= 0 or max_steps < 10:
            return
        # Avoid drifting in the last 10% (no time to react).
        start = max(3, max_steps // 4)
        end = max(start + 1, int(max_steps * 0.85))
        step_candidates = list(range(start, end))
        self._rng.shuffle(step_candidates)
        self._schedule = sorted(step_candidates[:drift_count])

    # ------------------------------------------------------------------
    # Step driver
    # ------------------------------------------------------------------
    def maybe_drift(self, step: int) -> PolicyVersion | None:
        """If `step` is in the drift schedule, append a new PolicyVersion."""
        if step not in self._schedule:
            return None
        prev = self._versions[-1]
        kind = self._rng.choice(["field_rename", "severity_threshold", "admin_escalate", "tc_update"])
        new_version = PolicyVersion(
            version=prev.version + 1,
            step_activated=step,
            severity_threshold_high=prev.severity_threshold_high,
            field_rename_map=dict(prev.field_rename_map),
            admin_must_escalate=prev.admin_must_escalate,
            description="",
        )
        if kind == "field_rename":
            new_version.field_rename_map["src_ip"] = "source_address"
            new_version.description = "Schema rename: src_ip → source_address"
        elif kind == "severity_threshold":
            new_version.severity_threshold_high = 8.5
            new_version.description = "Severity tightening: HIGH now requires CVSS ≥ 8.5"
        elif kind == "admin_escalate":
            new_version.admin_must_escalate = True
            new_version.description = "Policy: all admin-account alerts MUST be escalated"
        else:
            new_version.description = "Terms & conditions updated — session tokens retention cut to 24h"
        self._versions.append(new_version)
        return new_version

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def active_at(self, step: int) -> PolicyVersion:
        """Return policy in effect at `step`."""
        active = self._versions[0]
        for v in self._versions:
            if v.step_activated <= step:
                active = v
            else:
                break
        return active

    def current(self) -> PolicyVersion:
        return self._versions[-1]

    def history(self) -> list[PolicyVersion]:
        return list(self._versions)

    # ------------------------------------------------------------------
    # Compliance check for grader
    # ------------------------------------------------------------------
    def policy_compliance(self, action_log: list[dict]) -> dict[str, float]:
        """
        Score an action log against policy constraints.

        Each action_log entry: {'step': int, 'kind': str, 'is_admin': bool,
                                'escalated': bool, ...}
        Returns {'compliance_rate': 0..1, 'violations': int, 'total': int}.
        """
        violations = 0
        total = 0
        for entry in action_log:
            step = entry.get("step", 0)
            pol = self.active_at(step)
            if pol.admin_must_escalate and entry.get("is_admin"):
                total += 1
                if not entry.get("escalated"):
                    violations += 1
        if total == 0:
            return {"compliance_rate": 1.0, "violations": 0, "total": 0}
        return {
            "compliance_rate": 1.0 - violations / total,
            "violations": violations,
            "total": total,
        }

    def to_dict(self) -> dict:
        return {
            "schedule": list(self._schedule),
            "versions": [v.model_dump() for v in self._versions],
        }
