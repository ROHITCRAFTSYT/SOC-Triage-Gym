"""
ExpertPanel — simulated experts-in-the-loop with drifting preferences
(Snorkel sub-theme, Theme #4).

A rotating panel of three expert profiles:

    dr_accuracy     — weights correctness heavily, dislikes speed bonuses
    speedy_sam      — rewards compact, fast resolution
    thorough_thea   — rewards tool-coverage breadth and detailed evidence

Which expert is judging depends on the curriculum round (derived from the
episode_count of the Red-Team curriculum). The agent receives a hint in
their inbox ("Dr. Accuracy is reviewing today") — they must adapt.

This implements Snorkel's brief of "changing requirements / preferences":
the reward function itself rotates without the agent's input schema changing.
"""
from __future__ import annotations

from models import ExpertProfile


def build_standard_panel() -> list[ExpertProfile]:
    return [
        ExpertProfile(
            expert_id="dr_accuracy",
            display_name="Dr. Accuracy",
            weight_accuracy=0.60,
            weight_reasoning=0.25,
            weight_actionability=0.10,
            weight_speed=0.00,
            weight_thoroughness=0.05,
            personality_note="Prefers correct, defensible classifications. Penalizes guesses.",
        ),
        ExpertProfile(
            expert_id="speedy_sam",
            display_name="Speedy Sam",
            weight_accuracy=0.30,
            weight_reasoning=0.15,
            weight_actionability=0.25,
            weight_speed=0.25,
            weight_thoroughness=0.05,
            personality_note="Rewards fast, decisive resolution. Hates long investigations.",
        ),
        ExpertProfile(
            expert_id="thorough_thea",
            display_name="Thorough Thea",
            weight_accuracy=0.30,
            weight_reasoning=0.20,
            weight_actionability=0.15,
            weight_speed=0.00,
            weight_thoroughness=0.35,
            personality_note="Rewards breadth: wide tool coverage, many correlations, rich evidence.",
        ),
    ]


class ExpertPanel:
    """Panel that rotates across curriculum rounds."""

    def __init__(self, panel: list[ExpertProfile] | None = None) -> None:
        self._panel: list[ExpertProfile] = panel or build_standard_panel()

    def for_round(self, round_index: int) -> ExpertProfile:
        if not self._panel:
            raise ValueError("ExpertPanel is empty")
        return self._panel[round_index % len(self._panel)]

    def all_profiles(self) -> list[ExpertProfile]:
        return list(self._panel)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, signals: dict[str, float], expert: ExpertProfile) -> dict[str, float]:
        """
        Blend raw agent signals with the active expert's preference weights.

        Expected keys in `signals` (all 0..1):
            accuracy       — grader accuracy on classifications
            reasoning      — ManagerJudge reasoning quality
            actionability  — ManagerJudge actionability
            speed          — 1 - steps_used/max_steps  (higher = faster)
            thoroughness   — tool_coverage_fraction
        """
        s = {k: max(0.0, min(1.0, float(signals.get(k, 0.0)))) for k in (
            "accuracy", "reasoning", "actionability", "speed", "thoroughness"
        )}
        total = (
            s["accuracy"]       * expert.weight_accuracy
            + s["reasoning"]    * expert.weight_reasoning
            + s["actionability"] * expert.weight_actionability
            + s["speed"]        * expert.weight_speed
            + s["thoroughness"] * expert.weight_thoroughness
        )
        # Normalize by weight sum so result stays in [0,1] regardless of profile.
        weight_sum = (
            expert.weight_accuracy + expert.weight_reasoning + expert.weight_actionability
            + expert.weight_speed + expert.weight_thoroughness
        ) or 1.0
        return {
            "expert_id": expert.expert_id,
            "total": total / weight_sum,
            "signals": s,
            "weights": {
                "accuracy": expert.weight_accuracy,
                "reasoning": expert.weight_reasoning,
                "actionability": expert.weight_actionability,
                "speed": expert.weight_speed,
                "thoroughness": expert.weight_thoroughness,
            },
        }

    def hint_message(self, expert: ExpertProfile) -> str:
        """One-line note surfaced to the agent at reset()."""
        return f"This shift's reviewer: {expert.display_name}. Note: {expert.personality_note}"
