"""
Red-Team Grader
===============
Computes the red-team reward as a capped inverse of the blue-team score.

  blue_score     = TeamGrader().grade(...)
  novelty_bonus  = 0.1 if blue_score in [0.35, 0.65] else 0.0
  red_reward     = min(0.8, 1.0 - blue_score) + novelty_bonus

The novelty bonus rewards scenarios that sit in the "trainable sweet spot"
where the blue team neither trivially wins nor trivially loses.

A standalone static method `scenario_novelty_score` allows callers to
measure how novel a scenario fingerprint is relative to a history of
previously generated fingerprints.
"""

from typing import Dict

from graders.base import BaseGrader
from graders.team_grader import TeamGrader
from models import InvestigationState, ScenarioConfig


class RedTeamGrader(BaseGrader):
    """Grader for the red-team generator — rewards scenarios that defeat the blue team."""

    def grade(
        self,
        config: ScenarioConfig,
        investigations: Dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> float:
        """
        Compute red-team reward and return a score in (0.001, 0.999).

        Args:
            config: Full scenario config including ground_truth.
            investigations: Dict of alert_id → InvestigationState.
            steps_used: Number of steps taken by the blue team.
            max_steps: Maximum allowed steps.

        Returns:
            Score in (0.001, 0.999).
        """
        blue_score = TeamGrader().grade(config, investigations, steps_used, max_steps)
        novelty_bonus = 0.1 if 0.35 <= blue_score <= 0.65 else 0.0
        red_reward = min(0.8, 1.0 - blue_score) + novelty_bonus
        return self._clamp(red_reward)

    def grade_with_breakdown(
        self,
        config: ScenarioConfig,
        investigations: Dict[str, InvestigationState],
        steps_used: int,
        max_steps: int,
    ) -> tuple:
        """
        Grade and return (score, breakdown_dict, feedback_str).

        Returns:
            (score,
             {"blue_score": float, "red_base": float,
              "novelty_bonus": float, "total": float},
             feedback_str)
        """
        blue_score = TeamGrader().grade(config, investigations, steps_used, max_steps)
        novelty_bonus = 0.1 if 0.35 <= blue_score <= 0.65 else 0.0
        red_base = min(0.8, 1.0 - blue_score)
        red_reward = red_base + novelty_bonus
        score = self._clamp(red_reward)

        breakdown = {
            "blue_score": blue_score,
            "red_base": red_base,
            "novelty_bonus": novelty_bonus,
            "total": score,
        }

        sweet_spot = "yes" if novelty_bonus > 0.0 else "no"
        feedback_str = (
            f"Red-team reward: {score:.3f}. "
            f"Blue team scored {blue_score:.3f}. "
            f"Red base (capped inverse): {red_base:.3f}. "
            f"In trainable sweet spot [0.35, 0.65]: {sweet_spot} "
            f"(novelty_bonus={novelty_bonus:.2f})."
        )

        return score, breakdown, feedback_str

    @staticmethod
    def scenario_novelty_score(scenario_fingerprint: str, history: list) -> float:
        """
        Compute novelty of *scenario_fingerprint* relative to *history*.

        Novelty = 1.0 - max(0.0, overlap_fraction), where overlap_fraction
        is the highest fraction of tokens in *scenario_fingerprint* that
        appear in any single history entry.  If no history entry covers more
        than 70 % of the fingerprint tokens the fingerprint is considered
        novel (score → 1.0).

        Args:
            scenario_fingerprint: Space-separated token string describing the
                new scenario (e.g. attack pattern + IOC types + severity).
            history: List of previously seen fingerprint strings.

        Returns:
            Float in [0.0, 1.0]. Higher means more novel.
        """
        fp_tokens = set(scenario_fingerprint.lower().split())
        if not fp_tokens or not history:
            return 1.0

        max_overlap = 0.0
        for past in history:
            past_tokens = set(past.lower().split())
            if not past_tokens:
                continue
            overlap = len(fp_tokens & past_tokens) / len(fp_tokens)
            if overlap > max_overlap:
                max_overlap = overlap

        return 1.0 - max(0.0, max_overlap)
