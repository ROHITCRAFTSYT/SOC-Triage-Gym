"""
Staged task curriculum with promotion gates
===========================================

Small models fail on hard tasks not because the gradient is wrong but
because the reward is too sparse to find. The curriculum starts training on
the easiest team task and only *promotes* to the next stage once the
policy's rolling mean reward clears the stage's threshold — so the model
always trains at the edge of its competence instead of drowning in noise.

The final stage mixes in ``red_team_generated`` scenarios, closing the RLVE
loop: the adaptive red-team generator keeps producing scenarios in the
trainable sweet spot as the blue policy improves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    """One rung of the curriculum ladder."""

    name: str
    tasks: list[str]
    promotion_threshold: float  # rolling mean reward needed to advance
    min_samples: int = 32  # rewards observed before promotion is considered
    max_rounds: int = 3  # training rounds before force-advancing


@dataclass
class CurriculumScheduler:
    """Tracks rewards per stage and decides when to promote.

    Usage in the training loop::

        scheduler = default_team_curriculum()
        while not scheduler.finished:
            stage = scheduler.current
            ...train one round on stage.tasks, calling scheduler.record(r)
               for every reward observed...
            scheduler.end_round()   # promotes or repeats per the gate
    """

    stages: list[CurriculumStage]
    window: int = 256  # rolling window of recent rewards per stage

    _stage_idx: int = 0
    _round_in_stage: int = 0
    _rewards: deque = field(default_factory=deque)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Curriculum needs at least one stage.")
        self._rewards = deque(maxlen=self.window)

    # -- state ------------------------------------------------------------------

    @property
    def current(self) -> CurriculumStage:
        return self.stages[min(self._stage_idx, len(self.stages) - 1)]

    @property
    def stage_index(self) -> int:
        return self._stage_idx

    @property
    def finished(self) -> bool:
        return self._stage_idx >= len(self.stages)

    @property
    def rolling_mean(self) -> float | None:
        if not self._rewards:
            return None
        return sum(self._rewards) / len(self._rewards)

    # -- updates -----------------------------------------------------------------

    def record(self, reward: float) -> None:
        """Feed one observed reward (call from the reward function)."""
        self._rewards.append(float(reward))

    def record_batch(self, rewards: list[float]) -> None:
        for r in rewards:
            self.record(r)

    def should_promote(self) -> bool:
        stage = self.current
        if len(self._rewards) < stage.min_samples:
            return False
        mean = self.rolling_mean
        return mean is not None and mean >= stage.promotion_threshold

    def end_round(self) -> dict:
        """Close one training round: promote, repeat, or force-advance.

        Returns a decision record (also appended to ``history``).
        """
        stage = self.current
        self._round_in_stage += 1
        mean = self.rolling_mean
        promoted = self.should_promote()
        forced = not promoted and self._round_in_stage >= stage.max_rounds

        decision = {
            "stage": stage.name,
            "stage_index": self._stage_idx,
            "round_in_stage": self._round_in_stage,
            "rolling_mean": mean,
            "threshold": stage.promotion_threshold,
            "samples": len(self._rewards),
            "promoted": promoted,
            "forced_advance": forced,
        }
        self.history.append(decision)

        if promoted or forced:
            self._stage_idx += 1
            self._round_in_stage = 0
            self._rewards.clear()
        return decision


def default_team_curriculum() -> CurriculumScheduler:
    """The standard 3-stage SOC team curriculum.

    Stage thresholds are rolling *step*-reward means (env step reward +
    parse shaping), not episode scores — they gate on "is the policy taking
    productive actions yet", which is observable long before episodes score
    well.
    """
    return CurriculumScheduler(
        stages=[
            CurriculumStage(
                name="foundations",
                tasks=["team_phishing_escalation"],
                promotion_threshold=0.05,
                min_samples=32,
                max_rounds=3,
            ),
            CurriculumStage(
                name="noisy-queues",
                tasks=["team_phishing_escalation", "team_lateral_team"],
                promotion_threshold=0.08,
                min_samples=48,
                max_rounds=3,
            ),
            CurriculumStage(
                name="adversarial",
                tasks=["team_lateral_team", "red_team_generated"],
                promotion_threshold=0.10,
                min_samples=48,
                max_rounds=2,
            ),
        ]
    )
