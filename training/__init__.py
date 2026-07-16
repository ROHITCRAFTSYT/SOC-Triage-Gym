"""
SOC-Triage-Gym training toolkit
===============================

Production-grade building blocks for the GRPO training pipeline:

  * ``training.rewards``     — action parsing, reward shaping, and the
    session-parallel reward evaluator (scores GRPO completion groups
    concurrently against the multi-session server).
  * ``training.run_manager`` — structured, resumable training runs:
    ``runs/<run_id>/`` with config + git provenance, JSONL metrics,
    best-checkpoint tracking, and an auto-generated model card.
  * ``training.curriculum``  — staged task curriculum with promotion gates
    (easy → hard as the policy's rolling reward clears each threshold).
  * ``training.evaluation``  — held-out policy evaluation on disjoint seeds.
  * ``training.callbacks``   — optional TRL/transformers TrainerCallback that
    streams trainer logs into the run manager and applies early stopping.

Everything here is importable without torch/trl/transformers installed;
only ``training.callbacks`` needs transformers, and it degrades gracefully.
"""

from training.curriculum import CurriculumScheduler, CurriculumStage, default_team_curriculum
from training.evaluation import evaluate_policy
from training.rewards import (
    PARSE_BONUS,
    ParallelRewardEvaluator,
    classify_parse_quality,
    parse_action_from_text,
    score_completion,
)
from training.run_manager import TrainingRunManager

__all__ = [
    "PARSE_BONUS",
    "CurriculumScheduler",
    "CurriculumStage",
    "ParallelRewardEvaluator",
    "TrainingRunManager",
    "classify_parse_quality",
    "default_team_curriculum",
    "evaluate_policy",
    "parse_action_from_text",
    "score_completion",
]
