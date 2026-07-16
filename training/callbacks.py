"""
TRL / transformers trainer integration
======================================

``SOCGymCallback`` streams every trainer log record into the run manager's
``metrics.jsonl`` and applies reward-plateau early stopping — so a 4-hour
GPU run that converged after 90 minutes stops itself instead of burning the
budget.

transformers is an optional dependency here: the module imports cleanly
without it (the callback class just becomes unusable), so the rest of the
training toolkit stays importable on CPU-only dev machines.
"""

from __future__ import annotations

try:
    from transformers import TrainerCallback

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised only without transformers
    TrainerCallback = object
    _HAS_TRANSFORMERS = False

from training.run_manager import TrainingRunManager


class SOCGymCallback(TrainerCallback):
    """Logs trainer metrics to the run manager and early-stops on plateau.

    Args:
        run_manager: destination for the metric stream.
        early_stop_patience: stop when the rolling mean train reward hasn't
            improved for this many log events (0 disables).
        min_delta: improvement smaller than this counts as a plateau.
    """

    def __init__(
        self,
        run_manager: TrainingRunManager,
        early_stop_patience: int = 0,
        min_delta: float = 0.005,
    ) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is required for SOCGymCallback")
        self.run_manager = run_manager
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self._best = float("-inf")
        self._stale = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        self.run_manager.log("trainer_log", step=state.global_step, **logs)

        reward = logs.get("reward") or logs.get("rewards/mean")
        if self.early_stop_patience > 0 and reward is not None:
            if reward > self._best + self.min_delta:
                self._best = reward
                self._stale = 0
            else:
                self._stale += 1
                if self._stale >= self.early_stop_patience:
                    self.run_manager.log(
                        "early_stop",
                        step=state.global_step,
                        best_reward=self._best,
                        stale_logs=self._stale,
                    )
                    control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.run_manager.log("checkpoint_saved", step=state.global_step)
        return control
