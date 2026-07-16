"""
Structured, resumable training runs
===================================

Every training run gets a self-describing directory instead of loose files
scattered at the repo root:

    runs/
      20260716-093000-tier1-qwen2.5-1.5b/
        config.json        — full hyperparameters + provenance (git commit,
                             env server version, dataset stats, seed manifest)
        metrics.jsonl      — append-only event stream (one JSON per line:
                             trainer logs, reward batches, eval results)
        checkpoints/       — trainer checkpoints (save_total_limit-rotated)
        best/              — best checkpoint by eval reward (+ best.json)
        eval/              — held-out evaluation reports
        MODEL_CARD.md      — auto-generated, publish-ready model card
        MANIFEST.json      — final summary written by finalize()

The JSONL metric stream replays cleanly into pandas / W&B / a notebook:
``[json.loads(l) for l in open("metrics.jsonl")]``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _git_provenance(repo_root: Path) -> dict:
    """Best-effort git commit/branch/dirty capture — never raises."""
    out: dict[str, Any] = {}
    for key, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
        ("dirty", ["status", "--porcelain"]),
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            value = result.stdout.strip()
            out[key] = bool(value) if key == "dirty" else value
        except (OSError, subprocess.SubprocessError):
            out[key] = None
    return out


def _slugify(text: str, max_len: int = 40) -> str:
    keep = [c.lower() if c.isalnum() or c in ".-" else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:max_len] or "run"


class TrainingRunManager:
    """Creates and owns one run directory; all writes are crash-safe appends."""

    def __init__(
        self,
        role: str,
        model_name: str,
        runs_dir: str | Path = "runs",
        run_id: str | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.role = role
        self.model_name = model_name
        self.runs_dir = Path(runs_dir)
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent

        if run_id is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{role}-{_slugify(model_name.split('/')[-1])}"
        self.run_id = run_id
        self.run_dir = self.runs_dir / run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.best_dir = self.run_dir / "best"
        self.eval_dir = self.run_dir / "eval"
        for d in (self.run_dir, self.checkpoints_dir, self.best_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._best_reward: float | None = None
        self.started_at = time.time()

    # -- config / provenance ---------------------------------------------------

    def write_config(self, config: dict) -> None:
        """Persist the full run config with git + environment provenance."""
        payload = {
            "run_id": self.run_id,
            "role": self.role,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "git": _git_provenance(self.repo_root),
            "server_url": os.environ.get("SERVER_URL", "http://localhost:7860"),
            **config,
        }
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # -- metric stream -----------------------------------------------------------

    def log(self, event: str, **fields) -> None:
        """Append one event to metrics.jsonl (crash-safe, replayable)."""
        record = {"ts": time.time(), "event": event, **fields}
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_metrics(self, event: str | None = None) -> list[dict]:
        """Load the metric stream back (optionally filtered by event type)."""
        if not self._metrics_path.exists():
            return []
        records = []
        with open(self._metrics_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event is None or rec.get("event") == event:
                    records.append(rec)
        return records

    # -- eval + best-checkpoint tracking -----------------------------------------

    def record_eval(self, report: dict, tag: str = "eval") -> Path:
        """Persist an evaluation report and stream it into metrics.jsonl."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.eval_dir / f"{tag}-{stamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.log("eval", tag=tag, **{k: v for k, v in report.items() if not isinstance(v, (list, dict))})
        return path

    def consider_best(self, mean_reward: float, checkpoint_dir: str | Path | None = None) -> bool:
        """Track the best eval reward; returns True when this one is a new best.

        When ``checkpoint_dir`` is given, a pointer (not a copy — checkpoints
        can be GBs) is recorded in best/best.json.
        """
        if self._best_reward is not None and mean_reward <= self._best_reward:
            return False
        self._best_reward = mean_reward
        payload = {
            "mean_reward": mean_reward,
            "checkpoint": str(checkpoint_dir) if checkpoint_dir else None,
            "recorded_at": time.time(),
        }
        (self.best_dir / "best.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log("new_best", mean_reward=mean_reward, checkpoint=str(checkpoint_dir))
        return True

    @property
    def best_reward(self) -> float | None:
        return self._best_reward

    # -- finalization ---------------------------------------------------------------

    def write_model_card(self, summary: dict) -> Path:
        """Auto-generate a publish-ready model card from the run's data."""
        cfg = {}
        cfg_path = self.run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        git = cfg.get("git", {})
        lines = [
            "---",
            "tags:",
            "  - reinforcement-learning",
            "  - grpo",
            "  - cybersecurity",
            "  - soc",
            "  - openenv",
            f"base_model: {self.model_name}",
            "---",
            "",
            f"# SOC-Triage-Gym GRPO — {self.role.upper()} analyst",
            "",
            f"GRPO-trained `{self.role}` policy for the "
            "[SOC-Triage-Gym](https://github.com/ROHITCRAFTSYT/SOC-Triage-Gym) "
            "multi-agent SOC environment.",
            "",
            "## Training provenance",
            "",
            f"- **Run ID:** `{self.run_id}`",
            f"- **Base model:** `{self.model_name}`",
            f"- **Git commit:** `{git.get('commit', 'unknown')}`" + (" (dirty tree)" if git.get("dirty") else ""),
            f"- **Tasks:** {summary.get('tasks', 'n/a')}",
            f"- **Dataset:** {summary.get('dataset_size', 'n/a')} per-step prompts, "
            f"seeds {summary.get('seed_range', 'n/a')}",
            "",
            "## Results",
            "",
            f"- **Final mean training reward:** {summary.get('final_mean_reward', 'n/a')}",
            f"- **Best held-out eval reward:** {summary.get('best_eval_reward', 'n/a')}",
            f"- **Oracle ceiling:** {summary.get('oracle_avg', 'n/a')}",
            "",
            "## Usage",
            "",
            "The model emits a JSON action object. Parse it with",
            "`training.rewards.parse_action_from_text()` and step the env via",
            "`POST /step`.",
            "",
            "Reproduce with:",
            "",
            "```bash",
            f"python train_grpo.py --role {self.role} --model {self.model_name} --curriculum",
            "```",
        ]
        path = self.run_dir / "MODEL_CARD.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def finalize(self, summary: dict) -> Path:
        """Write MANIFEST.json + MODEL_CARD.md; returns the manifest path."""
        manifest = {
            "run_id": self.run_id,
            "role": self.role,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "duration_seconds": round(time.time() - self.started_at, 1),
            "best_eval_reward": self._best_reward,
            **summary,
        }
        path = self.run_dir / "MANIFEST.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        self.write_model_card({**summary, "best_eval_reward": self._best_reward})
        self.log("finalized", **{k: v for k, v in manifest.items() if not isinstance(v, (list, dict))})
        return path

    # -- discovery ---------------------------------------------------------------------

    @classmethod
    def list_runs(cls, runs_dir: str | Path = "runs") -> list[dict]:
        """Summaries of every run under runs_dir (newest first)."""
        runs_dir = Path(runs_dir)
        if not runs_dir.exists():
            return []
        out = []
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            manifest = d / "MANIFEST.json"
            config = d / "config.json"
            info: dict[str, Any] = {"run_id": d.name, "path": str(d), "finalized": manifest.exists()}
            source = manifest if manifest.exists() else config
            if source.exists():
                try:
                    data = json.loads(source.read_text(encoding="utf-8"))
                    info.update({k: data[k] for k in ("role", "model_name", "best_eval_reward") if k in data})
                except json.JSONDecodeError:
                    pass
            out.append(info)
        return out
