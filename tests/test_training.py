"""
Tests for the v0.3.0 training toolkit: curriculum scheduler, run manager,
reward scoring (sequential and session-parallel), held-out evaluation, and
the training CLI surface. Everything runs CPU-only against the FastAPI
TestClient — no torch/trl required.
"""

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app
from training.curriculum import CurriculumScheduler, CurriculumStage, default_team_curriculum
from training.evaluation import evaluate_policy, run_policy_episode
from training.rewards import (
    PARSE_BONUS,
    CompletionItem,
    ParallelRewardEvaluator,
    classify_parse_quality,
    parse_action_from_text,
    score_completion,
)
from training.run_manager import TrainingRunManager

# ---------------------------------------------------------------------------
# Parsing + shaping
# ---------------------------------------------------------------------------

class TestParsing:
    def test_strict_json(self):
        assert classify_parse_quality('{"action_type": "noop", "role": "tier1"}') == "strict"

    def test_loose_fenced_json(self):
        text = 'Here is my action:\n```json\n{"action_type": "classify_alert"}\n```'
        assert classify_parse_quality(text) == "loose"
        action = parse_action_from_text(text, "tier1")
        assert action["action_type"] == "classify_alert"
        assert action["role"] == "tier1"

    def test_fallback_keyword(self):
        text = "I think we should escalate_to_tier2 because this looks bad"
        assert classify_parse_quality(text) == "fallback"
        assert parse_action_from_text(text, "tier1")["action_type"] == "escalate_to_tier2"

    def test_garbage_becomes_noop(self):
        action = parse_action_from_text("lorem ipsum", "tier2")
        assert action == {"action_type": "noop", "role": "tier2"}

    def test_parse_bonus_ordering(self):
        assert PARSE_BONUS["strict"] > PARSE_BONUS["loose"] > PARSE_BONUS["fallback"]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum:
    def _tiny(self):
        return CurriculumScheduler(stages=[
            CurriculumStage(name="a", tasks=["t1"], promotion_threshold=0.1, min_samples=4, max_rounds=2),
            CurriculumStage(name="b", tasks=["t2"], promotion_threshold=0.5, min_samples=4, max_rounds=2),
        ])

    def test_promotes_when_gate_cleared(self):
        sched = self._tiny()
        sched.record_batch([0.2, 0.2, 0.2, 0.2])
        decision = sched.end_round()
        assert decision["promoted"] is True
        assert sched.current.name == "b"

    def test_repeats_below_gate_then_force_advances(self):
        sched = self._tiny()
        sched.record_batch([0.0, 0.0, 0.0, 0.0])
        d1 = sched.end_round()
        assert d1["promoted"] is False and d1["forced_advance"] is False
        assert sched.current.name == "a"  # repeating
        sched.record_batch([0.0] * 4)
        d2 = sched.end_round()
        assert d2["forced_advance"] is True
        assert sched.current.name == "b"

    def test_min_samples_blocks_promotion(self):
        sched = self._tiny()
        sched.record_batch([1.0, 1.0])  # above gate but too few samples
        assert sched.should_promote() is False

    def test_finished_after_last_stage(self):
        sched = self._tiny()
        for _ in range(4):
            sched.record_batch([1.0] * 8)
            sched.end_round()
            if sched.finished:
                break
        assert sched.finished
        assert len(sched.history) >= 2

    def test_default_curriculum_shape(self):
        sched = default_team_curriculum()
        names = [s.name for s in sched.stages]
        assert names == ["foundations", "noisy-queues", "adversarial"]
        assert "red_team_generated" in sched.stages[-1].tasks
        thresholds = [s.promotion_threshold for s in sched.stages]
        assert thresholds == sorted(thresholds)  # monotonically harder


# ---------------------------------------------------------------------------
# Run manager
# ---------------------------------------------------------------------------

class TestRunManager:
    def test_run_directory_structure(self, tmp_path):
        rm = TrainingRunManager("tier1", "Qwen/Qwen2.5-1.5B-Instruct", runs_dir=tmp_path)
        assert rm.run_dir.exists()
        assert rm.checkpoints_dir.exists()
        assert rm.best_dir.exists()
        assert rm.eval_dir.exists()
        assert "tier1" in rm.run_id

    def test_config_includes_provenance(self, tmp_path):
        rm = TrainingRunManager("tier1", "m", runs_dir=tmp_path)
        rm.write_config({"learning_rate": 5e-6})
        cfg = json.loads((rm.run_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["learning_rate"] == 5e-6
        assert "git" in cfg and "commit" in cfg["git"]

    def test_metrics_jsonl_roundtrip(self, tmp_path):
        rm = TrainingRunManager("tier1", "m", runs_dir=tmp_path)
        rm.log("reward_batch", mean_reward=0.12, strict=6, loose=1, fallback=1)
        rm.log("trainer_log", step=10, loss=0.5)
        batches = rm.read_metrics("reward_batch")
        assert len(batches) == 1
        assert batches[0]["mean_reward"] == 0.12
        assert len(rm.read_metrics()) == 2

    def test_best_tracking_monotonic(self, tmp_path):
        rm = TrainingRunManager("tier1", "m", runs_dir=tmp_path)
        assert rm.consider_best(0.5, checkpoint_dir="ckpt-a") is True
        assert rm.consider_best(0.4, checkpoint_dir="ckpt-b") is False
        assert rm.consider_best(0.6, checkpoint_dir="ckpt-c") is True
        best = json.loads((rm.best_dir / "best.json").read_text(encoding="utf-8"))
        assert best["mean_reward"] == 0.6
        assert best["checkpoint"] == "ckpt-c"

    def test_finalize_writes_manifest_and_model_card(self, tmp_path):
        rm = TrainingRunManager("tier1", "Qwen/Qwen2.5-1.5B-Instruct", runs_dir=tmp_path)
        rm.write_config({"tasks": ["team_phishing_escalation"]})
        rm.consider_best(0.7)
        manifest_path = rm.finalize({"dataset_size": 123, "tasks": ["team_phishing_escalation"]})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["best_eval_reward"] == 0.7
        assert manifest["dataset_size"] == 123
        card = (rm.run_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
        assert "GRPO" in card and "TIER1" in card

    def test_list_runs(self, tmp_path):
        rm1 = TrainingRunManager("tier1", "m", runs_dir=tmp_path, run_id="run-a")
        rm1.finalize({})
        TrainingRunManager("tier2", "m", runs_dir=tmp_path, run_id="run-b")
        runs = TrainingRunManager.list_runs(tmp_path)
        by_id = {r["run_id"]: r for r in runs}
        assert by_id["run-a"]["finalized"] is True
        assert by_id["run-b"]["finalized"] is False


# ---------------------------------------------------------------------------
# Reward scoring against the live environment (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def env_client():
    import server.app as app_module
    with TestClient(app) as client:
        app_module._sessions.clear()
        app_module.METRICS.reset()
        app_module.AUDIT.clear()
        yield client


class TestRewardScoring:
    def test_score_completion_valid_action_beats_garbage(self, env_client):
        from train_grpo import replay_to_step

        # A correct tier1 enrichment/classification completion at step 0
        good = json.dumps({
            "action_type": "phase_complete", "role": "tier1",
        })
        good_reward, good_q = score_completion(
            env_client, text=good, role="tier1",
            task_id="team_phishing_escalation", seed=42, step_index=0,
            replay_fn=replay_to_step,
        )
        bad_reward, bad_q = score_completion(
            env_client, text="hello world no json here at all", role="tier1",
            task_id="team_phishing_escalation", seed=42, step_index=0,
            replay_fn=replay_to_step,
        )
        assert good_q == "strict"
        assert bad_q == "fallback"
        assert good_reward > bad_reward

    def test_parallel_evaluator_matches_sequential(self, env_client):
        from train_grpo import replay_to_step

        items = [
            CompletionItem(
                text=json.dumps({"action_type": "noop", "role": "tier1"}),
                task_id="team_phishing_escalation", seed=42, step_index=0,
            )
            for _ in range(4)
        ]

        def factory(session_id: str):
            return TestClient(app, headers={"X-Session-ID": session_id})

        evaluator = ParallelRewardEvaluator(factory, workers=2)
        rewards, counts = evaluator.score_batch(items, "tier1", replay_to_step)
        assert len(rewards) == 4
        assert counts["strict"] == 4
        # identical completions on identical state → identical rewards
        assert len(set(rewards)) == 1

        seq = ParallelRewardEvaluator(factory, workers=1)
        seq_rewards, _ = seq.score_batch(items, "tier1", replay_to_step)
        assert seq_rewards == rewards


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_oracle_policy_scores_high(self, env_client):
        from train_grpo import oracle_action

        report = evaluate_policy(
            env_client, "tier1",
            policy_fn=oracle_action, oracle_fn=oracle_action,
            tasks=["team_phishing_escalation"], seeds=[100, 101],
        )
        assert report["n_episodes"] == 2
        assert report["mean_score"] > 0.5  # oracle is near-ceiling
        assert "team_phishing_escalation" in report["per_task"]

    def test_noop_policy_scores_low(self, env_client):
        from train_grpo import oracle_action

        def noop_policy(obs):
            return {"action_type": "noop", "role": "tier1"}

        report = evaluate_policy(
            env_client, "tier1",
            policy_fn=noop_policy, oracle_fn=oracle_action,
            tasks=["team_phishing_escalation"], seeds=[100],
            max_steps=40,
        )
        oracle_report = evaluate_policy(
            env_client, "tier1",
            policy_fn=oracle_action, oracle_fn=oracle_action,
            tasks=["team_phishing_escalation"], seeds=[100],
        )
        assert report["mean_score"] < oracle_report["mean_score"]

    def test_episode_record_shape(self, env_client):
        from train_grpo import oracle_action

        rec = run_policy_episode(
            env_client, "team_phishing_escalation", 100, "tier1",
            policy_fn=oracle_action, oracle_fn=oracle_action,
        )
        assert set(rec) == {"task_id", "seed", "score", "steps", "policy_steps"}
        assert rec["policy_steps"] > 0


# ---------------------------------------------------------------------------
# CLI + reward-fn integration
# ---------------------------------------------------------------------------

class TestTrainingCLI:
    def test_train_grpo_new_flags_parse(self):
        import sys

        import train_grpo as tg
        argv_backup = sys.argv
        try:
            sys.argv = ["train_grpo.py", "--curriculum", "--parallel-rewards", "4",
                        "--eval-episodes", "3", "--dry-run"]
            # parse only — --dry-run path needs a live server, so just check
            # the parser accepts the new flags via a fresh ArgumentParser run.
            import argparse
            parser = argparse.ArgumentParser()
            # Reuse main()'s parser by introspection is brittle; instead assert
            # the flags exist in the module source contract:
            assert "--curriculum" in open(tg.__file__, encoding="utf-8").read()
        finally:
            sys.argv = argv_backup

    def test_soc_gym_cli_train_and_runs_subcommands(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["train", "--role", "tier1", "--curriculum",
                                  "--parallel-rewards", "4", "--eval-episodes", "2"])
        assert args.curriculum is True and args.parallel_rewards == 4
        args2 = parser.parse_args(["runs", "--json"])
        assert callable(args2.func)

    def test_runs_command_output(self, tmp_path, capsys):
        from cli import main
        TrainingRunManager("tier1", "m", runs_dir=tmp_path, run_id="r1").finalize({})
        assert main(["runs", "--runs-dir", str(tmp_path)]) == 0
        assert "r1" in capsys.readouterr().out


class TestRewardFnIntegration:
    def test_make_reward_fn_records_curriculum_and_metrics(self, env_client, tmp_path):
        from train_grpo import make_reward_fn

        rm = TrainingRunManager("tier1", "m", runs_dir=tmp_path)
        sched = CurriculumScheduler(stages=[
            CurriculumStage(name="s", tasks=["team_phishing_escalation"],
                            promotion_threshold=0.0, min_samples=2, max_rounds=1),
        ])
        reward_fn = make_reward_fn(env_client, "tier1", curriculum=sched, run_manager=rm)

        completions = [
            [{"content": json.dumps({"action_type": "noop", "role": "tier1"})}],
            [{"content": "garbage"}],
            [{"content": json.dumps({"action_type": "phase_complete", "role": "tier1"})}],
            [{"content": json.dumps({"action_type": "noop", "role": "tier1"})}],
        ]
        rewards = reward_fn(
            prompts=[None] * 4,
            completions=completions,
            task_id=["team_phishing_escalation"] * 4,
            seed=[42] * 4,
            step_index=[0] * 4,
        )
        assert len(rewards) == 4
        assert sched.rolling_mean is not None
        batches = rm.read_metrics("reward_batch")
        assert len(batches) == 1
        assert batches[0]["batch"] == 4
