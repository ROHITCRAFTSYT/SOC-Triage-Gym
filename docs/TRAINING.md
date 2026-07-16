# Training Guide — GRPO with the v0.3 Toolkit

How to train SOC analyst policies against the environment, from a free-tier
Colab T4 to a multi-session production server. The toolkit lives in
[`training/`](../training/) and is wired into [`train_grpo.py`](../train_grpo.py);
every feature is opt-in and the original single-shot path is unchanged.

## TL;DR — recommended recipe

```bash
# 1. Start the environment server (or point SERVER_URL at a shared one)
soc-gym serve

# 2. Train Tier-1 with the full v0.3 stack
python train_grpo.py --role tier1 --model Qwen/Qwen2.5-1.5B-Instruct --unsloth \
    --curriculum \
    --parallel-rewards 4 \
    --eval-episodes 5 \
    --early-stop-patience 20

# 3. Inspect the run
soc-gym runs
cat runs/<run_id>/MANIFEST.json
```

## How GRPO works here (unchanged core)

Training uses **real per-step GRPO**, not full-episode rollouts:

```
Dataset row = (observation at step_index, task_id, seed, step_index)
Reward      = env.step(model_action).reward + JSON-validity shaping
```

For every group-sampled completion the reward function replays the
environment deterministically to `step_index` (same seed → same state),
applies the model's single action, and returns the env's immediate blended
step reward. A shaping bonus (`strict` +0.05 / `loose` +0.01 / `fallback`
−0.10) bootstraps sub-3B models out of the malformed-JSON trap.

## The v0.3 enhancements

### 1. Staged curriculum (`--curriculum`)

Small models drown in sparse rewards on hard tasks. The curriculum
([`training/curriculum.py`](../training/curriculum.py)) trains in three
stages and only promotes when the rolling mean step reward clears the
stage's gate:

| Stage | Tasks | Promotion gate |
|---|---|---|
| `foundations` | team_phishing_escalation | mean reward ≥ 0.05 |
| `noisy-queues` | + team_lateral_team | ≥ 0.08 |
| `adversarial` | team_lateral_team + **red_team_generated** | ≥ 0.10 |

A stage that can't clear its gate is retrained up to `max_rounds` times and
then force-advanced (recorded as `forced_advance` in the run history, so you
can see exactly where the policy struggled). The final stage generates fresh
adversarial scenarios via `POST /generate_scenario` before each round — the
RLVE loop: as the blue policy improves, the red-team generator keeps
producing scenarios in the trainable sweet spot.

Custom curricula are plain data:

```python
from training.curriculum import CurriculumScheduler, CurriculumStage

scheduler = CurriculumScheduler(stages=[
    CurriculumStage("warmup", ["team_phishing_escalation"], promotion_threshold=0.03),
    CurriculumStage("full",   ["team_lateral_team"],        promotion_threshold=0.10),
])
```

### 2. Parallel reward scoring (`--parallel-rewards N`)

Reward evaluation dominates wall-clock time in env-in-the-loop GRPO: each
completion needs a full reset + oracle replay + step over HTTP, and a group
size of 8 means 8 of those per prompt. `ParallelRewardEvaluator`
([`training/rewards.py`](../training/rewards.py)) scores the whole group
concurrently, giving each worker thread its **own isolated server session**
(`X-Session-ID: grpo-worker-N`) so replays never collide — this is what the
v0.2.0 multi-session server was built for.

Rule of thumb: `--parallel-rewards` = GRPO group size (8), capped by server
CPU. Requires server ≥ 0.2.0; with `--parallel-rewards 1` behaviour is
exactly sequential.

### 3. Structured run storage (`runs/<run_id>/`)

Every `train()` call creates a self-describing run directory
([`training/run_manager.py`](../training/run_manager.py)):

```
runs/20260716-093000-tier1-qwen2.5-1.5b-instruct/
  config.json      full hyperparameters + git commit/branch/dirty flag,
                   server URL, seed range, curriculum stages
  metrics.jsonl    append-only event stream: reward batches (with parse-
                   quality percentages), trainer logs, curriculum decisions,
                   eval results — one JSON object per line
  checkpoints/     trainer checkpoints (save_total_limit-rotated)
  best/best.json   pointer to the best checkpoint by held-out eval reward
  eval/            held-out evaluation reports
  MODEL_CARD.md    auto-generated, publish-ready model card
  MANIFEST.json    final summary (written even on the CPU dry-run fallback)
```

Analyze a run in three lines:

```python
import json
events = [json.loads(l) for l in open("runs/<id>/metrics.jsonl")]
rewards = [e["mean_reward"] for e in events if e["event"] == "reward_batch"]
```

`soc-gym runs` lists all runs with their best eval score;
`scripts/hf_publish.py` can push the checkpoint + model card to the Hub.

### 4. Held-out evaluation & best-checkpoint tracking (`--eval-episodes N`)

After training, the trained model itself (greedy decoding) drives the
trained role through full team episodes on **disjoint seeds 100+** — the
same protocol as `benchmark.py` — with the oracle driving the other roles.
The report lands in `eval/`, and `best/best.json` tracks the highest mean
score across runs of the same manager.

### 5. Early stopping (`--early-stop-patience N`)

`SOCGymCallback` ([`training/callbacks.py`](../training/callbacks.py))
watches the trainer's reward logs and stops training when the reward hasn't
improved by ≥ 0.005 for N consecutive log events. A 4-hour budget that
converges after 90 minutes stops itself.

### 6. Better trainer defaults

Applied automatically (and skipped gracefully on old TRL versions):
cosine LR schedule with 5% warmup (`--lr-scheduler`, `--warmup-ratio`),
gradient clipping at 1.0, `save_total_limit=3` checkpoint rotation, and
bf16 autocast when the GPU supports it.

## Compute recipes

| Budget | Recipe |
|---|---|
| **Colab T4 (free)** | `SOC_TRAIN_N_SEEDS=15 python train_grpo.py --role tier1 --unsloth --curriculum --parallel-rewards 4 --early-stop-patience 15` |
| **Kaggle T4 (30h/wk)** | `python scripts/train_and_evaluate.py` with `NUM_EPOCHS=2` |
| **A100/H100** | Full 50 seeds, `--parallel-rewards 8 --eval-episodes 10`, then repeat for `--role tier2` and `--role manager` with the trained Tier-1 frozen |
| **No GPU** | `python train_grpo.py --role tier1 --dry-run` (oracle reward curve) or `--compare` (learnable-gap plot). `train()` without the ML stack degrades to the dry-run and still writes the run manifest. |

## Training against a shared production server

The toolkit composes with the v0.2.0 production features
([docs/PRODUCTION.md](PRODUCTION.md)): point `SERVER_URL` at a shared
deployment and each trainer's reward workers use their own sessions — two
teammates can train different roles against one server without interference.
Watch training load on the server via `GET /metrics`
(`socgym_episodes_started_total`, `socgym_active_sessions`).

## Testing

The entire toolkit is CPU-testable: `pytest tests/test_training.py -q`
(25 tests) exercises the curriculum gates, run storage, reward scoring, and
the parallel evaluator against an in-process server — no torch required.
