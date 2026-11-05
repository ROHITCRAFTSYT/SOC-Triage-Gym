# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed (reward/grading — affects comparability with earlier runs)
- `ransomware` and `cloud_compromise` had no grader: every episode scored 0
  ("No grader registered"). Both now use the single-alert grader.
- 13 ATT&CK techniques expected by scenario ground truth (cloud_compromise,
  ransomware, insider_threat, apt_campaign) were missing from the catalog, so
  `map_technique` rejected them and full technique credit was unreachable.
  The scenario validator now flags uncatalogued expected techniques.
- Loop-penalty signature ignored `technique_id`, `response_action`,
  `classification` and `ticket_id`, so distinct correct actions on one alert
  were penalised as loops (−0.30 on a perfect phishing run). Per-step shaping
  rewards for correct trajectories are now higher; terminal scores unchanged.
- Empty submissions earned free credit (up to 0.24 on queue_management, 0.40 on
  false-positive single-alert seeds) from efficiency, policy-compliance and
  "no action on an FP" components. These are now gated on triage coverage, and
  inaction scores the 0.001 floor on every task.
- Ticket IDs were `uuid4`, making team-mode observations and the manager-judge
  trajectory hash non-deterministic. They are now per-episode sequential
  (`TKT-EP0001`).
- Ending team-mode triage with zero escalations announced a −0.10 penalty but
  applied it only to the running total. The step reward returned (what
  per-step GRPO trains on) was +0.001, making "phase_complete immediately" one
  of the few non-negative actions an untrained policy could find. The penalty
  now reaches the step reward, and applies only when the scenario requires an
  escalation.
- Team-mode feedback told Tier-1 "All alerts classified. Call
  submit_investigation to finalize." In team mode that ends the whole episode
  and skips Tier-2 and the Manager: a correctly-triaging policy that followed
  the hint scored 0.001 (vs 0.999 escalating). Team mode now says to escalate
  true positives and phase_complete; solo mode is unchanged.
- In team mode any role could end the episode with `submit_investigation`:
  the shortcut ran before the per-role action check, so even Tier-2 (whose
  action set never allowed it) could skip the remaining phases. Only the
  Manager may submit now; Tier-1/Tier-2 get the standard wrong-role penalty
  and play continues. `inference.py`'s timeout path and the baseline `submit`
  agent use `phase_complete` for non-manager roles.
- Held-out text-policy eval (`scripts/train_and_evaluate.py`) aborted an
  episode on the first rejected action and kept nothing but the score. It now
  records every completion (parse quality, action type, env response) to
  `eval_trace.jsonl`, adds a diagnostics block to `training_summary.json`, and
  continues past rejected actions. At least five distinct failure modes all
  score exactly 0.001 on team tasks, so the score alone can't say why.

### Changed (training — invalidates comparison with the published adapter)
- LLM-policy prompts moved to `training/prompts.py` and rewritten. The old
  observation prompt hid the agent's own progress (classifications, evidence,
  escalations) and the env's feedback, showed only IP indicators, and the
  system prompt listed no action fields or allowed values. The new one shows
  per-alert status, the last result, all indicator types (hashes for Tier-2
  only), and field schemas generated from the `models` enums. The prompt's
  step number now comes from the env, fixing a train (0-based) / eval
  (1-based) mismatch. A text-only policy reading just this prompt matches the
  full-observation heuristic (0.999 / 0.800 on the team tasks).
- `max_prompt_length` 512 → 768 (worst case is ~676 Qwen2.5 tokens), eval
  `max_seq_length` 768 → 1024, notebook `max_seq_length` 768 → 896.
  `check_prompt_budget` tokenises every training prompt with the real
  tokenizer and fails if any would be truncated (TRL truncates from the left,
  dropping the system prompt first). The notebook's 3B default may now exceed
  T4 memory; the published adapter is 1.5B.

### Added
- `baselines.py` (submit / random / heuristic / oracle agents + in-process
  runner) and a rewritten `benchmark.py` that prints a leaderboard, checks
  determinism across repeats and exits 1 on divergence. Results are committed in
  `benchmarks/` and checked for freshness in CI. The old HTTP `--server` mode is
  removed (the `/baseline` endpoint remains).
- Structured logging (`server/logging_config.py`): JSON or text via
  `SOC_GYM_LOG_FORMAT`, request IDs (`X-Request-ID`), `episode.reset` /
  `episode.complete` events.
- Input validation: bounded string lengths on `SOCAction`, strict uint32 seeds,
  `time_window_hours` 1–168, ranges on curriculum and reward-config knobs.
- README: dropped the hackathon pitch (prize/theme claims, "why this project
  wins", judge-demo and judging-criteria sections); the theme table is now a
  neutral feature map. Training results are labelled as pre-fix, and the
  "trained policy underfit" explanation is replaced by what the eval
  investigation actually found.
- CI: blocking format check, Python 3.13, coverage gate, benchmark/determinism
  job, Docker build + smoke test, zizmor workflow audit.
- Docker: base image pinned by digest (Dependabot-managed), Python healthcheck
  instead of curl, JSON logs by default.
- New `cloud_compromise` scenario: a single-alert cloud identity-compromise task
  with true-positive (impossible-travel login, IAM access-key creation, a bucket
  made public, a security group opened, CloudTrail `StopLogging`) and
  false-positive (SSO+MFA admin on approved travel) variants. Exercises the
  `auth` and `cloud_trail` log sources. Registered, catalogued, and tested.
- New `ransomware` scenario: a single-alert endpoint ransomware triage task with
  true-positive (mass encryption, shadow-copy deletion via `vssadmin`, ransom
  note, C2 beacon) and false-positive (legitimate signed backup agent) variants.
  Registered in `SCENARIO_REGISTRY`, surfaced in the task catalog and
  `openenv.yaml`, and covered by scenario tests.
- `get_parent_technique` and `get_sub_techniques` helpers in
  `data/mitre_attack.py` for navigating the technique hierarchy (sub-technique →
  base, base → sub-techniques). Both re-exported from the `data` package, which
  now also surfaces `get_technique_name` and `normalize_technique_id`.
- `soc-gym techniques` CLI subcommand: prints the MITRE ATT&CK technique catalog
  the graders recognize, with sub-techniques indented under their base. Supports
  `--tactic <tactic>` filtering and `--json`.

### Fixed
- `normalize_technique_id` now strips wrapping brackets/quotes and trailing
  sentence punctuation, so agent answers like `(T1059.001)`, `T1566.001.` and
  `T1566,` normalize to their bare IDs instead of being dropped by exact-match
  grading. The internal sub-technique dot is preserved.

### Security
- Resolved all 71 open Dependabot alerts by upgrading pillow 13.0.0, PyJWT
  2.15.1, certifi 2026.10.23, urllib3 2.8.0, anyio 4.14.2, cryptography 50.0.2,
  fastmcp 3.4.11, mcp 1.30.0, fsspec 2026.9.0, filelock 4.2.4, brotli 1.2.1 and
  opentelemetry-api 1.45.1. gradio still pins `pillow<13`, so Pillow is lifted
  via a `[tool.uv]` override in the lockfile and a post-install upgrade in the
  Dockerfile. fastmcp and mcp are capped below their next majors (`<4`, `<2`)
  because openenv-core targets the fastmcp 3 API.

## [0.3.0] — 2026-07-16 — Training toolkit

GRPO training grows a production-grade harness: staged curriculum learning,
session-parallel reward scoring, structured run storage, held-out
evaluation, and early stopping. All opt-in; the original single-shot
training path is byte-compatible. Guide: [docs/TRAINING.md](docs/TRAINING.md).

### Added
- **`training/` package** — CPU-importable toolkit (no torch/trl needed):
  - `training/curriculum.py` — 3-stage easy→hard task ladder with rolling
    reward promotion gates; final stage regenerates adaptive
    `red_team_generated` scenarios each round (automated RLVE loop).
  - `training/rewards.py` — parse-quality classification, action extraction,
    per-completion scoring, and `ParallelRewardEvaluator`: scores GRPO
    completion groups concurrently across isolated server sessions
    (`X-Session-ID: grpo-worker-N`) for a near-linear reward-latency speedup.
  - `training/run_manager.py` — structured `runs/<run_id>/` storage:
    config + git provenance, append-only `metrics.jsonl`, best-checkpoint
    tracking, auto-generated `MODEL_CARD.md`, final `MANIFEST.json`.
  - `training/evaluation.py` — held-out policy evaluation on disjoint seeds
    (100+) with per-task breakdowns.
  - `training/callbacks.py` — TRL callback streaming trainer logs into the
    run manager and applying reward-plateau early stopping.
- **`train_grpo.py` flags**: `--curriculum`, `--parallel-rewards N`,
  `--runs-dir`, `--eval-episodes N`, `--early-stop-patience N`,
  `--lr-scheduler`, `--warmup-ratio`; trainer config now applies cosine LR +
  warmup, gradient clipping, `save_total_limit=3`, and automatic bf16
  (all guarded for older TRL versions).
- **CLI**: `soc-gym train` (wraps train_grpo with the new flags) and
  `soc-gym runs` (lists structured runs with best eval scores).
- 25 new CPU-only tests (`tests/test_training.py`) — suite now 162 passing.

### Changed
- `train_grpo.py` parsing/shaping helpers moved to `training/rewards.py`
  (`parse_action_from_text` re-exported for script compatibility);
  `train()` without the ML stack now degrades to the oracle dry-run *and*
  still finalizes the run manifest instead of crashing on missing `peft`.
- `runs/` and `checkpoints/` added to `.gitignore`.
- Version bumped to 0.3.0.

## [0.2.0] — 2026-07-16 — Production hardening

The release that turns the hackathon environment into a service a company
can run: multi-tenant, authenticated, observable, and auditable. Everything
is opt-in and backward compatible with stock OpenEnv clients. Deployment
guide: [docs/PRODUCTION.md](docs/PRODUCTION.md).

### Added
- **Multi-session concurrency** (`server/sessions.py`): per-tenant isolated
  environments (episode + NPC actors + policy drift + ticketing + expert
  rotation + reward blend, each behind its own lock), selected via the
  `X-Session-ID` header or `session_id` field on `POST /reset`. TTL and LRU
  eviction (`SOC_GYM_SESSION_TTL`, `SOC_GYM_MAX_SESSIONS`); admin endpoints
  `GET /sessions` and `DELETE /sessions/{id}`.
- **API-key auth + rate limiting** (`server/security.py`): opt-in via
  `SOC_GYM_API_KEY` (constant-time compare; `Bearer` or `X-API-Key`) and
  `SOC_GYM_RATE_LIMIT` (token bucket per client, `429` + `Retry-After`).
  Health, landing pages, and API docs stay open.
- **Prometheus metrics** (`server/metrics.py`, `GET /metrics`): request
  counts/latency by route template, episodes started/completed, steps, and
  reward accumulators per task, active-session gauge — no new dependencies.
- **Episode audit trail** (`server/audit.py`): every reset/step recorded
  (action, reward, running total, role, timestamp); `GET /episodes`,
  `GET /episodes/{id}/trace` (JSON or JSONL for SIEM export); bounded window
  (`SOC_GYM_AUDIT_MAX_EPISODES`) plus optional durable JSONL export
  (`SOC_GYM_AUDIT_DIR`).
- **`soc-gym` CLI** (`cli.py`): `serve`, `demo`, `benchmark`, `tasks`,
  `validate` (deploy smoke test probing all production endpoints).
- **SDK upgrade** (`client.py`): automatic retries with exponential backoff
  on connection errors/429/5xx, session + API-key wiring, `run_episode()`
  policy driver, and accessors for grader/tasks/audit/metrics.
- **Deployment**: `docker-compose.yml` (healthcheck, read-only root FS,
  audit volume, env-driven config); Dockerfile now runs as a non-root user;
  `.env.example` documents every `SOC_GYM_*` knob.
- 26 new tests (`tests/test_production.py`) covering session isolation and
  eviction, auth on/off, rate limiting, metrics format, audit replay, CLI,
  and SDK wiring — suite now 137 passing.

### Changed
- `server/app.py` refactored from module-global environment state to the
  session registry; `/reset`, `/step`, `/state`, MCP tools, and all v3 theme
  endpoints are session-aware (default session preserves the original
  single-tenant behaviour exactly).
- Version bumped to 0.2.0 (`pyproject.toml`, `/health`, `/metadata`, OpenAPI).

### Added (presentation & developer experience, landed pre-0.2.0)
- `demo_live.py` — presenter-paced five-act live demo (reset → ticket bus →
  grader breakdown → learnable gap → safeguards) with `--auto` rehearsal and
  `--train` dry-run modes; used for the CCCL BLR6 (Securing AI Agents) meetup.
- `site/` — self-contained presentation website (immersive scroll build in
  `index.html`, flat fallback in `classic.html`) plus `TALK_PLAN.md`,
  `DEMO_RUNBOOK.md`, and `DESIGN_ARCHITECTURE.md`.
- GitHub Actions CI: `ruff` lint + `pytest` on a Python 3.11 / 3.12 matrix.
- `Makefile` task runner (`install`, `lint`, `fmt`, `test`, `serve`, `demo`, `plots`).
- `scripts/gen_readme_assets.py` — reproducible README charts from committed
  metadata (task landscape, efficiency-multiplier curve, theme-coverage matrix).
- Animated SOC pipeline SVG (`assets/pipeline_animated.svg`) in the README.
- `CONTRIBUTING.md`, `CHANGELOG.md`, and a ruff/pre-commit dev workflow.
- Regression test for the `_heuristic_baseline_action` classification fallback.

### Fixed
- **Windows console crash**: every CLI entry point (`demo.py`, `inference.py`,
  `benchmark.py`, `train_grpo.py`, `scripts/replay.py`,
  `scripts/train_and_evaluate.py`) raised `UnicodeEncodeError` on stock cp1252
  consoles when printing Unicode glyphs (✓ → ①). All now force UTF-8 stdout/stderr.
- **Lint**: stdlib/third-party import separation in `server/app.py:main()`
  (`ruff` I001) that failed `ruff check .` in CI.
- **Local server exposure**: the auto-start convenience path in `demo.py`,
  `inference.py`, `benchmark.py`, and `scripts/train_and_evaluate.py` bound
  uvicorn to `0.0.0.0` while only ever connecting to localhost; now binds
  `127.0.0.1` (deployments keep `0.0.0.0` via `server.app:main` /
  `SOC_TRIAGE_HOST`).
- **Stale links**: Colab badge/links and the judges' clone command pointed at
  the pre-rename `-Metas-OpenEnv-2` repository; now point at `SOC-Triage-Gym`.
  `JUDGES_START_HERE.md` also referenced a nonexistent `requirements.txt`
  (→ `pip install -e ".[dev]"`) and an outdated test count.
- **Build backend**: `pyproject.toml` used an invalid PEP 517 backend
  (`setuptools.backends.legacy:build`); corrected to `setuptools.build_meta`,
  which unblocks `pip install -e .`.
- **Latent `NameError`**: `server/app.py` referenced `AlertClassification` in
  `_heuristic_baseline_action` without importing it. Added the import and a
  regression test.
- Proper exception chaining (`raise ... from`) across 12 error handlers so the
  original cause is preserved (client errors) or intentionally suppressed after
  logging (500s).

### Changed
- Adopted `ruff` as the single lint/format source of truth (config in
  `pyproject.toml`): line length 120, PEP 585/604 type hints, sorted imports.
- Removed 66 unused imports and dead local variables; RNG-load-bearing draws in
  scenarios kept (call preserved) to protect seed determinism.
- Hardened `.gitattributes` to pin text files to LF (Makefile/YAML/shell were
  being rewritten to CRLF on Windows checkouts).

---

*Earlier history predates this changelog; see the git log for the full record
of the OpenEnv hackathon build-out (8 tasks, 3-role team, adaptive red-team
curriculum, GRPO training pipeline).*
