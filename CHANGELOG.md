# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

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
