# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- GitHub Actions CI: `ruff` lint + `pytest` on a Python 3.11 / 3.12 matrix.
- `Makefile` task runner (`install`, `lint`, `fmt`, `test`, `serve`, `demo`, `plots`).
- `scripts/gen_readme_assets.py` — reproducible README charts from committed
  metadata (task landscape, efficiency-multiplier curve, theme-coverage matrix).
- Animated SOC pipeline SVG (`assets/pipeline_animated.svg`) in the README.
- `CONTRIBUTING.md`, `CHANGELOG.md`, and a ruff/pre-commit dev workflow.
- Regression test for the `_heuristic_baseline_action` classification fallback.

### Fixed
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
