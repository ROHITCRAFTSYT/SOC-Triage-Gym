# Contributing to SOC-Triage-Gym

Thanks for your interest! This project is an OpenEnv-compliant reinforcement
learning environment that simulates a three-tier Security Operations Center.
Contributions of all sizes are welcome — bug fixes, new scenarios, new graders,
docs, and tests.

## Development setup

```bash
git clone <your-fork-url>
cd SOC-Triage-Gym
python -m pip install -e ".[dev]"     # or: make install
```

Requires **Python 3.11+**. The optional `dev` extra pulls in `pytest`,
`pytest-asyncio`, `httpx`, and `ruff`.

## The dev loop

| Command | What it does |
|---|---|
| `make test`   | Run the full pytest suite |
| `make lint`   | Run `ruff check .` |
| `make fmt`    | Auto-fix lint issues and format |
| `make serve`  | Start the OpenEnv server on `:7860` |
| `make demo`   | Run the one-command judge demo |
| `make plots`  | Regenerate the README charts from repo metadata |

On Windows without `make`, run the underlying commands directly (see the
[`Makefile`](Makefile)).

## Before you open a PR

1. **Lint is clean** — `ruff check .` passes (CI enforces this).
2. **Tests pass** — `pytest -q` is green. Add tests for new behaviour.
3. **Determinism holds** — scenarios must be seed-reproducible. If you touch a
   scenario's RNG draws, keep the call order stable (a `same seed → same alert
   ids` check is the quickest guard).
4. **Rewards are hard to farm** — new reward paths should ship with an
   adversarial regression test. See `tests/test_themes_coverage.py` and
   `tests/test_team_mode.py` for the pattern, and the *Reward Integrity*
   section of the [README](README.md).

## Project layout

See the **Repository Layout** section in the [README](README.md#repository-layout).
The short version:

- `server/`    — FastAPI app + `SOCEnvironment`
- `scenarios/` — scenario generators (+ red-team generator, policy drift)
- `graders/`   — programmatic reward functions + LLM judges
- `tools/`     — SOC tool implementations (enrichment, containment, ticketing…)
- `actors/`    — external NPC actors
- `tests/`     — pytest suite

## Adding a task/scenario

1. Add a scenario generator in `scenarios/` subclassing `BaseScenario`.
2. Add a matching grader in `graders/` subclassing `BaseGrader`.
3. Register the task in `server/app.py` `TASKS` and the scenario/grader
   registries.
4. Add tests covering generation determinism and the grader's scoring.

## Commit messages

Keep the subject line imperative and under ~72 chars, with a body explaining
the *why*. Group unrelated changes into separate commits.

## Code style

`ruff` is the single source of truth (config in `pyproject.toml`): line length
120, PEP 585/604 type hints, sorted imports. Run `make fmt` before committing.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
