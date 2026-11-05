"""
SOC-Triage-Gym Benchmark
========================
Runs the baseline agents in ``baselines.py`` across every task and a set of
seeds, prints a leaderboard (mean ± std task score per agent), and verifies
determinism: each (agent, task, seed) is played ``--repeat`` times and must
produce identical scores and step-reward sequences.

Runs in-process against ``SOCEnvironment`` — no server, no GPU, no API key —
so the numbers are a pure function of the code at this commit.

Usage:
    python benchmark.py                                 # full leaderboard
    python benchmark.py --task phishing --seed 42       # one cell
    python benchmark.py --agents heuristic,oracle --seeds 1,2,3
    python benchmark.py --write benchmarks/             # refresh committed results

Exit code is 1 if any run is non-deterministic, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

from baselines import AGENTS, SOLO_TASKS, TEAM_TASKS, EpisodeResult, make_agent, run_episode

# Windows consoles default to cp1252 and crash on the Unicode glyphs printed
# below; force UTF-8 so output is identical everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

TASKS: tuple[str, ...] = SOLO_TASKS + TEAM_TASKS
DEFAULT_AGENTS: tuple[str, ...] = ("submit", "random", "heuristic", "oracle")
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 256, 789, 1024)
# The oracle reads ground truth and only plays solo mode.
_UNSUPPORTED = {("oracle", t) for t in TEAM_TASKS}


def run_benchmark(
    agents: list[str],
    tasks: list[str],
    seeds: list[int],
    repeat: int = 2,
    progress: bool = False,
) -> dict:
    """Run every (agent, task, seed) ``repeat`` times and summarise.

    Returns a JSON-serialisable dict with per-episode results, per-cell
    mean/std, per-agent overall means, and a list of determinism failures.
    """
    episodes: list[EpisodeResult] = []
    failures: list[str] = []
    for agent_name in agents:
        for task in tasks:
            if (agent_name, task) in _UNSUPPORTED:
                continue
            for seed in seeds:
                runs = [run_episode(make_agent(agent_name), task, seed) for _ in range(max(1, repeat))]
                first = runs[0]
                for other in runs[1:]:
                    if other.score != first.score or other.step_rewards != first.step_rewards:
                        failures.append(f"{agent_name}/{task}/seed={seed}: {first.score} vs {other.score}")
                episodes.append(first)
                if progress:
                    print(f"  {agent_name:<10} {task:<26} seed={seed:<5} score={first.score:.3f} steps={first.steps}")

    cells: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ep in episodes:
        by_cell[(ep.agent, ep.task_id)].append(ep.score)
    for (agent_name, task), scores in by_cell.items():
        cells[task][agent_name] = {
            "mean": round(statistics.fmean(scores), 4),
            "std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "n": len(scores),
        }

    overall: dict[str, float] = {}
    for agent_name in agents:
        means = [cells[t][agent_name]["mean"] for t in tasks if agent_name in cells.get(t, {})]
        if means:
            overall[agent_name] = round(statistics.fmean(means), 4)

    return {
        "agents": agents,
        "tasks": tasks,
        "seeds": seeds,
        "repeat": repeat,
        "cells": {t: cells[t] for t in tasks if t in cells},
        "overall": overall,
        "deterministic": not failures,
        "determinism_failures": failures,
        "episodes": [ep.as_dict() for ep in episodes],
    }


def render_markdown(summary: dict) -> str:
    """Render the leaderboard as a GitHub-flavoured markdown table."""
    agents = summary["agents"]
    lines = [
        "| Task | " + " | ".join(f"`{a}`" for a in agents) + " |",
        "|---|" + "---:|" * len(agents),
    ]
    for task in summary["tasks"]:
        row = summary["cells"].get(task, {})
        cells = []
        for a in agents:
            c = row.get(a)
            cells.append("n/a" if c is None else f"{c['mean']:.3f} ± {c['std']:.3f}")
        lines.append(f"| `{task}` | " + " | ".join(cells) + " |")
    overall = summary["overall"]
    lines.append(
        "| **Mean over tasks** | " + " | ".join(f"**{overall[a]:.3f}**" if a in overall else "n/a" for a in agents) + " |"
    )
    seeds = ", ".join(str(s) for s in summary["seeds"])
    lines.append("")
    lines.append(
        f"Task score in [0.001, 0.999], mean ± population std over seeds {seeds}. "
        f"Determinism: {'PASS' if summary['deterministic'] else 'FAIL'} "
        f"({summary['repeat']} repeats per cell)."
    )
    return "\n".join(lines)


def _parse_list(value: str | None, default: tuple, cast=str) -> list:
    if not value:
        return list(default)
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOC-Triage-Gym baseline benchmark")
    parser.add_argument("--task", choices=TASKS, default=None, help="Benchmark a single task")
    parser.add_argument("--tasks", default=None, help="Comma-separated task list (default: all)")
    parser.add_argument("--agents", default=None, help=f"Comma-separated agents (default: {','.join(DEFAULT_AGENTS)})")
    parser.add_argument("--seed", type=int, default=None, help="Single-seed shorthand (overrides --seeds)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (default: 42,123,256,789,1024)")
    parser.add_argument("--repeat", type=int, default=2, help="Repeats per cell for the determinism check")
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON")
    parser.add_argument("--quiet", action="store_true", help="Skip per-episode progress lines")
    parser.add_argument("--write", metavar="DIR", default=None, help="Write results.json and results.md into DIR")
    args = parser.parse_args(argv)

    tasks = [args.task] if args.task else _parse_list(args.tasks, TASKS)
    agents = _parse_list(args.agents, DEFAULT_AGENTS)
    seeds = [args.seed] if args.seed is not None else _parse_list(args.seeds, DEFAULT_SEEDS, int)
    for name in agents:
        if name not in AGENTS:
            parser.error(f"unknown agent '{name}' (valid: {', '.join(sorted(AGENTS))})")
    for task in tasks:
        if task not in TASKS:
            parser.error(f"unknown task '{task}' (valid: {', '.join(TASKS)})")

    t0 = time.perf_counter()
    summary = run_benchmark(agents, tasks, seeds, repeat=args.repeat, progress=not (args.quiet or args.json))
    elapsed = time.perf_counter() - t0

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n## SOC-Triage-Gym Baseline Leaderboard\n")
        print(render_markdown(summary))
        for failure in summary["determinism_failures"]:
            print(f"  NON-DETERMINISTIC: {failure}")
        print(f"\n{len(summary['episodes'])} episodes x {args.repeat} repeats in {elapsed:.1f}s")

    if args.write:
        os.makedirs(args.write, exist_ok=True)
        stable = {k: v for k, v in summary.items() if k != "episodes"}
        with open(os.path.join(args.write, "results.json"), "w", encoding="utf-8") as f:
            json.dump(stable, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(args.write, "results.md"), "w", encoding="utf-8") as f:
            f.write("<!-- Generated by `python benchmark.py --write benchmarks/` — do not edit by hand. -->\n\n")
            f.write(render_markdown(summary) + "\n")

    return 0 if summary["deterministic"] else 1


if __name__ == "__main__":
    sys.exit(main())
