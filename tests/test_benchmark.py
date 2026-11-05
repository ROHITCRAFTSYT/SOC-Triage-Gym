"""Benchmark harness + committed leaderboard freshness.

``benchmarks/results.json`` backs the README leaderboard. If a reward or grader
change shifts any baseline score, this test fails until the results are
regenerated with ``make benchmark`` — so the published numbers can't go stale.
"""

import json
import os

from benchmark import DEFAULT_AGENTS, DEFAULT_SEEDS, TASKS, main, render_markdown, run_benchmark

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_committed_results_match_current_code():
    with open(os.path.join(ROOT, "benchmarks", "results.json"), encoding="utf-8") as f:
        committed = json.load(f)
    fresh = run_benchmark(list(DEFAULT_AGENTS), list(TASKS), list(DEFAULT_SEEDS), repeat=1)
    assert fresh["cells"] == committed["cells"], "benchmarks/results.json is stale: run `make benchmark`"
    assert fresh["overall"] == committed["overall"]


def test_committed_markdown_matches_json():
    with open(os.path.join(ROOT, "benchmarks", "results.json"), encoding="utf-8") as f:
        committed = json.load(f)
    with open(os.path.join(ROOT, "benchmarks", "results.md"), encoding="utf-8") as f:
        md = f.read()
    assert render_markdown(committed) in md


def test_benchmark_is_deterministic_and_exits_zero(capsys):
    assert main(["--task", "phishing", "--seeds", "1,2", "--repeat", "2", "--quiet"]) == 0
    assert "Determinism: PASS" in capsys.readouterr().out


def test_oracle_is_skipped_on_team_tasks():
    summary = run_benchmark(["oracle"], ["team_phishing_escalation"], [42], repeat=1)
    assert summary["episodes"] == []
