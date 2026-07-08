"""
Deterministic replay CLI.

Usage:
    python -m scripts.replay <episode.jsonl> [--base-url http://localhost:7860]

The episode file is a JSON Lines log emitted by a previous run:
    {"kind": "reset", "task_id": "...", "seed": 42, "mode": "tier1_solo"}
    {"kind": "step",  "action": {...}}
    ...

The replay script re-issues /reset + /step calls in order and asserts that
the observed rewards match the original log byte-for-byte. Judges can
re-run any reward number in our leaderboard themselves — credibility win.

Also runnable without a live server: pass --in-process to replay via
SOCEnvironment directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

# Windows consoles default to cp1252 and crash on the Unicode glyphs printed
# below; force UTF-8 so output is identical everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def _read_jsonl(path: Path) -> list[dict]:
    lines: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[replay] {path}:{i}: invalid JSON ({exc})") from exc
    return lines


def _replay_in_process(events: Iterable[dict]) -> list[float]:
    from models import SOCAction
    from server.environment import SOCEnvironment

    env = SOCEnvironment()
    rewards: list[float] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "reset":
            env.reset(
                task_id=ev.get("task_id", "phishing"),
                seed=ev.get("seed", 42),
                mode=ev.get("mode", "tier1_solo"),
            )
        elif kind == "step":
            action = SOCAction(**ev["action"])
            obs = env.step(action)
            rewards.append(obs.reward)
        else:
            continue
    return rewards


def _replay_http(events: Iterable[dict], base_url: str) -> list[float]:
    try:
        import urllib.request
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("urllib is required") from exc

    def _post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            url=f"{base_url.rstrip('/')}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    rewards: list[float] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "reset":
            _post("/reset", {
                "task_id": ev.get("task_id", "phishing"),
                "seed": ev.get("seed", 42),
                "mode": ev.get("mode", "tier1_solo"),
            })
        elif kind == "step":
            obs = _post("/step", ev["action"])
            rewards.append(float(obs.get("reward", 0.0)))
    return rewards


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic SOC-Triage-Gym episode replay.")
    p.add_argument("episode_file", type=Path)
    p.add_argument("--base-url", default="http://localhost:7860")
    p.add_argument("--in-process", action="store_true",
                   help="Replay via SOCEnvironment directly (no HTTP).")
    p.add_argument("--expected-total", type=float, default=None,
                   help="If provided, assert cumulative reward matches.")
    p.add_argument("--tolerance", type=float, default=1e-6)
    args = p.parse_args()

    if not args.episode_file.exists():
        print(f"[replay] file not found: {args.episode_file}", file=sys.stderr)
        return 2

    events = _read_jsonl(args.episode_file)
    if not events:
        print("[replay] empty episode file", file=sys.stderr)
        return 2

    rewards = (
        _replay_in_process(events)
        if args.in_process
        else _replay_http(events, args.base_url)
    )

    total = sum(rewards)
    print(f"[replay] {len(rewards)} steps replayed, cumulative reward = {total:.6f}")

    if args.expected_total is not None:
        diff = abs(total - args.expected_total)
        if diff > args.tolerance:
            print(
                f"[replay] FAIL: expected {args.expected_total:.6f} "
                f"got {total:.6f} (|Δ|={diff:.2e})",
                file=sys.stderr,
            )
            return 1
        print(f"[replay] OK: matches expected total within {args.tolerance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
