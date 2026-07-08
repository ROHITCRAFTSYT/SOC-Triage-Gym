"""
SOC-Triage-Gym Multi-Seed Reproducibility Benchmark
=====================================================
Runs all tasks across multiple seeds using the heuristic agent (no LLM needed).
Prints a markdown table of scores and verifies determinism (same seed = same score).

Usage:
    python benchmark.py
    python benchmark.py --seeds 42,123,256
    python benchmark.py --server http://localhost:7860
"""

import argparse
import subprocess
import sys
import time

import httpx

# Windows consoles default to cp1252 and crash on the Unicode glyphs printed
# below; force UTF-8 so output is identical everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

TASKS = ["phishing", "lateral_movement", "queue_management", "insider_threat", "apt_campaign"]
DEFAULT_SEEDS = [42, 123, 256, 789, 1024]
DEFAULT_SERVER = "http://localhost:7860"


def ensure_server(server_url: str) -> subprocess.Popen | None:
    """Return a server subprocess if one needs to be started, else None."""
    try:
        r = httpx.get(f"{server_url}/health", timeout=3)
        r.raise_for_status()
        print(f"[OK] Server already running at {server_url}")
        return None
    except Exception:
        pass

    print(f"[INFO] Starting server subprocess on {server_url} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            r = httpx.get(f"{server_url}/health", timeout=3)
            r.raise_for_status()
            print(f"[OK] Server started at {server_url}")
            return proc
        except Exception:
            pass
    print("[ERROR] Server failed to start after 30 seconds.")
    proc.terminate()
    sys.exit(1)


def run_baseline(client: httpx.Client, task_id: str, seed: int) -> float:
    """Run /baseline for a single task+seed and return the score."""
    resp = client.post("/baseline", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    data = resp.json()
    return data.get("score", 0.0)


def main():
    parser = argparse.ArgumentParser(description="SOC-Triage-Gym reproducibility benchmark")
    parser.add_argument("--task", type=str, default=None, choices=TASKS,
                        help="Benchmark a single task instead of all five")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single-seed shorthand (overrides --seeds)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list (default: 42,123,256,789,1024)")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER,
                        help=f"Server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--repeat", type=int, default=2,
                        help="How many times to repeat each (task, seed) for determinism check (default: 2)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS
    server_url = args.server
    repeats = args.repeat

    server_proc = ensure_server(server_url)

    # results[task][seed] = [score_run1, score_run2, ...]
    results: dict[str, dict[int, list[float]]] = {t: {s: [] for s in seeds} for t in tasks}

    total_runs = len(tasks) * len(seeds) * repeats
    run_idx = 0
    t0 = time.time()

    try:
        with httpx.Client(base_url=server_url, timeout=120) as client:
            for rep in range(repeats):
                for task in tasks:
                    for seed in seeds:
                        run_idx += 1
                        score = run_baseline(client, task, seed)
                        results[task][seed].append(score)
                        pct = (score * 100)
                        print(f"  [{run_idx}/{total_runs}] {task:<22} seed={seed:<5} run={rep+1}  score={pct:6.2f}%")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        if server_proc:
            server_proc.terminate()
        sys.exit(1)

    elapsed = time.time() - t0

    # ---- Print markdown table ----
    print("\n")
    print("## SOC-Triage-Gym Reproducibility Benchmark")
    print()

    # Header
    seed_hdrs = " | ".join(f"Seed {s}" for s in seeds)
    print(f"| Task | {seed_hdrs} | Mean |")
    print(f"|{'---|' * (len(seeds) + 2)}")

    task_means = []
    for task in tasks:
        cells = []
        row_scores = []
        for seed in seeds:
            scores = results[task][seed]
            avg = sum(scores) / len(scores)
            row_scores.append(avg)
            cells.append(f"{avg*100:.1f}%")
        mean = sum(row_scores) / len(row_scores)
        task_means.append(mean)
        print(f"| {task:<22} | {' | '.join(cells)} | **{mean*100:.1f}%** |")

    overall = sum(task_means) / len(task_means)
    print(f"| **Overall Average** | {' | '.join(['—']*len(seeds))} | **{overall*100:.1f}%** |")
    print()

    # ---- Determinism check ----
    print("## Determinism Check")
    print()
    all_deterministic = True
    for task in tasks:
        for seed in seeds:
            scores = results[task][seed]
            if len(set(round(s, 10) for s in scores)) > 1:
                print(f"  FAIL: {task} seed={seed} produced different scores: {scores}")
                all_deterministic = False

    if all_deterministic:
        print(f"  PASS: All {len(tasks)*len(seeds)} (task, seed) pairs produced identical scores across {repeats} runs.")
    else:
        print("  FAIL: Non-deterministic results detected!")

    print(f"\nTotal time: {elapsed:.1f}s ({total_runs} runs)")
    print(f"Average per run: {elapsed/max(1,total_runs):.2f}s")

    # Cleanup
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
