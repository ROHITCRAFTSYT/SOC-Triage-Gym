"""
soc-gym — unified command-line interface
========================================

One entry point for everything an operator or researcher needs:

    soc-gym serve                 # start the environment server
    soc-gym demo                  # run the 5-beat guided demo
    soc-gym benchmark             # deterministic multi-seed benchmark
    soc-gym tasks                 # print the task catalog
    soc-gym validate              # check a running server's health/endpoints

Installed via [project.scripts] in pyproject.toml. Each subcommand defers to
the existing module (demo.py, benchmark.py, server.app) so behaviour is
identical to calling those modules directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    os.environ.setdefault("SOC_TRIAGE_HOST", args.host)
    os.environ.setdefault("SOC_TRIAGE_PORT", str(args.port))
    if args.api_key:
        os.environ["SOC_GYM_API_KEY"] = args.api_key
    if args.rate_limit:
        os.environ["SOC_GYM_RATE_LIMIT"] = str(args.rate_limit)
    from server.app import main as serve_main

    serve_main()
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    sys.argv = ["demo.py"] + (["--task", args.task] if args.task else [])
    from demo import main as demo_main

    demo_main()
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    argv = ["benchmark.py"]
    if args.task:
        argv += ["--task", args.task]
    if args.seeds:
        argv += ["--seeds", args.seeds]
    sys.argv = argv
    from benchmark import main as benchmark_main

    benchmark_main()
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    from server.app import TASKS

    if args.json:
        print(json.dumps(TASKS, indent=2))
        return 0
    width = max(len(t["id"]) for t in TASKS)
    print(f"{'TASK':<{width}}  {'DIFFICULTY':<10}  {'MAX STEPS':>9}  NAME")
    for t in TASKS:
        print(f"{t['id']:<{width}}  {t['difficulty']:<10}  {t['max_steps']:>9}  {t['name']}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    argv = ["train_grpo.py", "--role", args.role]
    if args.model:
        argv += ["--model", args.model]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.curriculum:
        argv.append("--curriculum")
    if args.parallel_rewards:
        argv += ["--parallel-rewards", str(args.parallel_rewards)]
    if args.eval_episodes:
        argv += ["--eval-episodes", str(args.eval_episodes)]
    if args.unsloth:
        argv.append("--unsloth")
    if args.dry_run:
        argv.append("--dry-run")
    sys.argv = argv
    from train_grpo import main as train_main

    train_main()
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from training.run_manager import TrainingRunManager

    runs = TrainingRunManager.list_runs(args.runs_dir)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print(f"No training runs found under {args.runs_dir}/.")
        return 0
    print(f"{'RUN ID':<45} {'ROLE':<8} {'BEST EVAL':>9}  STATUS")
    for r in runs:
        best = r.get("best_eval_reward")
        best_s = f"{best:.3f}" if isinstance(best, (int, float)) else "-"
        status = "finalized" if r.get("finalized") else "in-progress"
        print(f"{r['run_id']:<45} {r.get('role', '?'):<8} {best_s:>9}  {status}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Probe a running server and report which capabilities are live."""
    import httpx

    base = args.url.rstrip("/")
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    checks: list[tuple[str, str, str]] = [
        ("health", "GET", "/health"),
        ("metadata", "GET", "/metadata"),
        ("task catalog", "GET", "/tasks"),
        ("schema", "GET", "/schema"),
        ("metrics", "GET", "/metrics"),
        ("audit trail", "GET", "/episodes"),
        ("sessions", "GET", "/sessions"),
    ]
    failed = 0
    with httpx.Client(base_url=base, timeout=10.0, headers=headers) as client:
        for name, method, path in checks:
            try:
                r = client.request(method, path)
                ok = r.status_code == 200
            except httpx.HTTPError as exc:
                print(f"  FAIL  {name:<13} {path}  ({type(exc).__name__})")
                failed += 1
                continue
            status = "ok" if ok else f"HTTP {r.status_code}"
            print(f"  {'ok' if ok else 'FAIL':<4}  {name:<13} {path}  ({status})")
            if not ok:
                failed += 1
    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll checks passed — server is production-ready.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc-gym",
        description="SOC-Triage-Gym — train and evaluate AI agents as a coordinated SOC team.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the environment server.")
    p_serve.add_argument("--host", default=os.environ.get("SOC_TRIAGE_HOST", "0.0.0.0"))
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("SOC_TRIAGE_PORT", "7860")))
    p_serve.add_argument("--api-key", default=None, help="Require this API key on all requests.")
    p_serve.add_argument("--rate-limit", type=int, default=None, help="Requests/minute per client (0=off).")
    p_serve.set_defaults(func=_cmd_serve)

    p_demo = sub.add_parser("demo", help="Run the 5-beat guided demo.")
    p_demo.add_argument("--task", default=None)
    p_demo.set_defaults(func=_cmd_demo)

    p_bench = sub.add_parser("benchmark", help="Run the deterministic multi-seed benchmark.")
    p_bench.add_argument("--task", default=None)
    p_bench.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_tasks = sub.add_parser("tasks", help="Print the task catalog.")
    p_tasks.add_argument("--json", action="store_true")
    p_tasks.set_defaults(func=_cmd_tasks)

    p_val = sub.add_parser("validate", help="Probe a running server's endpoints.")
    p_val.add_argument("--url", default="http://localhost:7860")
    p_val.add_argument("--api-key", default=None)
    p_val.set_defaults(func=_cmd_validate)

    p_train = sub.add_parser("train", help="GRPO-train a SOC role against the environment.")
    p_train.add_argument("--role", choices=["tier1", "tier2", "manager"], default="tier1")
    p_train.add_argument("--model", default=None, help="HF model name (default: Qwen2.5-1.5B-Instruct)")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--curriculum", action="store_true",
                         help="Staged easy→hard curriculum with promotion gates")
    p_train.add_argument("--parallel-rewards", type=int, default=0, metavar="N",
                         help="Concurrent reward scoring across N server sessions")
    p_train.add_argument("--eval-episodes", type=int, default=0,
                         help="Held-out eval episodes per task after training")
    p_train.add_argument("--unsloth", action="store_true")
    p_train.add_argument("--dry-run", action="store_true",
                         help="No model — just plot the oracle reward curve")
    p_train.set_defaults(func=_cmd_train)

    p_runs = sub.add_parser("runs", help="List structured training runs and their results.")
    p_runs.add_argument("--runs-dir", default="runs")
    p_runs.add_argument("--json", action="store_true")
    p_runs.set_defaults(func=_cmd_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
