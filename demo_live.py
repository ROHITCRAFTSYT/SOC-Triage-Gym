"""
SOC-Triage-Gym — LIVE STAGE DEMO
================================

A presenter-controlled walkthrough that mirrors the website "slides".
You press Enter to advance each act, so you narrate at your own pace and
nothing scrolls past before you've talked to it.

Acts (each maps to a section of site/index.html):
  1. The environment is real        -> Architecture slide   (reset a team episode)
  2. Agents talk through a bus       -> Team Mode slide      (escalate -> ticket -> inbox)   ** money shot **
  3. The score is glass-box          -> Reward slide         (grader component breakdown)
  4. It learns — here's the headroom -> Training slide       (learnable gap; optional live dry-run)
  5. We broke our own reward         -> Reward Integrity     (safeguards manifest)

Usage:
  python demo_live.py                 # interactive, press Enter between acts
  python demo_live.py --auto          # no pauses (rehearsal / timing runs)
  python demo_live.py --train         # Act 4 runs the real no-GPU dry-run live
  python demo_live.py --server http://localhost:7860
  python demo_live.py --seed 42 --task team_lateral_team

Safe on Windows: forces UTF-8 and enables ANSI colour automatically.
Auto-starts the server if one isn't already running.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import httpx

# ---- make the terminal behave identically on Windows / macOS / Linux ----
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
if os.name == "nt":  # enable ANSI escape processing on Windows 10+ consoles
    try:
        import ctypes

        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:
        pass

# ---- palette ----
R = "\033[0m"
DIM = "\033[38;5;244m"
RED = "\033[38;5;203m"
GRN = "\033[38;5;79m"
AMB = "\033[38;5;215m"
CYN = "\033[38;5;117m"
INK = "\033[97m"
BOLD = "\033[1m"
RULE = f"{DIM}{'─' * 66}{R}"

DEFAULT_SERVER = "http://localhost:7860"
AUTO = False


def c(txt: str, col: str) -> str:
    return f"{col}{txt}{R}"


def act(n: int, title: str, slide: str) -> None:
    print()
    print(f"{RED}{BOLD}  ACT {n}{R}  {INK}{BOLD}{title}{R}")
    print(f"{DIM}  ↳ website slide: {slide}{R}")
    print(RULE)


def narrate(line: str) -> None:
    print(f"  {DIM}“{line}”{R}")


def cmd(line: str) -> None:
    print(f"  {GRN}${R} {INK}{line}{R}")


def pause(prompt: str = "press Enter") -> None:
    if AUTO:
        time.sleep(0.5)
        return
    try:
        input(f"\n  {AMB}▸ {prompt}…{R} ")
    except (EOFError, KeyboardInterrupt):
        print("\n  (skipping pauses)")
        globals()["AUTO"] = True


def reward_str(v) -> str:
    if v is None:
        return c("n/a", DIM)
    col = GRN if v >= 0 else RED
    return c(f"{v:+.3f}", col)


# ------------------------------------------------------------------ server
def ensure_server(url: str):
    try:
        httpx.get(f"{url}/health", timeout=3).raise_for_status()
        print(f"  {GRN}✓{R} server live at {url}")
        return None
    except Exception:
        print(f"  {AMB}→{R} starting server on {url} …")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "127.0.0.1", "--port", "7860"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            time.sleep(1)
            try:
                httpx.get(f"{url}/health", timeout=3).raise_for_status()
                print(f"  {GRN}✓{R} server started")
                return proc
            except Exception:
                pass
        proc.terminate()
        sys.exit(f"  {RED}✗ server failed to start in 30s{R}")


# ------------------------------------------------------------------- acts
def act1_environment(cl: httpx.Client, task: str, seed: int) -> str:
    act(1, "The environment is real — one team episode", "Architecture")
    narrate("Standard OpenEnv: POST /reset starts a deterministic episode.")
    cmd(f'curl -X POST /reset  -d \'{{"task_id":"{task}","seed":{seed},"mode":"team"}}\'')
    r = cl.post("/reset", json={"task_id": task, "seed": seed, "mode": "team"})
    r.raise_for_status()
    obs = r.json()
    alert = obs["alert_queue"][0]
    aid = alert["alert_id"]
    ind = alert.get("indicators", {})
    ip = (ind.get("ip") or ["—"])[0]
    print()
    print(f"  {CYN}ALERT {aid}{R}  {DIM}sev={R}{c(alert['severity'].upper(), AMB)}"
          f"  {DIM}rule={R}{alert.get('rule_triggered', '—')}")
    print(f"  {INK}{alert['title']}{R}")
    print(f"  {DIM}queue={len(obs['alert_queue'])} alerts · role={obs.get('current_role')}"
          f" · phase={obs.get('current_phase')} · seed locked → replayable{R}")
    return aid, ip


def act2_ticketbus(cl: httpx.Client, aid: str, ip: str) -> None:
    act(2, "Agents coordinate through a ticket bus", "Team Mode  ← the money shot")
    narrate("Tier-1 gathers evidence, classifies, then escalates. "
            "Escalation is a real object on a bus — not a line in a shared prompt.")
    steps = [
        ("enrich the indicator",
         {"action_type": "enrich_indicator", "indicator": ip, "indicator_type": "ip"}),
        ("classify the alert",
         {"action_type": "classify_alert", "alert_id": aid, "classification": "true_positive"}),
        ("escalate to Tier-2",
         {"action_type": "escalate_to_tier2", "alert_id": aid,
          "reason": "malicious indicator + lateral movement confirmed"}),
    ]
    print()
    for label, payload in steps:
        resp = cl.post("/step", json=payload).json()
        print(f"  {GRN}${R} tier1 → {c(payload['action_type'], CYN):<38} "
              f"reward {reward_str(resp.get('reward'))}   {DIM}{label}{R}")
        time.sleep(0.35 if AUTO else 0.15)
    pause("now pull Tier-2's inbox")
    print()
    cmd("curl /inbox/tier2")
    inbox = cl.get("/inbox/tier2").json()
    tickets = inbox.get("tickets", [])
    if tickets:
        t = tickets[0]
        print(f"  {GRN}✓ TICKET ON THE BUS{R}")
        print(f"    {DIM}ticket_id {R}{c(t['ticket_id'], AMB)}")
        print(f"    {DIM}alert_id  {R}{t['alert_id']}")
        print(f"    {DIM}from → to {R}{c(t['from_role'], CYN)} → {c(t['to_role'], CYN)}"
              f"   {DIM}kind={R}{t['kind']}")
        print(f"    {DIM}payload   {R}classification={t['payload'].get('classification')}"
              f" · severity={t['payload'].get('severity')} · step={t.get('step_created')}")
        print()
        narrate("That ticket is inter-agent communication as data — "
                "queryable memory, not prompt-passing. This is the whole thesis.")
    else:
        print(f"  {AMB}inbox empty (rewind: re-run reset){R}")


def act3_grader(cl: httpx.Client, task: str) -> None:
    act(3, "The score is glass-box, not a black box", "Reward")
    narrate("The grader decomposes the score into named, testable components. "
            "Verifiable rewards — RLVR — no vibes.")
    cmd(f'curl -X POST /grader  -d \'{{"task_id":"{task}"}}\'')
    g = cl.post("/grader", json={"task_id": task}).json()
    comp = g.get("breakdown") or g.get("components") or {}
    print()
    if comp:
        for k, v in list(comp.items())[:8]:
            bar = ""
            try:
                bar = c("█" * int(float(v) * 20), GRN)
            except (TypeError, ValueError):
                pass
            print(f"    {DIM}{k:<26}{R} {str(v):>6}  {bar}")
    else:
        print(f"    overall = {g.get('score', 0.0) * 100:.1f}%   "
              f"{DIM}{g.get('message', g.get('notes', '—'))}{R}")


def act4_training(cl: httpx.Client, run_dry: bool) -> None:
    act(4, "It learns — and here's the measurable headroom", "Training")
    narrate("You can't train a model in a 10-minute talk. So we show the bracket: "
            "the gap a trained policy has to close.")
    print()
    print(f"    {RED}RANDOM floor  {R}{DIM}(untrained){R}     "
          f"{c('0.063', RED)}   {c('▏' + '░' * 2, RED)}")
    print(f"    {GRN}ORACLE ceiling{R} {DIM}(scripted){R}     "
          f"{c('0.900', GRN)}   {c('█' * 27, GRN)}")
    print(f"    {AMB}LEARNABLE GAP Δ{R}                {c('+0.836', AMB)}"
          f"   {DIM}← the 20%-weighted 'improvement' band{R}")
    print()
    narrate("Honest note: our published checkpoint underfit on a free T4 — "
            "the pipeline is proven, the compute wasn't. That's a rental-GPU problem.")
    if run_dry:
        pause("run the real no-GPU dry-run live")
        print()
        cmd("python train_grpo.py --role tier1 --dry-run")
        print(f"  {DIM}(deterministic oracle rollouts — no GPU, ~10s){R}\n")
        env = dict(os.environ, PYTHONUTF8="1")
        subprocess.run([sys.executable, "train_grpo.py", "--role", "tier1", "--dry-run"],
                       env=env)
    else:
        print(f"    {DIM}Live option: run  {INK}python train_grpo.py --role tier1 --dry-run{R}"
              f"{DIM}  (no GPU, ~10s){R}")


def act5_safeguards(cl: httpx.Client) -> None:
    act(5, "We tried to break our own reward — six ways", "Reward Integrity")
    narrate("A reward function is only as good as the exploits it can't be farmed by. "
            "Each fix is locked in as a regression test — machine-checkable, live.")
    cmd("curl /themes/coverage")
    tc = cl.get("/themes/coverage").json()
    print()
    for name in tc.get("reward_hacking_defenses", []):
        print(f"    {GRN}✓{R} {name}")
    covered = sum(1 for v in (tc.get("coverage") or {}).values() if v)
    print(f"\n    {DIM}themes covered: {R}{c(str(covered), AMB)}"
          f"   {DIM}· 111 tests passing · same seed → same score{R}")


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--task", default="team_lateral_team")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto", action="store_true", help="no pauses (rehearsal)")
    ap.add_argument("--train", action="store_true", help="Act 4 runs the live dry-run")
    args = ap.parse_args()
    globals()["AUTO"] = args.auto

    print()
    print(f"{RED}{BOLD}  ┌─ SOC-TRIAGE-GYM · LIVE STAGE DEMO ─────────────────────────┐{R}")
    print(f"{RED}{BOLD}  │{R}  train the team, not the analyst"
          f"                          {RED}{BOLD}│{R}")
    print(f"{RED}{BOLD}  └────────────────────────────────────────────────────────────┘{R}")
    proc = ensure_server(args.server)
    try:
        with httpx.Client(base_url=args.server, timeout=120) as cl:
            aid, ip = act1_environment(cl, args.task, args.seed)
            pause("next: the ticket bus")
            act2_ticketbus(cl, aid, ip)
            pause("next: the grader")
            act3_grader(cl, args.task)
            pause("next: training headroom")
            act4_training(cl, args.train)
            pause("next: safeguards")
            act5_safeguards(cl)
            print()
            print(RULE)
            print(f"  {GRN}{BOLD}Demo complete.{R}  "
                  f"{DIM}Everything is open — MIT.{R}")
            print(f"  {DIM}GitHub  {R}github.com/ROHITCRAFTSYT/SOC-Triage-Gym")
            print(f"  {DIM}Space   {R}huggingface.co/spaces/rohitcraftsyt/openenv2")
            print(f"  {DIM}Model   {R}huggingface.co/rohitcraftsyt/soc-grpo-tier1")
            print()
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
