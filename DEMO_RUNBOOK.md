# SOC-Triage-Gym — Live Demo Runbook

How to actually show the thing on stage. Everything here is **tested** on this
machine (Windows, Python 3.11). Two commands drive the whole demo.

---

## TL;DR — the answers to your three questions

1. **How do I show the real demo?**
   Run `python demo_live.py` in one terminal while the website is your slides in
   the browser. It's presenter-paced — you press Enter between acts, and each act
   maps to a website section. It hits a live server, real endpoints, real rewards.

2. **Do I show the training process?**
   **No — you never train live.** Real training is 70–90 min on a T4; you can't do
   that in 10 minutes and it would underfit anyway. Instead you show **three things
   that take seconds**: (a) the *learnable-gap* bracket — random 0.063 vs oracle
   0.900 — which is the honest, impressive result; (b) optionally the **no-GPU
   dry-run** (`--train`) that prints a real oracle reward curve in ~10s; (c) the
   committed loss/curve PNGs already in the website's drag-gallery. You *explain*
   the per-step GRPO mechanism; you *don't* run the GPU job.

3. **The remade demo** is `demo_live.py` (new) + a bug-fixed `demo.py`. See below.

---

## What I fixed / built (so it doesn't break on stage)

| Item | Status |
|---|---|
| `demo.py` crashed on Windows (cp1252 can't encode `✓ → ①`) | **FIXED** — forces UTF-8 |
| Curl payload shape in old plan was wrong (`{"action":{...}}`) | **CORRECTED** — API wants flat `{"action_type":...}` |
| No presenter-paced live demo existed | **BUILT** — `demo_live.py`, Enter-to-advance, colorized, slide-mapped |
| Beat-3 "trained = 0.1%" footgun (looks like failure) | **SIDESTEPPED** — `demo_live.py` shows the learnable *gap*, not a misleading delta |

---

## Setup (do this at T-15 min, once)

You need **two terminals** and **one browser**.

**Terminal A — the environment server (leave running the whole talk):**
```powershell
cd D:\SOC-Triage-Gym
uvicorn server.app:app --host 127.0.0.1 --port 7860
```
Wait for `Application startup complete`. Don't touch this window again.

**Browser — your slides:**
```powershell
python -m http.server 8765 --directory site
```
Open `http://localhost:8765`, press **F11** (fullscreen).

**Terminal B — where you'll run the demo.** Make the font **big (≥ 20 pt)** and
use **Windows Terminal** or the **VS Code terminal** (both render the colors).
Dry-run it once now so you've seen it:
```powershell
cd D:\SOC-Triage-Gym
python demo_live.py --auto        # rehearsal: no pauses, just watch it fly by
```

> If `uvicorn`/`httpx` aren't installed yet: `pip install -e ".[dev]"` (once).
> `demo_live.py` will auto-start a server if Terminal A isn't up — but starting it
> yourself is safer (no 30 s cold-start eating your slot).

---

## The demo, mapped to your slides

Run this in Terminal B when you reach the demo portion of the talk:
```powershell
python demo_live.py
```
Press **Enter** to advance. Each act lines up with a section you scroll to in the
browser. Suggested flow: **scroll the website to the matching slide first, then
run the act.**

| Act | Terminal shows | Scroll website to | Your line |
|---|---|---|---|
| **1 · Environment is real** | `POST /reset` → a real HIGH-sev alert, 8-alert queue, seed locked | **Architecture** | "Standard OpenEnv — reset gives a deterministic episode." |
| **2 · Ticket bus** ⭐ | Tier-1 enrich `+0.072` → classify `+0.180` → escalate `+0.036`, then the **ticket appears in Tier-2's inbox** | **Team Mode** | "Escalation is a real object on a bus — inter-agent comms as *data*, not prompt-passing. This is the whole thesis." |
| **3 · Glass-box score** | `POST /grader` → per-component reward breakdown with bars | **Reward** | "The score decomposes into named, testable checks. RLVR — no vibes." |
| **4 · Learnable gap** | random `0.063` vs oracle `0.900`, Δ `+0.836`; honest underfit note | **Training** | "You can't train in 10 minutes — here's the *gap* a trained policy closes." |
| **5 · Broke our own reward** | `GET /themes/coverage` → 6 named defenses, live | **Reward Integrity** | "A reward is only as good as the exploits it can't be farmed by. Each fix is a regression test." |

**Act 2 is the money shot.** It's the one thing no single-agent RL demo can show.
Slow down there. Let the ticket object sit on screen while you say the thesis line.

---

## If you want to literally show "training" running

Only if you have spare time (it adds ~15s). Either:

**Inside the demo:**
```powershell
python demo_live.py --train
```
Act 4 will run the real dry-run live.

**Or standalone, in Terminal B:**
```powershell
python train_grpo.py --role tier1 --dry-run
```
This prints a deterministic **oracle reward curve** across tier-1 tasks and seeds
(avg **0.8995**) and saves `reward_curve_tier1_oracle.png` — **no GPU, ~10s**. Say:
*"This is the target line. Real GRPO training is a 90-minute T4 job that produces
the checkpoint in the gallery — the pipeline's proven, the compute's rented."*

**Never** run `python train_grpo.py` without `--dry-run` on stage — that's the
real GPU job.

---

## Rehearsal / timing

```powershell
python demo_live.py --auto          # full run, no pauses — time it end to end
```
Auto mode runs ~20–25s. With narration and pauses, budget **~2.5 min** for the
five acts. That fits the 4:30–7:00 demo window in TALK_PLAN.md.

---

## Failure playbook (memorize the first row)

| If… | Do this |
|---|---|
| A command errors / server hiccups | Say *"it's deterministic — here's the same run"* and scroll the website to the **Live Wire** section — the terminal animation there replays the exact episode. Keep talking. Never debug live. |
| Colors show as `[38;5…` garbage | You're in a terminal without ANSI. Switch to **Windows Terminal** or **VS Code terminal**. (The demo already enables ANSI on Win10+; this only bites old consoles.) |
| Server won't start (port busy) | `python demo_live.py --server http://localhost:7860` will reuse it; or change the port on both uvicorn and the flag. |
| Whole laptop dies | Open the HF Space from any machine: `huggingface.co/spaces/rohitcraftsyt/openenv2`, narrate from TALK_PLAN.md §3. |

---

## Exact commands, copy-paste block (for your staging file)

```powershell
# ── T-15: setup ──────────────────────────────────────────────
# Terminal A (leave running):
uvicorn server.app:app --host 127.0.0.1 --port 7860
# Browser slides:
python -m http.server 8765 --directory site   # -> http://localhost:8765  (F11)

# ── rehearsal ────────────────────────────────────────────────
python demo_live.py --auto

# ── ON STAGE (Terminal B) ────────────────────────────────────
python demo_live.py            # Enter between acts; scroll website to match
# optional live training moment:
python demo_live.py --train
```

That's the whole demo. Two commands, five acts, every number real and
reproducible on the same seed.
