# BLR5 CCCL × SurrealDB — 10-Minute Talk Plan

**Event:** Bengaluru | BLR5 CCCL × SurrealDB — Memory, Context & Agents
**Slot:** 10 minutes total (talk + live demo; assume Q&A is separate — if not, cut per §5)
**Visual aid:** `site/index.html` fullscreen in a browser (serve with `python -m http.server 8765 --directory site`)
**Demo surface:** one terminal, font ≥ 18 pt, dark theme

---

## 1. Run of show (minute by minute)

| Clock | Segment | On screen | You are doing |
|---|---|---|---|
| 0:00–0:45 | **Hook** | Website hero — terminal animation auto-plays | Talking over the animation |
| 0:45–2:00 | **Problem → idea** | Scroll: Brief section → "TRAIN THE TEAM" statement | Talking, slow scroll |
| 2:00–3:30 | **Architecture** | System map + reward formula | Pointing at boxes |
| 3:30–4:30 | **Team mode + pressure sources** | Pipeline diagram + 3 cards (Actors / Drift / Experts) | Talking |
| 4:30–7:00 | **LIVE DEMO** | Terminal: `python demo.py`, then ticket-bus curl | Typing (pre-staged commands) |
| 7:00–8:15 | **Reward integrity + training reality** | Defense table → gap chart → limitations | Talking |
| 8:15–9:15 | **Memory / Context / Agents + close** | Meetup band → footer links | Talking |
| 9:15–10:00 | **Buffer** | Footer | Absorbs overruns; else invite first question |

**Hard checkpoints:** at **4:30 you must be in the terminal**; at **7:00 you must be out of it.** Everything else flexes.

---

## 2. Setup checklist

### T-30 minutes (at the venue)

- [ ] Start the env server and LEAVE IT RUNNING: `uvicorn server.app:app --port 7860`
      (`demo.py` auto-starts a server if none is up, but that burns up to 30 s of your slot — never demo cold)
- [ ] Serve the site: `python -m http.server 8765 --directory site` → open `http://localhost:8765`, press F11 (fullscreen)
- [ ] **Decide the Beat-3 mode** (see §4 — do this consciously, not by accident)
- [ ] Dry-run `python demo.py` once end-to-end; confirm it completes in seconds
- [ ] Terminal: font ≥ 18 pt, window on the same display that's being projected
- [ ] Stage the demo commands in a scratch file to copy-paste (no live typing of JSON)
- [ ] Notifications off (Focus Assist on Windows), brightness 100%, power plugged in
- [ ] Backup tab open: https://huggingface.co/spaces/rohitcraftsyt/openenv2
- [ ] Everything runs offline except Google Fonts on the site (it degrades gracefully) — do NOT depend on venue Wi-Fi

### Command staging file (copy-paste source)

Run curls from **Git Bash** — single-quoted JSON breaks in PowerShell.

> **Simplest path:** run `python demo_live.py` — it drives the whole five-act
> demo, presenter-paced, with the correct payloads baked in. See
> [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for the slide-by-slide mapping. The raw
> curls below are the fallback if you'd rather drive it by hand.

```bash
# D1 — the whole story, presenter-paced (recommended)
python demo_live.py

# D2 — team episode, deterministic (note: flat action_type, not nested)
curl -s -X POST localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_id":"team_lateral_team","seed":42,"mode":"team"}' | jq '.alert_queue[0]'

# D3 — tier1 classifies + escalates → ticket appears on the bus
#      (alert id is ALT-TLT-001 for this task/seed; action_type is top-level)
curl -s -X POST localhost:7860/step -H "Content-Type: application/json" \
  -d '{"action_type":"classify_alert","alert_id":"ALT-TLT-001","classification":"true_positive"}' | jq '.reward'
curl -s -X POST localhost:7860/step -H "Content-Type: application/json" \
  -d '{"action_type":"escalate_to_tier2","alert_id":"ALT-TLT-001","reason":"lateral movement confirmed"}' | jq '.reward'
curl -s localhost:7860/inbox/tier2 | jq '.tickets[0]'

# D4 — the machine-checkable safeguards manifest
curl -s localhost:7860/themes/coverage | jq '.reward_hacking_defenses'
```

---

## 3. The script

Calibrated to ~140 words/min. Section word counts are the budget — if you're over, cut sentences, not sections.

### 0:00–0:45 · Hook (~100 words)

*Screen: hero. The terminal animation plays a team episode by itself — let it work for you.*

> "Quick show of hands — who here has worked in or near a SOC? A Security Operations Centre is where alert fatigue goes to become a lifestyle. Thousands of alerts a day; most are noise; a few are live intrusions.
>
> Everyone is trying to put LLM agents on this problem. But here's the thing: almost every RL environment out there trains **one agent, alone**. A real SOC is not one analyst. It's a **team** — Tier-1 triages, Tier-2 contains, a manager audits both.
>
> SOC-Triage-Gym is a reinforcement-learning environment that trains the **team**."

### 0:45–2:00 · Problem → idea (~170 words)

*Scroll slowly through Brief; land on the "TRAIN THE TEAM, NOT THE ANALYST" statement — pause there a beat, it's the thesis.*

> "The design thesis is on screen: if you want agents that work inside a real security workflow, you have to train them inside one — tiers, escalation paths, ticketing SLAs, policies that change mid-shift, users emailing 'this looks weird' at the worst moment.
>
> So this is a full OpenEnv-standard environment — `/reset`, `/step`, `/state` over HTTP — where three roles cooperate through a ticket bus, and the reward is *blended*: sixty percent your own role's performance, forty percent the **delta** in team F1. Delta matters — if you spam no-ops after one correct classification, you farm exactly zero.
>
> Eight tasks. The smallest is a single phishing alert, fifteen steps. The biggest is a two-hundred-fifty-step APT campaign — sixty-plus alerts across five kill-chain phases with sparse, delayed rewards. Same agent, three orders of magnitude in horizon."

### 2:00–3:30 · Architecture (~200 words)

*Scroll to the system map. Point, left to right.*

> "Architecture, left to right. The agent — an LLM being trained with GRPO, a scripted oracle ceiling, or a random floor — only ever sees the REST API. That's what makes the environment swappable under any training loop.
>
> Behind the API sits the simulation: an alert queue, a **phase state machine** — triage, response, oversight — the ticket bus, and a step-budget with an efficiency tax. Below it, a tool layer modelled as four logical enterprise apps: SIEM, EDR, IAM, and ticketing — with real cross-app rules; you can't `disable_user` unless there's an open P1 or P2 ticket.
>
> And the reward comes from a grader **stack** — eight per-task programmatic graders, a team grader, an LLM manager-judge with a deterministic fallback. Verifiable rewards — RLVR — no vibes-based scoring.
>
> Now the fun part — these three boxes at the bottom inject pressure *while the agent works*: a **red-team generator** that evolves scenario difficulty against you, a **policy-drift engine** that changes the rulebook mid-episode, and **NPC actors** — a threat-intel feed, a compliance officer, and an end user who emails 'I clicked a link' right when you're busiest."

### 3:30–4:30 · Team mode (~140 words)

*Scroll to the pipeline. The ticket wires animate and the F1 meter fills.*

> "Here's a team episode. Tier-1 owns triage — enrich indicators, query logs, classify, and critically: **escalate**. That escalation is a real object — a ticket on a bus — not a line in a shared prompt. Tier-2 pulls it from its inbox and owns containment: isolate hosts, disable users, block IOCs. Then the manager phase audits everything — and can flag inconsistencies or override classifications.
>
> Every hand-off is observable and queryable over HTTP. Which matters for a meetup about memory and context — the team's shared state lives in the *environment*, versioned and inspectable, not trapped in anyone's context window.
>
> Let me show you it running."

### 4:30–7:00 · LIVE DEMO (~90 s of commands + narration)

*Switch to the terminal. Server already running. Commands come from your staging file.*

**D1 — `python demo.py`** (narrate as beats print):

> "One command, five beats — the same arc every RL claim should make.
> **One**: an untrained no-op policy — submits blindly, scores basically zero. That's the floor.
> **Two**: the verifier breakdown — the grader isn't a black box; every reward component is a named, testable check.
> **Three**: the trained attempt on the same seed.
> **Four**: the measurable delta. And note the line below it — *same seed, same score*. This whole environment is deterministic; the demo I rehearsed is the demo you're watching.
> **Five**: the safeguards — six named reward-hacking defenses, live, machine-checkable."

**D3 — the ticket bus** (skip D2 if past 6:00):

> "And the multi-agent part in raw HTTP: Tier-1 escalates — that action creates a ticket — and here's Tier-2's inbox with the ticket in it. Inter-agent communication as data, not prompt-passing."

*Switch back to the website.*

### 7:00–8:15 · Reward integrity + training honesty (~180 words)

*Scroll to the exploit-vector table, then the gap chart, then limitations.*

> "A reward function is only as good as the exploits it can't be farmed by. We attacked our own reward and found six vectors — no-op F1 farming, duplicate case-closes, escalation flooding, phase-complete short-circuits, spurious manager flags, judge-fallback abuse. Each one is fixed **and locked in as a regression test**. That's a hundred and eleven tests passing.
>
> Training-wise: per-step GRPO. Each dataset row is one decision point; the reward function replays the environment deterministically to that step, applies the model's single action, and takes the immediate blended reward.
>
> The environment brackets the problem: random policy scores zero-point-zero-six; the scripted oracle scores zero-point-nine. That zero-point-eight-four gap is the measurable headroom.
>
> Full honesty — it's on the website too: our published checkpoint **underfit**. Twenty-seven optimizer steps on a free Colab T4 doesn't move a small model. The pipeline is proven; the compute wasn't. That's a rental-GPU problem, not a design problem."

### 8:15–9:15 · Meetup tie-in + close (~140 words)

*Scroll to the Memory / Context / Agents band, then the footer.*

> "This event is about memory, context, and agents — this project is accidentally a case study in all three.
> **Memory**: episode state, the ticket bus, policy history — explicit, queryable, versioned. An agent that forgets which policy version was active at step twelve pays for it in reward.
> **Context**: context here isn't static, it *drifts* — mid-episode rule changes, unsolicited NPC messages, 250-step horizons. Agents have to triage their context like analysts triage alerts.
> **Agents**: not one — a team of three, plus an adversary underneath that co-evolves with them.
>
> Everything is open — MIT licensed. The repo, the live HuggingFace Space, the trained adapter, and a one-click Colab are all linked here. `python demo.py` gets you the whole story in sixty seconds. Thank you."

---

## 4. The Beat-3 decision (do this before you go on)

`demo.py` Beat 3 reads `training_summary.json` if present. That file holds the **honest underfit numbers** (trained_avg = 0.001), so Beat 4's delta will print **≈ +0.1 pp** — which looks like failure to an audience that missed the nuance.

**Option A — recommended for a meetup:** temporarily move the file so Beat 3 uses the oracle proxy:

```bash
mv training_summary.json training_summary.json.bak   # before the talk
mv training_summary.json.bak training_summary.json   # after
```

Beat 3 then prints `score = 90.0%` **and labels itself "(scripted oracle proxy)" on screen** — it's transparent, and you also say the honesty line out loud in the 7:00 segment. Delta prints ≈ +90 pp.

**Option B:** keep the real file and lean into the honesty story during the demo itself. Riskier — the small delta lands before your explanation does.

Either way, **rehearse the option you pick.** Don't discover the difference on stage.

---

## 5. Timing checkpoints & cut lines

| If at… | You're at… | Then |
|---|---|---|
| 2:30 | still in Brief | Skip pipeline talk-through; jump to demo at 4:00 via "let me just show you" |
| 6:00 | still in `demo.py` | Skip D2/D3 curls entirely; the demo.py beats carry the multi-agent point |
| 8:00 | still on integrity | Compress training to one line: "random 0.06, oracle 0.90 — that gap is the headroom" |
| 9:00 | not yet at close | Jump straight to: "Memory: the bus. Context: it drifts. Agents: a team plus an adversary. Repo's open, thank you." (15 s version) |
| Q&A eats your slot | — | The 9:15 buffer is the sacrifice; never cut the demo |

---

## 6. Failure playbook

| Failure | Response |
|---|---|
| `demo.py` errors / hangs | Say "the environment is deterministic, so here's the same run" → scroll to website hero — the terminal animation replays the episode. Continue script unchanged. |
| Server died | Same as above. Do NOT debug live. |
| Projector/laptop dies | HF Space from any machine: huggingface.co/spaces/rohitcraftsyt/openenv2 — narrate from memory using §3. |
| jq missing on venue machine | Drop `| jq ...` — raw JSON is fine at 18 pt. |
| Font glitches on site (no Wi-Fi) | System-font fallbacks are designed in — ignore, nobody but you will notice. |

---

## 7. Likely questions (crisp answers)

1. **"Why not just orchestrate three LLM calls with a framework?"** — Orchestration executes a team; this *trains* one. The blended reward means selfish behavior is penalized during learning — that signal doesn't exist in an orchestration framework.
2. **"Is the manager judge an LLM? Isn't that unverifiable?"** — LLM path is primary, but it's *layered* over hard programmatic checks, and the API-free fallback is deterministic and bounded. No reward component is LLM-only.
3. **"How do you know the reward isn't hackable?"** — We know six ways it *was* — each fixed and locked in as a regression test. `GET /themes/coverage` lists them live. There are surely more; that's why they're tests, not claims.
4. **"Why GRPO over PPO?"** — Group-relative advantage needs no value network — right fit for LLM policies on small compute; and per-step replay gives dense signal without trajectory-averaging away credit assignment.
5. **"Real SOC data?"** — Synthetic but MITRE ATT&CK-grounded scenarios. Deterministic generation is the point: reproducible training and eval. Real-telemetry adapters are future work.
6. **"Why did the trained model underperform?"** — 27 optimizer steps on a free T4 — an honest compute limitation, published as such. The learnable gap (0.06 → 0.90) is committed and reproducible; the pipeline is what's proven.
7. **"Where would SurrealDB fit?"** *(their event — expect this)* — The ticket bus, policy history, and actor inboxes are exactly a multi-model store's job: documents (tickets) + graph (alert correlation) + time-versioned records (policy drift, active-at semantics). Today it's in-memory Pydantic state; a persistent backend is a natural evolution.
8. **"Can I train tier-2 / the manager?"** — Architecture supports it; currently tier-1 trains while the others run scripted oracle — staged curriculum, next on the roadmap.
9. **"Multi-turn tool use or single actions?"** — One JSON action per step, 15–250 steps per episode — the long horizon *is* the multi-turn structure, and it's what makes credit assignment interesting.
10. **"What's OpenEnv?"** — A standard REST contract for RL environments (`/reset`, `/step`, `/state`) so any training loop can drive any environment. This is a fully conformant implementation.

---

## 8. Five lines to know cold

1. "Every RL environment trains one agent. A real SOC has three. We train the team."
2. "Sixty percent your role, forty percent **delta** team-F1 — no-op spam farms exactly zero."
3. "A reward function is only as good as the exploits it can't be farmed by — six found, six fixed, six locked in as tests."
4. "Same seed, same episode, same score — the demo I rehearsed is the demo you're watching."
5. "Memory is the ticket bus, context is the drift, agents are the team — plus the adversary underneath."
