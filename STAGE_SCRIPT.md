# STAGE SCRIPT: 10 Minutes, Word for Word

**Event:** Bengaluru | CCCL BLR6 — Securing AI Agents (Claude Community Events)
**When/where:** Sun 12 Jul, 6:00–8:30 pm · WeWork Roshni Tech Hub, Marathahalli
**Assets:** Website fullscreen (`http://localhost:8765`), Terminal B (`python demo_live.py`)
**Speaking pace:** ~140 words/min. Every segment's word count is budgeted. If you're
over, cut sentences. Never cut the demo.

---

## Why this hook works

You asked for a Durov-style opening: a real story before the pitch. Here it is.
The **2013 Target breach** is the most famous security failure in history, and the
part almost nobody remembers is that **the alerts were caught by Target's monitoring
team in Bengaluru**, the same city your audience is sitting in. The Bengaluru team
escalated to Minneapolis headquarters, and headquarters dismissed it. The breach ran
on. Forty million cards.

It's not a detection failure. It's a **handoff failure**, which is *literally the
thing your project trains*. The ticket bus in your demo is the fixed version of the
exact thing that broke. You open with the story, and you close by calling back to it.
A tech crowd in Bengaluru will feel this one personally.

(These are widely documented facts: Bloomberg Businessweek's 2014 investigation and
congressional testimony. You're recounting history, not making claims.)

---

## Pre-flight (T-15, from DEMO_RUNBOOK.md)

```powershell
# Terminal A, leave running:
uvicorn server.app:app --host 127.0.0.1 --port 7860
# Browser slides:
python -m http.server 8765 --directory site     # open http://localhost:8765, F11
# Terminal B, rehearse once:
python demo_live.py --auto
```
Font ≥ 20pt in Terminal B. Notifications off. Power plugged in.
Website open at the **hero**. Terminal B ready with `python demo_live.py` typed
but NOT yet run.

---

# THE SCRIPT

## 0:00 – 1:20 · THE HOOK (~185 words)

*Screen: website hero. The terminal animation plays behind you. Walk to center
stage. Do NOT introduce yourself yet. Let the first line land in silence.*

> Before I tell you what I built, let me tell you about the most expensive ignored
> notification in history.
>
> November 2013. Attackers are inside Target, the American retail giant, quietly
> stealing forty million credit cards. Target had spent millions on security. And
> here's the thing: **the tools worked.** Alerts fired.
>
> Now, the part of this story nobody remembers. The team that caught those alerts
> was sitting **in this city.** Target's monitoring centre in Bengaluru saw the
> malware, flagged it, and escalated to headquarters in Minneapolis.
>
> *(pause, two full seconds)*
>
> And Minneapolis closed the ticket. Called it noise. The breach ran for two more
> weeks. It cost hundreds of millions of dollars, and eventually, the CEO's job.
>
> So that was **not** a detection failure. Detection worked. In Bengaluru.
> It was a **handoff** failure. A *team* failure.
>
> And here's what keeps me up at night: everyone in this room is trying to fix
> security with AI agents, and almost every RL environment we train them in has
> **one agent, working alone, with nobody to hand off to.**

## 1:20 – 1:45 · SELF-INTRO (~55 words)

*Now you earn the right to say your name, after they care.*

> I'm Rohit. I build reinforcement-learning environments, and for the last few
> months I've been building the one I wished existed: **SOC-Triage-Gym**, the
> first OpenEnv environment that trains AI agents not as a lone analyst, but as
> a **coordinated SOC team.** Tonight I'll show it to you live.

*(If you have one more credibility line, hackathon, employer, open-source work,
insert ONE sentence here. Not two.)*

## 1:45 – 2:40 · THE THESIS (~130 words)

*Scroll slowly through the Brief section; land on the giant "TRAIN THE TEAM, NOT
THE ANALYST" statement. Stop there. Point at it.*

> A real SOC runs on three roles. Tier-1 triages alerts. Tier-2 contains confirmed
> threats. And a manager audits both. Between them: tickets, escalation paths, SLA
> clocks, policies that change mid-shift, and users who email "I clicked a link"
> at the worst possible moment.
>
> So this environment models **all of it.** Standard OpenEnv REST, reset, step,
> state, and inside: eight tasks, from a single phishing alert, fifteen steps,
> up to a **two-hundred-and-fifty-step APT campaign** with sixty-plus alerts
> across the full kill chain.
>
> This sentence on screen is the whole project. If one agent maximizes its own
> score at the team's expense, **the reward punishes it.**

## 2:40 – 3:50 · ARCHITECTURE (~165 words)

*Scroll to the system map. Point left to right. Numbers slowly.*

> Left to right. The agent, an LLM under GRPO training, or a scripted oracle, or
> a random-floor policy, only ever sees the REST API. Which means any training
> loop can drive this environment.
>
> Behind the API: an alert queue, a **phase state machine**, triage, response,
> oversight, a ticket bus, and a step budget with an efficiency tax. The tool
> layer is four enterprise apps: SIEM, EDR, IAM, ticketing, with real cross-app
> rules. You cannot disable a user without an open P1 or P2 ticket. Sound familiar?
>
> Rewards come from a grader **stack**: eight programmatic graders, a team
> grader, an LLM judge with a deterministic fallback. Verifiable rewards. No
> vibes.
>
> And these three boxes at the bottom inject pressure *while the agent works*:
> a red-team generator that evolves difficulty against you, a policy-drift engine
> that rewrites the rulebook mid-episode, and NPC actors, including an end user
> who emails "I clicked a link" right when the queue is deepest.
>
> The reward blend is on screen: sixty percent your role, forty percent **delta**
> team-F1. Spam no-ops after one correct call, you farm exactly zero.

## 3:50 – 4:20 · TEAM MODE (~70 words)

*Scroll to the pipeline diagram, the animated ticket wires.*

> Here's a team episode. Tier-1 investigates and escalates, and that escalation
> is a **real object on a bus**, not a line in a shared prompt. Tier-2 pulls it
> from an inbox and contains. The manager audits everything, and can override.
>
> Every handoff is observable over HTTP.
>
> *(turn to audience)* Remember Minneapolis? Let me show you the handoff, live.

## 4:20 – 6:50 · THE LIVE DEMO

*Walk to Terminal B. Press Enter to run `python demo_live.py`. You control the
pace; each act waits for your Enter. Narrate over the output; these lines are
short because the terminal is doing the talking.*

**ACT 1: reset** *(website was on Team Mode; terminal shows a real alert)*
> One POST to /reset. A real high-severity phishing alert, eight alerts in queue,
> seed forty-two. Deterministic: same seed, same episode, every single time.
> The demo I rehearsed is the demo you're watching.

**ACT 2: the ticket bus** *(the money shot, SLOW DOWN)*
> Tier-1 enriches the indicator, plus zero-point-zero-seven. Classifies true
> positive, plus zero-point-one-eight. Escalates.
>
> *(press Enter for the inbox, pause while it prints)*
>
> And there it is. A ticket. In Tier-2's inbox. With the classification, the
> severity, the step it was created. **This** is the handoff that failed in 2013,
> except here it's data on a bus, it's queryable, and the reward function grades
> it. Inter-agent communication as memory, not prompt-passing.

**ACT 3: the grader**
> The score isn't a black box. The grader decomposes it into named components:
> containment, dismissals, team score. Every one is a deterministic, testable
> check. That's RLVR, verifiable rewards.

**ACT 4: the learnable gap**
> You can't train a model in a ten-minute talk. So here's the honest bracket: a
> random policy scores zero-point-zero-six. The scripted oracle ceiling: zero-
> point-nine. That gap, plus zero-point-eight-four, is the measurable headroom
> a trained policy closes. And full honesty: our published checkpoint underfit.
> Twenty-seven optimizer steps on a free T4 moves nothing. The pipeline is
> proven. The compute was rented.

**ACT 5: safeguards**
> Last act. We attacked our own reward function and found six exploits: no-op
> farming, duplicate case-closes, escalation flooding. Every one is fixed and
> locked in as a regression test, a hundred and eleven tests passing, and
> served live at slash-themes-slash-coverage. If you only remember one line:
> **a reward function is only as good as the exploits it can't be farmed by.**

*Close Terminal B window or just switch back to the browser.*

## 6:50 – 7:50 · WHY THIS FITS TONIGHT: SECURING AI AGENTS (~135 words)

*Scroll back up to the Reward & Integrity section, so the six-exploit table is
on screen behind you. Speak to the theme of the night.*

> Tonight is called Securing AI Agents. This project is that phrase running in
> **both directions.**
>
> One direction: agents **for** security. An AI SOC team that triages, escalates,
> contains, audits.
>
> The other direction is the one I care about more: securing **the agents
> themselves.** Those six reward exploits on screen? That *is* agent security.
> The red-team generator? An adversary attacking the team during training. The
> NPC actors pushing unsolicited messages into agent inboxes mid-episode?
> That's your **injection surface.** Policy drift? That's the ground truth
> shifting under the agent's feet.
>
> If you deploy an agent that was never trained under attack, the attack is
> simply scheduled for production. This gym is where SOC agents take the hits
> **first.**

## 7:50 – 8:40 · THE CLOSE (~110 words)

*Scroll to the outro, "Train the team.↗" Stand still. Slow down.*

> Thirteen years ago, the right alert died in a handoff between this city and
> Minneapolis.
>
> The next generation of SOC agents will detect better; that part is coming
> either way. But detection was never the failure. **The handoff was.** And as of
> now, the handoff is trainable.
>
> Everything you saw is open, MIT licensed. The environment, the trained
> checkpoint, the training notebook, this website, one repo, one command:
> `python demo.py`.
>
> Train the team. Not the analyst.
>
> Thank you.

*(Hold two seconds. Then: "I'll take questions.")*

## 8:40 – 10:00 · BUFFER / Q&A

Absorbs every overrun. If you're on time, you get 3 to 4 questions; answers are
pre-written in [TALK_PLAN.md §7](TALK_PLAN.md). For a security crowd, expect
"real SIEM data?", "will this replace analysts?", and "how do you know the
reward isn't hackable?"; all three are in the list.

---

# DELIVERY NOTES: BIG CROWD

1. **The first 10 seconds decide the talk.** Walk out, plant your feet, say the
   first line to the BACK row, and do not move during the Target story. Movement
   starts when the pitch starts (1:20).
2. **The two pauses that matter:** after "Minneapolis closed the ticket." and
   after the ticket prints in Act 2. Two full seconds each. Count them.
3. **Never talk to the screen.** Glance, point, turn back to the crowd. The demo
   waits for your Enter key; that's why it's built that way. Silence while
   output prints is fine; it reads as confidence.
4. **Numbers slow, everything else fast.** "Zero-point-eight-four" said slowly
   lands harder than a paragraph.
5. **Say the alert IP wrong? Score off by one? Keep going.** Nobody has the
   README memorized. Only a stopped presenter looks wrong.
6. **If demo_live.py fails:** say "it's deterministic, here's the same run,"
   scroll to the website's Live Wire section (the terminal animation replays the
   episode), continue the script unchanged. Never debug on stage.
7. **Plant a phone timer** at the podium: one glance at 4:20 ("am I in the
   terminal?") and one at 7:00 ("am I out?"). Those are the only two checkpoints
   that matter.
8. **The show-of-hands option:** if the crowd is warm, after "the most expensive
   ignored notification in history" you can add: "Quick hands, who here has ever
   worked a SOC shift, or sat next to someone who has?" In Bengaluru you'll get
   hands. Then: "Then you already know this story." If the crowd is cold, skip
   it; a failed hands-raise deflates the open.

---

# WORD-COUNT LEDGER (sanity check)

| Segment | Time | Words | Pace check |
|---|---|---|---|
| Hook | 1:20 | ~185 | 139 wpm ✓ |
| Intro | 0:25 | ~55 | 132 wpm ✓ |
| Thesis | 0:55 | ~130 | 142 wpm ✓ |
| Architecture | 1:10 | ~165 | 141 wpm ✓ |
| Team mode | 0:30 | ~70 | 140 wpm ✓ |
| Demo (5 acts) | 2:30 | ~280 spoken | terminal carries the rest |
| Why it matters | 1:00 | ~140 | 140 wpm ✓ |
| Close | 0:50 | ~110 | 132 wpm ✓ |
| **Total** | **8:40** | **~1135** | 1:20 buffer ✓ |
