# Judges: start here

**90-second read. Links go to source code.**

### What it is

An OpenEnv environment for training a 3-role SOC team (Tier-1 → Tier-2 → Manager), not a single agent. Trained with GRPO; evaluated on 8 tasks ranging from single-alert triage to a **250-step APT campaign**.

### The one thing that makes it different

Every other submission trains one policy. This one trains a **team** with a ticket bus, a 3-phase FSM, and a reward that mixes individual and team F1 — so an agent maximizing personal score at team cost gets penalized. See [server/environment.py](server/environment.py).

### 60-second proof it works

```bash
# 1. Clone + start env (port 7860)
git clone https://github.com/ROHITCRAFTSYT/SOC-Triage-Gym.git && cd SOC-Triage-Gym
pip install -e ".[dev]"
uvicorn server.app:app --port 7860 &

# 2. Machine-checkable theme manifest — every claim is code-backed
curl -s localhost:7860/themes/coverage | jq '.coverage, .reward_hacking_defenses'

# 3. Full 5-beat walkthrough (baseline → verifier → trained → delta → safeguards)
python demo.py
```

### What it covers

| Theme | Status | Source |
|---|---|---|
| **#1 Multi-Agent** (primary) | 3-role team, ticket bus, blended reward | [server/environment.py](server/environment.py) |
| **#2 Long-Horizon** | 250-step APT campaign, 60+ alerts, sparse reward | [scenarios/apt_campaign.py](scenarios/apt_campaign.py) |
| **#3.1 Professional Tasks** | 8 realistic SOC tools (enrichment, containment, forensics) | [server/app.py](server/app.py) |
| **#4 Self-Improvement** | Red-team curriculum that adapts to blue win rate | [scenarios/red_team_generator.py](scenarios/red_team_generator.py) |
| Fleet AI | LLM-judged manager oversight with heuristic fallback | [graders/manager_judge.py](graders/manager_judge.py) |
| Halluminate | 3 NPC actor types pushing unsolicited messages | [actors/](actors/) |
| Mercor | Token-length-scaled rewards w/ floor+cap | [graders/token_scaled_reward.py](graders/token_scaled_reward.py) |
| Patronus | Mid-episode policy drift (schema version-aware graders) | [scenarios/policy_drift.py](scenarios/policy_drift.py) |
| Snorkel | 3 rotating expert judges with drifting weights | [graders/expert_panel.py](graders/expert_panel.py) |
| Scale AI | Ticketing system with SLA clocks cross-enforced on tools | [tools/ticketing.py](tools/ticketing.py) |

### Verifiable facts

- **111 tests passing (1 skipped).** `pytest tests/ -q`. Includes 21 theme-coverage regression tests.
- **Reward-hacking defenses** are asserted as tests, not just claimed: see [tests/test_themes_coverage.py](tests/test_themes_coverage.py).
- **Deterministic**: same seed → same score. Verified in [benchmark.py](benchmark.py).
- **Trained artifacts shipped**: [trained_vs_baseline.png](trained_vs_baseline.png), [training_summary.json](training_summary.json), LoRA adapter in `checkpoints/soc_grpo_tier1/`.

### Honest limitations

1. Trained on **tier1 only** (tier2 and manager are frozen oracle during tier1 training — intentional, staged curriculum).
2. Manager-judge heuristic fallback triggers if no `OPENAI_API_KEY` — brittle on synonyms. We ship the API path as primary.
3. Free-tier T4 trains in ~90 min; A10G cuts that to ~25. No H100 optimizations.

### Where to look

- 5-beat walkthrough: [demo.py](demo.py)
- One-command train+eval: [scripts/train_and_evaluate.py](scripts/train_and_evaluate.py)
- Full rubric mapping: [README.md](README.md#why-this-project-wins)
- Failure modes + future work: [README.md#honest-limitations](README.md#honest-limitations)
