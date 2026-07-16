# Running SOC-Triage-Gym in Production

This guide covers deploying SOC-Triage-Gym as a shared service — for a
security team benchmarking agents, an ML platform running RL training jobs,
or a vendor evaluating SOC copilots. Everything here ships in the box; there
are no extra dependencies to install.

## What "production" adds (v0.2.0)

| Capability | Default | Enable with |
|---|---|---|
| Multi-session concurrency | On (auto) | `X-Session-ID` header per client |
| API-key authentication | Off | `SOC_GYM_API_KEY` |
| Rate limiting | Off | `SOC_GYM_RATE_LIMIT` (req/min per client) |
| Prometheus metrics | On | scrape `GET /metrics` |
| Episode audit trail | On (in-memory) | durable export via `SOC_GYM_AUDIT_DIR` |
| Session admin API | On | `GET /sessions`, `DELETE /sessions/{id}` |
| Hardened container | On | non-root user, read-only FS in compose |

All features are additive and backward compatible: a stock OpenEnv client
that only knows `/reset`, `/step`, `/state` works unchanged.

## Quick deploy

```bash
# 1. Configure
cp .env.example .env
# set SOC_GYM_API_KEY=<long random string>, SOC_GYM_RATE_LIMIT=120

# 2. Launch
docker compose up -d

# 3. Verify
pip install -e .
soc-gym validate --url http://localhost:7860 --api-key "$SOC_GYM_API_KEY"
```

`soc-gym validate` probes health, metadata, task catalog, schema, metrics,
audit trail, and session endpoints, and exits non-zero if any fail — wire it
into your deploy pipeline as a smoke test.

## Multi-session concurrency

Every client picks its own isolated environment by sending a session ID.
Sessions have independent episodes, NPC actors, policy-drift schedules,
ticketing, and reward-blend configs, each behind its own lock.

```python
from client import SOCTriageClient

# Two trainers sharing one server, zero interference:
a = SOCTriageClient("http://gym.internal:7860", session_id="trainer-1", api_key="...")
b = SOCTriageClient("http://gym.internal:7860", session_id="trainer-2", api_key="...")
a.reset("phishing", seed=1)
b.reset("apt_campaign", seed=7)   # does not disturb trainer-1
```

Raw HTTP clients set the header directly:

```bash
curl -X POST http://localhost:7860/reset \
  -H "X-Session-ID: trainer-1" \
  -H "Authorization: Bearer $SOC_GYM_API_KEY" \
  -d '{"task_id": "phishing", "seed": 42}'
```

Session IDs are `[A-Za-z0-9._-]{1,64}`. Requests without an ID share the
`default` session (the original single-tenant behaviour). Pool sizing:

- `SOC_GYM_MAX_SESSIONS` (default 64) — LRU eviction beyond the cap; the
  default session is never evicted.
- `SOC_GYM_SESSION_TTL` (default 3600s) — idle sessions are reclaimed.

Operators can inspect and manage the pool:

```bash
curl http://localhost:7860/sessions            # list live sessions + episode state
curl -X DELETE http://localhost:7860/sessions/trainer-1
```

## Authentication and rate limiting

Set `SOC_GYM_API_KEY` and every API request must carry the key:

```
Authorization: Bearer <key>      # preferred
X-API-Key: <key>                 # alternative
```

Keys are compared in constant time. `/health`, the landing pages (`/`,
`/ui*`), and the API docs stay open so load balancers and humans aren't
locked out. TLS is your reverse proxy's job — terminate HTTPS at nginx,
Caddy, or your cloud load balancer in front of the container.

Set `SOC_GYM_RATE_LIMIT` (requests/minute) to throttle each client (keyed by
API key, else client IP) with a token bucket. Throttled requests get `429`
with a `Retry-After` header, and the bundled SDK retries them automatically
with exponential backoff.

## Observability

`GET /metrics` serves Prometheus text format with no extra dependencies:

- `socgym_requests_total{method,path,status}` and per-route latency
  accumulators (`socgym_request_seconds_sum/count`)
- `socgym_episodes_started_total{task}` / `socgym_episodes_completed_total{task}`
- `socgym_steps_total{task}` and final-reward accumulators
  (`socgym_reward_sum/count{task}` → average reward per task in PromQL:
  `socgym_reward_sum / socgym_reward_count`)
- `socgym_active_sessions`, `socgym_uptime_seconds`

Scrape config:

```yaml
scrape_configs:
  - job_name: soc-gym
    static_configs: [{ targets: ["gym.internal:7860"] }]
```

## Episode audit trail

Every reset and step is recorded per episode — action taken, reward
received, running totals, timestamps, acting role. That gives you replayable
evidence of what an agent actually did, which is the difference between "the
model scored 0.8" and a defensible triage record.

```bash
curl http://localhost:7860/episodes                       # list (filter: ?session_id=)
curl http://localhost:7860/episodes/<id>/trace            # full JSON trace
curl "http://localhost:7860/episodes/<id>/trace?format=jsonl"  # SIEM/data-lake export
```

The in-memory window keeps the most recent `SOC_GYM_AUDIT_MAX_EPISODES`
(default 200). For durable storage set `SOC_GYM_AUDIT_DIR` — each episode
appends to its own JSONL file (the compose file mounts a volume at
`/data/audit` for exactly this).

## SDK for integrators

`client.py` is a production-grade SDK: retries with exponential backoff on
connection errors / 429 / 5xx, session + API-key wiring, and a one-call
episode driver:

```python
from client import SOCTriageClient
from models import SOCAction

def policy(obs):
    return SOCAction(action_type="submit_investigation")

with SOCTriageClient("http://gym.internal:7860", session_id="eval-9") as c:
    final, trajectory = c.run_episode(policy, task_id="phishing", seed=42)
    print(final.cumulative_reward, len(trajectory))
    print(c.episodes()[0])          # audit summary of the run just recorded
```

## CLI

```bash
soc-gym serve --api-key "$KEY" --rate-limit 120   # start the server
soc-gym tasks                                     # task catalog
soc-gym demo                                      # 5-beat guided demo
soc-gym benchmark --task phishing --seeds 42,123  # deterministic benchmark
soc-gym validate --url https://gym.example.com --api-key "$KEY"
soc-gym train --role tier1 --curriculum --parallel-rewards 4   # see docs/TRAINING.md
soc-gym runs                                      # list structured training runs
```

## Scaling notes

- One container comfortably serves dozens of concurrent training sessions;
  episodes are in-memory and CPU-light. Scale vertically first.
- Sessions are process-local. If you run multiple replicas, use sticky
  routing on `X-Session-ID` (or give each trainer its own replica). A shared
  session store is intentionally out of scope — episodes are cheap to
  recreate from `(task_id, seed)`.
- The environment is deterministic per `(task_id, seed)`, so blue/green
  deploys don't invalidate benchmarks.

## Security posture

- Container runs as a non-root user; the compose file mounts the root FS
  read-only with a tmpfs `/tmp` and a dedicated writable audit volume.
- No secrets are baked into the image; all configuration is via environment.
- The environment itself is synthetic — no real IOCs, no real user data —
  so an exposed instance leaks nothing sensitive. Auth mainly protects
  compute and benchmark integrity.
- See [SECURITY.md](../SECURITY.md) for the vulnerability disclosure policy.
