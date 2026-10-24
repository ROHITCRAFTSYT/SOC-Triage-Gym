# Security Policy

## Supported Version

Security fixes are applied to the latest commit on the default branch (`main`), which is the active submission for the OpenEnv Apr 2026 Hackathon and the live Hugging Face Space deployment.

Assume the latest repository version is the only supported version unless stated otherwise.

---

## Scope

SOC-Triage-Gym v3 is a simulated cybersecurity training environment built for reinforcement learning research. It is **not** a production SOC platform, live threat intelligence service, or real incident response system.

### What the project contains

- Synthetic SIEM alert scenarios (phishing, lateral movement, insider threat, APT campaign)
- Simulated log, enrichment, and asset databases (all deterministic and fictitious)
- Deterministic grading logic and reward functions
- External NPC actor simulation (ThreatIntelFeed, ComplianceOfficer, EndUserReporter)
- Mid-episode policy/schema drift engine
- Enterprise ticketing simulation with SLA clocks
- Expert panel rotation for reward shaping
- FastAPI HTTP server exposing environment endpoints
- Per-step GRPO training harness and deterministic replay CLI

### What the project does NOT contain

- Real customer telemetry or live network data
- Live credentials or API secrets committed to the repository
- Production detection pipelines or real threat feeds
- Any personally identifiable information

---

## Reporting a Vulnerability

**Do not open a public GitHub issue containing exploit details, credentials, or sensitive payloads.**

Report privately through one of the following channels:

- **GitHub:** Open a [private security advisory](https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2/security/advisories/new) on this repository
- **Email:** ytrohitcrafts@gmail.com
- **GitHub profile:** [@ROHITCRAFTSYT](https://github.com/ROHITCRAFTSYT)

When reporting, please include:

- Short description of the issue
- Affected file(s) or endpoint(s)
- Reproduction steps
- Impact assessment (data leakage, RCE, privilege escalation, SSRF, etc.)
- Suggested fix if available

Allow reasonable time for confirmation and remediation before public disclosure.

---

## What To Report

Relevant security reports include:

- Accidental credential or secret exposure in the repository
- Unsafe dependency or deployment configuration
- Server-Side Request Forgery (SSRF) in any HTTP endpoint (e.g. `/generate_scenario`, `/mcp`, actor/policy endpoints)
- Path traversal or command injection via any API parameter
- Unintended remote code execution paths
- Authentication or authorization bypass if access controls are added in future versions
- Unsafe deserialization or prototype pollution via JSON inputs to the FastAPI server
- Supply chain issues in Python dependencies (FastAPI, Pydantic v2, Uvicorn, OpenAI SDK, TRL, Unsloth)

---

## Out Of Scope

The following are **not** considered security issues for this project:

- Weaknesses in the fictional scenario content (synthetic IOCs, simulated logs, made-up alert titles)
- Attacks against synthetic threat indicators or simulated enrichment data
- Grader disagreements or reward shaping decisions that do not create an actual security impact
- Policy drift engine producing unexpected policy versions (this is intentional behavior)
- Token-scaled reward values that differ from a reviewer's expectation (expected, configurable)
- Findings that require this environment to be deployed alongside unrelated insecure infrastructure
- DoS via high step-count episodes (the environment has per-episode step budgets)

---

## Operational Guidance

If you deploy SOC-Triage-Gym publicly (e.g. Hugging Face Spaces, cloud VM):

- **Never** commit real secrets, API keys, or credentials to the repository
- Use environment variables or your hosting platform's secret manager for `HF_TOKEN`, `OPENAI_API_KEY`, `API_BASE_URL`, and `MODEL_NAME`
- The `/generate_scenario` and `/mcp` endpoints accept arbitrary JSON — validate inputs in production deployments
- The ManagerJudge makes outbound LLM API calls; configure firewall rules if running in a restricted environment
- Treat this as a demo/research system, not a hardened production service
- Review `pyproject.toml` and `uv.lock` before publishing new builds
- Keep dependencies updated; GitHub Dependabot alerts are monitored on the default branch

---

## Dependency and Supply Chain Notes

Core runtime dependencies include:

| Package | Purpose |
|---------|---------|
| FastAPI + Uvicorn | HTTP server |
| Pydantic v2 | Data validation (v2 required — v1 is incompatible) |
| OpenAI SDK | ManagerJudge LLM calls (optional, falls back to heuristic) |
| TRL / Unsloth | GRPO training harness (Colab / GPU only) |
| Pytest + HTTPX | Test suite (108 tests) |

Review lockfiles and deployment manifests before publishing new builds. The two moderate Dependabot alerts on the default branch are pre-existing and do not affect the training or inference paths.
