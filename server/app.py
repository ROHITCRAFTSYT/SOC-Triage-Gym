"""
SOC-Triage-Gym FastAPI Application
=====================================
OpenEnv-compliant HTTP server exposing:
  POST /reset                          — start new episode
  POST /step                           — execute action, get observation
  GET  /state                          — current episode metadata
  GET  /health                         — liveness check

Additional REST tool endpoints (for LLM agents using direct REST calls):
  GET  /api/alerts                     — list current alerts
  GET  /api/alerts/{alert_id}          — get single alert detail
  GET  /threat-intel/ip/{ip}           — IP threat intelligence lookup
  GET  /threat-intel/domain/{domain}   — domain threat intelligence lookup
  GET  /threat-intel/hash/{file_hash}  — file hash threat intelligence lookup
  GET  /logs/{source}                  — query log source
  GET  /api/tasks                      — list available tasks

Thread safety: a single threading.Lock protects the SOCEnvironment instance.
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from baseline_agent import HeuristicBaselineAgent
from graders.token_scaled_reward import explain as explain_token_bonus
from models import (
    AgentRole,
    AlertClassification,
    EnvironmentState,
    RedTeamConfig,
    SOCAction,
    SOCObservation,
)
from scenarios.red_team_generator import RedTeamGenerator
from server.audit import AUDIT, trace_to_jsonl
from server.environment import SOCEnvironment
from server.landing_ui import UI_HTML
from server.metrics import METRICS, MetricsMiddleware
from server.page_ui import (
    render_metadata as _render_metadata_page,
)
from server.page_ui import (
    render_schema as _render_schema_page,
)
from server.page_ui import (
    render_state as _render_state_page,
)
from server.page_ui import (
    render_tasks as _render_tasks_page,
)
from server.page_ui import (
    render_themes as _render_themes_page,
)
from server.security import SecurityMiddleware
from server.sessions import DEFAULT_SESSION_ID, SessionManager, SessionState

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
# All episode state (environment + v3 theme modules) lives in per-session
# containers so concurrent clients never share or corrupt each other's
# episodes. Requests without an X-Session-ID header use the "default"
# session, which preserves the original single-tenant behaviour.

_sessions = SessionManager()
_baseline_agent = HeuristicBaselineAgent()

VERSION = "0.3.0"


def get_session(
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> SessionState:
    """FastAPI dependency: resolve (or create) the caller's session."""
    try:
        return _sessions.get_or_create(x_session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _do_reset(sess: SessionState, task_id: str, seed: int, mode: str) -> SOCObservation:
    """Reset a session's episode + v3 modules; record metrics/audit. Call under sess.lock."""
    obs = sess.env.reset(task_id=task_id, seed=seed, mode=mode)
    sess.on_reset(seed=seed)
    METRICS.record_episode_start(task_id)
    AUDIT.start_episode(
        episode_id=sess.env._episode_id or "unknown",
        session_id=sess.session_id,
        task_id=task_id,
        seed=seed,
        mode=mode,
    )
    return obs


def _do_step(sess: SessionState, action: SOCAction) -> SOCObservation:
    """Step a session's episode + v3 modules; record metrics/audit. Call under sess.lock."""
    obs = sess.env.step(action)
    sess.on_step()
    task = sess.env._task_id or "unknown"
    METRICS.record_step(task)
    role = getattr(action, "role", None)
    AUDIT.record_step(
        episode_id=sess.env._episode_id or "unknown",
        step=sess.env._step,
        action=action.model_dump(exclude_none=True),
        reward=float(getattr(obs, "reward", 0.0) or 0.0),
        cumulative_reward=float(sess.env._cumulative_reward),
        done=bool(obs.done),
        role=str(role) if role is not None else None,
    )
    if obs.done:
        METRICS.record_episode_complete(task, float(sess.env._cumulative_reward))
    return obs

TASKS = [
    {
        "id": "phishing",
        "name": "Single-Alert Phishing Triage",
        "description": "Triage a single phishing email alert. Enrich IOCs, query logs, classify as TP or FP, map MITRE ATT&CK technique, recommend response.",
        "difficulty": "easy",
        "max_steps": 15,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "lateral_movement",
        "name": "Multi-Alert Lateral Movement Kill Chain",
        "description": "Investigate 5 correlated alerts forming a kill chain: phishing, credential dump, lateral movement, data staging, exfiltration.",
        "difficulty": "medium",
        "max_steps": 30,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "queue_management",
        "name": "Alert Queue Management Under Noise",
        "description": "Triage 20 mixed alerts: 5 true positives in 2 attack chains, 3 benign true positives, 12 false positives.",
        "difficulty": "hard",
        "max_steps": 60,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "insider_threat",
        "name": "Insider Threat Investigation",
        "description": "Investigate 30 alerts hiding 3 insider threat attack chains: unauthorized data theft, compromised vendor, disgruntled employee. 9 TPs, 5 BTPs, 16 FPs.",
        "difficulty": "expert",
        "max_steps": 80,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "team_phishing_escalation",
        "name": "Team Phishing Escalation",
        "description": "Tier-1 triages a phishing alert, escalates to Tier-2 for containment, and Manager audits the decision trail.",
        "difficulty": "easy",
        "max_steps": 68,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "team_lateral_team",
        "name": "Team Lateral Movement",
        "description": "Tier-1 triages a noisy lateral movement queue, Tier-2 responds to escalations, and Manager flags missed threats and inconsistencies.",
        "difficulty": "medium",
        "max_steps": 68,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "apt_campaign",
        "name": "APT Campaign (Super Long-Horizon)",
        "description": "250-step composite campaign with 60+ alerts across 5 phases (initial access → persistence → lateral → exfil → cleanup). Sparse delayed reward, policy drift, rotating expert judge.",
        "difficulty": "super-hard",
        "max_steps": 250,
        "reward_range": [0.0, 1.0],
    },
    {
        "id": "red_team_generated",
        "name": "Generated Adversarial Scenario",
        "description": "Execute the most recently generated red-team curriculum scenario.",
        "difficulty": "adaptive",
        "max_steps": 30,
        "reward_range": [0.0, 1.0],
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the default session on startup so the first request is fast."""
    _sessions.get_or_create(DEFAULT_SESSION_ID)
    yield
    # Cleanup on shutdown (nothing needed for in-memory sessions)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SOC-Triage-Gym",
    description=(
        "OpenEnv-compliant reinforcement learning environment simulating a "
        "Security Operations Center analyst. An AI agent investigates SIEM alerts "
        "by enriching threat indicators, querying log sources, correlating events, "
        "and classifying alerts with MITRE ATT&CK mapping."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Session-ID"],
)
# Order matters: middleware added later runs first. Security (auth + rate
# limit) is innermost of the two; Metrics is outermost so it also counts
# rejected (401/429) requests.
app.add_middleware(SecurityMiddleware)
app.add_middleware(MetricsMiddleware)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: str = "phishing"
    seed: int = 42
    mode: Literal["tier1_solo", "team"] = "tier1_solo"
    # Optional body-level session selector for clients that can't set the
    # X-Session-ID header. Takes precedence over the header when present.
    session_id: str | None = None


class GenerateScenarioRequest(BaseModel):
    """Request body for POST /generate_scenario."""
    seed: int = 42
    difficulty_floor: float = 0.5
    attack_patterns: list[str] = ["phishing", "lateral_movement", "insider_threat"]
    noise_density: float = 0.6
    ioc_freshness: float = 0.7
    correlation_obfuscation: float = 0.3
    blue_team_win_rate: float = 0.5
    episode_count: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — returns 200 when server is running."""
    return {
        "status": "healthy",
        "version": VERSION,
        "env": "soc-triage-gym",
        "active_sessions": len(_sessions),
    }


@app.get("/metadata")
def metadata():
    """Environment metadata (OpenEnv runtime spec)."""
    return {
        "name": "soc-triage-gym",
        "version": VERSION,
        "description": (
            "A reinforcement learning environment simulating a Security Operations Center "
            "(SOC) analyst. The agent investigates SIEM alerts by enriching threat "
            "indicators, querying log sources, correlating events, and classifying alerts "
            "with MITRE ATT&CK technique mapping."
        ),
        "tasks": [task["id"] for task in TASKS],
        "author": "rohitcraftsyt",
        "tags": ["openenv", "cybersecurity", "soc", "siem", "mitre-attack", "reinforcement-learning", "multi-agent", "oversight", "self-improvement"],
    }


@app.get("/schema")
def schema():
    """Action, observation and state JSON schemas (OpenEnv runtime spec)."""
    from models import EnvironmentState, SOCAction, SOCObservation
    return {
        "action": SOCAction.model_json_schema(),
        "observation": SOCObservation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.post("/mcp")
def mcp_endpoint(
    request: dict | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """
    MCP (Model Context Protocol) JSON-RPC 2.0 endpoint.

    Supports:
      - tools/list  — enumerate all available tools with JSON Schema inputs
      - tools/call  — execute a tool by name with params
    """
    req = request or {}
    method = req.get("method", "")
    req_id = req.get("id", 1)
    params = req.get("params", {})

    def _jsonrpc_ok(result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _jsonrpc_err(code, message, data=None):
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    # -- MCP tool definitions -------------------------------------------------
    MCP_TOOLS = [
        {
            "name": "reset",
            "description": "Start a new episode. Returns initial observation with full alert queue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "enum": [task["id"] for task in TASKS], "default": "phishing"},
                    "seed": {"type": "integer", "default": 42},
                    "mode": {"type": "string", "enum": ["tier1_solo", "team"], "default": "tier1_solo"},
                },
            },
        },
        {
            "name": "step",
            "description": "Execute a single action in the current episode. Wraps POST /step.",
            "inputSchema": SOCAction.model_json_schema(),
        },
        {
            "name": "state",
            "description": "Get current episode metadata (step count, reward, done flag, etc.) without consuming a step.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "enrich_indicator",
            "description": "Enrich a threat indicator (IP, domain, hash, email, URL) via threat intelligence feeds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "Indicator value (e.g. IP address, domain)"},
                    "indicator_type": {"type": "string", "enum": ["ip", "domain", "file_hash", "email", "url", "user"]},
                    "query_alert_id": {"type": "string", "description": "Alert ID this enrichment is for (optional)"},
                },
                "required": ["indicator", "indicator_type"],
            },
        },
        {
            "name": "query_logs",
            "description": "Query a SIEM log source for events related to an alert.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "log_source": {"type": "string", "enum": ["firewall", "proxy", "dns", "endpoint", "auth", "email_gateway", "ids", "cloud_trail"]},
                    "query_alert_id": {"type": "string", "description": "Alert ID to scope the query"},
                    "time_window_hours": {"type": "integer", "default": 24, "minimum": 1, "maximum": 168},
                },
                "required": ["log_source"],
            },
        },
        {
            "name": "correlate_alerts",
            "description": "Check two alerts for shared indicators, IPs, users, techniques, or time-window overlap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_id_a": {"type": "string"},
                    "alert_id_b": {"type": "string"},
                },
                "required": ["alert_id_a", "alert_id_b"],
            },
        },
        {
            "name": "classify_alert",
            "description": "Classify an alert as true_positive, false_positive, or benign_true_positive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_id": {"type": "string"},
                    "classification": {"type": "string", "enum": ["true_positive", "false_positive", "benign_true_positive"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8},
                },
                "required": ["alert_id", "classification"],
            },
        },
        {
            "name": "map_technique",
            "description": "Map a MITRE ATT&CK technique ID to an alert (e.g. T1566.001).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_id": {"type": "string"},
                    "technique_id": {"type": "string", "description": "MITRE ATT&CK ID, e.g. T1566.001"},
                },
                "required": ["alert_id", "technique_id"],
            },
        },
        {
            "name": "recommend_action",
            "description": "Recommend a containment or response action for an alert.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alert_id": {"type": "string"},
                    "response_action": {"type": "string", "enum": ["isolate_endpoint", "disable_account", "block_ip", "block_domain", "quarantine_file", "reset_password", "revoke_sessions", "no_action"]},
                },
                "required": ["alert_id", "response_action"],
            },
        },
        {
            "name": "check_asset",
            "description": "Look up a host in the asset inventory by hostname.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                },
                "required": ["hostname"],
            },
        },
        {
            "name": "check_user",
            "description": "Look up a user profile in directory services.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                },
                "required": ["username"],
            },
        },
        {
            "name": "submit_investigation",
            "description": "Submit the investigation for grading. Ends the episode and returns the final score.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    # -- tools/list -----------------------------------------------------------
    if method == "tools/list":
        return _jsonrpc_ok({"tools": MCP_TOOLS})

    # -- tools/call -----------------------------------------------------------
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            if tool_name == "reset":
                with sess.lock:
                    obs = _do_reset(
                        sess,
                        task_id=tool_args.get("task_id", "phishing"),
                        seed=tool_args.get("seed", 42),
                        mode=tool_args.get("mode", "tier1_solo"),
                    )
                    return _jsonrpc_ok({"content": [{"type": "text", "text": obs.model_dump_json()}]})

            if tool_name == "state":
                with sess.lock:
                    st = sess.env.state()
                    return _jsonrpc_ok({"content": [{"type": "text", "text": st.model_dump_json()}]})

            if tool_name == "submit_investigation":
                with sess.lock:
                    if sess.env._config is None:
                        return _jsonrpc_err(-32602, "No active episode. Call reset first.")
                    obs = _do_step(sess, SOCAction(action_type="submit_investigation"))
                    return _jsonrpc_ok({"content": [{"type": "text", "text": obs.model_dump_json()}]})

            # Tools that map directly to a step action
            ACTION_MAP = {
                "step": None,  # pass-through
                "enrich_indicator": "enrich_indicator",
                "query_logs": "query_logs",
                "correlate_alerts": "correlate_alerts",
                "classify_alert": "classify_alert",
                "map_technique": "map_technique",
                "recommend_action": "recommend_action",
                "check_asset": "check_asset",
                "check_user": "check_user",
            }

            if tool_name in ACTION_MAP:
                with sess.lock:
                    if sess.env._config is None:
                        return _jsonrpc_err(-32602, "No active episode. Call reset first.")
                    if tool_name == "step":
                        action_data = tool_args
                    else:
                        action_data = {"action_type": ACTION_MAP[tool_name], **tool_args}
                    action = SOCAction(**action_data)
                    obs = _do_step(sess, action)
                    return _jsonrpc_ok({"content": [{"type": "text", "text": obs.model_dump_json()}]})

            return _jsonrpc_err(-32601, f"Unknown tool: '{tool_name}'")

        except ValueError as e:
            logger.warning("MCP tools/call invalid params: %s", e)
            return _jsonrpc_err(-32602, "Invalid params")
        except Exception:
            logger.exception("MCP tools/call error")
            return _jsonrpc_err(-32603, "Internal error")

    # -- unknown method -------------------------------------------------------
    return _jsonrpc_err(-32601, f"Unknown method: '{method}'")


@app.post("/reset", response_model=SOCObservation)
def reset(
    request: ResetRequest | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """
    Start a new episode.

    Args:
        task_id: "phishing" | "lateral_movement" | "queue_management" | "insider_threat" (default: "phishing")
        seed: RNG seed for deterministic scenario generation (default: 42)
        session_id: Optional session selector (overrides the X-Session-ID header)

    Returns:
        Initial SOCObservation with full alert queue.
    """
    req = request or ResetRequest()
    if req.session_id:
        try:
            sess = _sessions.get_or_create(req.session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    with sess.lock:
        try:
            return _do_reset(sess, task_id=req.task_id, seed=req.seed, mode=req.mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception("Error in /reset")
            raise HTTPException(status_code=500, detail="Internal server error.") from None


@app.post("/step", response_model=SOCObservation)
def step(action: SOCAction, sess: SessionState = Depends(get_session)):
    """
    Execute an action in the current episode.

    The action_type field determines which other fields are relevant.
    See the SOCAction model for the full action schema.

    Returns:
        Updated SOCObservation with step reward, new results, and done flag.
    """
    with sess.lock:
        if sess.env._config is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call POST /reset first.",
            )
        try:
            return _do_step(sess, action)
        except Exception:
            logger.exception("Error in /step")
            raise HTTPException(status_code=500, detail="Internal server error.") from None


@app.get("/state", response_model=EnvironmentState)
def state(sess: SessionState = Depends(get_session)):
    """
    Get current episode metadata without consuming a step.

    Returns episode_id, task_id, step_count, max_steps, done,
    cumulative_reward, alert_count, classified_count, seed.
    """
    with sess.lock:
        try:
            return sess.env.state()
        except Exception:
            logger.exception("Error in /state")
            raise HTTPException(status_code=500, detail="Internal server error.") from None


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Interactive browser dashboard for the SOC Triage Gym environment."""
    return UI_HTML


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Primary Space landing page."""
    return UI_HTML


@app.get("/blog.md", response_class=PlainTextResponse, include_in_schema=False)
def blog_md():
    """Raw blog.md so the landing-page modal can render it client-side via marked.js."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "blog.md"
    if not p.exists():
        return PlainTextResponse("# blog.md not found\n", status_code=404)
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


# ---------------------------------------------------------------------------
# /ui/* — styled HTML accessors for the JSON-bearing endpoints.
# API consumers keep hitting the raw routes; these are browser-facing.
# ---------------------------------------------------------------------------

@app.get("/ui/metadata", response_class=HTMLResponse, include_in_schema=False)
def ui_metadata():
    return _render_metadata_page(metadata())


@app.get("/ui/tasks", response_class=HTMLResponse, include_in_schema=False)
def ui_tasks():
    return _render_tasks_page(TASKS)


@app.get("/ui/themes", response_class=HTMLResponse, include_in_schema=False)
def ui_themes():
    return _render_themes_page(themes_coverage())


@app.get("/ui/state", response_class=HTMLResponse, include_in_schema=False)
def ui_state(sess: SessionState = Depends(get_session)):
    with sess.lock:
        sess.ensure_episode()
        snap = sess.env.state()
    payload = snap.model_dump() if hasattr(snap, "model_dump") else dict(snap)
    return _render_state_page(payload)


@app.get("/ui/schema", response_class=HTMLResponse, include_in_schema=False)
def ui_schema():
    return _render_schema_page(schema())


# ---------------------------------------------------------------------------
# REST Tool Endpoints (for LLM agents using direct REST tool calls)
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks():
    """List all available tasks (OpenEnv spec endpoint)."""
    return {"tasks": TASKS}


@app.post("/grader")
def grader(
    request: ResetRequest | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """
    Run the grader on the current episode state (OpenEnv spec endpoint).

    Evaluates the current investigation state against ground truth and returns
    a normalized score in [0.0, 1.0]. Does not terminate the episode.
    """
    with sess.lock:
        sess.ensure_episode()
        try:
            score, breakdown, feedback = sess.env.grade_with_breakdown()
            return {
                "score": score,
                "breakdown": breakdown,
                "feedback": feedback,
                "task_id": sess.env._task_id,
                "steps_used": sess.env._step,
                "max_steps": sess.env._config.max_steps if sess.env._config else 0,
                "done": sess.env._done,
            }
        except Exception:
            logger.exception("Error in /grader")
            raise HTTPException(status_code=500, detail="Internal server error.") from None


@app.post("/baseline")
def baseline(
    request: ResetRequest | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """
    Run the heuristic baseline agent on a fresh episode (OpenEnv spec endpoint).

    Resets the environment with the specified task/seed, runs the built-in
    heuristic agent to completion, and returns the final score.
    """
    req = request or ResetRequest()
    with sess.lock:
        try:
            # Reset to a fresh episode
            _do_reset(sess, task_id=req.task_id, seed=req.seed, mode=req.mode)
            _baseline_agent.reset()
            # Run heuristic steps until done
            env = sess.env
            steps = 0
            max_steps = env._config.max_steps if env._config else 0
            while not env._done and steps < max_steps:
                obs = env._build_observation(role=env._current_role(), reward=0.0)
                action = SOCAction(**_baseline_agent.next_action(obs.model_dump()))
                _do_step(sess, action)
                steps += 1
            # Grade the result with breakdown
            score, breakdown, feedback = env.grade_with_breakdown()
            return {
                "task_id": req.task_id,
                "seed": req.seed,
                "steps_used": steps,
                "score": score,
                "breakdown": breakdown,
                "feedback": feedback,
                "agent": "heuristic",
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception("Error in /baseline")
            raise HTTPException(status_code=500, detail="Internal server error.") from None


def _heuristic_baseline_action(env: "SOCEnvironment") -> SOCAction:
    """Simple heuristic for the /baseline endpoint — classify first unclassified alert."""
    config = env._config
    investigations = env._investigations
    # Find first unclassified alert
    for alert in config.alerts:
        aid = alert.alert_id
        inv = investigations.get(aid)
        if inv and inv.classification is None:
            cls = config.ground_truth.alert_classifications.get(aid, AlertClassification.FALSE_POSITIVE)
            return SOCAction(
                action_type="classify_alert",
                alert_id=aid,
                classification=cls,
                confidence=0.7,
            )
    # All classified — submit
    return SOCAction(action_type="submit_investigation")


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get details for a single task by ID."""
    tasks = {task["id"]: task for task in TASKS}
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found. Valid: {list(tasks.keys())}")
    return tasks[task_id]


@app.get("/api/tasks")
def list_tasks():
    """List all available tasks and their configuration."""
    return {"tasks": TASKS}


@app.post("/generate_scenario")
def generate_scenario(
    request: GenerateScenarioRequest | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """Generate and load an adaptive red-team scenario for reset(task_id='red_team_generated')."""
    req = request or GenerateScenarioRequest()
    rt_config = RedTeamConfig(
        difficulty_floor=req.difficulty_floor,
        attack_patterns=req.attack_patterns,
        noise_density=req.noise_density,
        ioc_freshness=req.ioc_freshness,
        correlation_obfuscation=req.correlation_obfuscation,
        blue_team_win_rate=req.blue_team_win_rate,
        episode_count=req.episode_count,
    )
    scenario = RedTeamGenerator(config=rt_config, seed=req.seed).generate()
    with sess.lock:
        sess.env.set_generated_scenario(scenario)
    return scenario.model_dump()


@app.get("/inbox/{role}")
def inbox(role: str, sess: SessionState = Depends(get_session)):
    """Debug endpoint to inspect role-filtered tickets in the current episode."""
    with sess.lock:
        sess.ensure_episode()
        try:
            parsed_role = AgentRole(role)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        obs = sess.env._build_observation(role=parsed_role, reward=0.0)
        return {"role": role, "tickets": [ticket.model_dump() for ticket in obs.tickets]}


@app.get("/api/alerts")
def list_alerts(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    sess: SessionState = Depends(get_session),
):
    """
    List alerts in the current episode queue.

    Returns paginated alert objects with their indicators and metadata.
    Call POST /reset first to start an episode, or a default phishing
    episode will be auto-started.
    """
    with sess.lock:
        sess.ensure_episode()
        alerts = [a.model_dump() for a in sess.env._config.alerts]
        total = len(alerts)
        page = alerts[offset: offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "alerts": page,
        }


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str, sess: SessionState = Depends(get_session)):
    """
    Get full details for a single alert including indicators and metadata.
    """
    with sess.lock:
        sess.ensure_episode()
        for alert in sess.env._config.alerts:
            if alert.alert_id == alert_id:
                inv = sess.env._investigations.get(alert_id)
                return {
                    "alert": alert.model_dump(),
                    "investigation": inv.model_dump() if inv else None,
                }
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.get("/threat-intel/ip/{ip}")
def threat_intel_ip(ip: str, sess: SessionState = Depends(get_session)):
    """
    Look up threat intelligence for an IP address.

    Returns enrichment data including malicious status, reputation score,
    associated threat actors, and related indicators.
    """
    with sess.lock:
        sess.ensure_episode()
        db = sess.env._config.enrichment_db
        result = db.get(ip)
        if result:
            return {
                "indicator": ip,
                "type": "ip",
                "found": True,
                **result.model_dump(),
            }
        return {
            "indicator": ip,
            "type": "ip",
            "found": False,
            "malicious": False,
            "reputation_score": 0,
            "message": "No threat intelligence found for this IP.",
        }


@app.get("/threat-intel/domain/{domain:path}")
def threat_intel_domain(domain: str, sess: SessionState = Depends(get_session)):
    """
    Look up threat intelligence for a domain name.

    Returns enrichment data including malicious status, category,
    registrar info, and associated indicators.
    """
    with sess.lock:
        sess.ensure_episode()
        db = sess.env._config.enrichment_db
        result = db.get(domain)
        if result:
            return {
                "indicator": domain,
                "type": "domain",
                "found": True,
                **result.model_dump(),
            }
        return {
            "indicator": domain,
            "type": "domain",
            "found": False,
            "malicious": False,
            "reputation_score": 0,
            "message": "No threat intelligence found for this domain.",
        }


@app.get("/threat-intel/hash/{file_hash}")
def threat_intel_hash(file_hash: str, sess: SessionState = Depends(get_session)):
    """
    Look up threat intelligence for a file hash (MD5, SHA-1, SHA-256).

    Returns enrichment data including malware family, AV detection rate,
    and associated campaigns.
    """
    with sess.lock:
        sess.ensure_episode()
        db = sess.env._config.enrichment_db
        result = db.get(file_hash)
        if result:
            return {
                "indicator": file_hash,
                "type": "file_hash",
                "found": True,
                **result.model_dump(),
            }
        return {
            "indicator": file_hash,
            "type": "file_hash",
            "found": False,
            "malicious": False,
            "reputation_score": 0,
            "message": "No threat intelligence found for this hash.",
        }


@app.get("/logs/{source}")
def query_log_source(
    source: str,
    alert_id: str | None = Query(default=None),
    hours: int = Query(default=24, ge=1, le=168),
    sess: SessionState = Depends(get_session),
):
    """
    Query a log source for events related to an alert.

    Args:
        source: Log source name (email_gateway, endpoint, auth, firewall,
                dns, proxy, ids, cloud_trail)
        alert_id: Filter logs to events related to this alert
        hours: Time window in hours (default 24, max 168)

    Returns list of log entries with timestamps, event types, and details.
    """
    with sess.lock:
        sess.ensure_episode()
        log_db = sess.env._config.log_db
        entries = []

        source_logs = log_db.get(source, {})

        if alert_id:
            # Get logs for a specific alert within this source.
            entries = [e.model_dump() for e in source_logs.get(alert_id, [])]
        else:
            # Return logs for this source across all alerts.
            for alert_entries in source_logs.values():
                entries.extend(e.model_dump() for e in alert_entries)

        return {
            "source": source,
            "alert_id": alert_id,
            "hours": hours,
            "count": len(entries),
            "entries": entries[:50],  # cap at 50 entries
        }


# ---------------------------------------------------------------------------
# v3 Theme-Coverage Endpoints
# (Halluminate / Patronus / Mercor / Snorkel / Scaler AI Labs)
# ---------------------------------------------------------------------------

@app.get("/actors/messages")
def actor_messages(role: str | None = Query(default=None), sess: SessionState = Depends(get_session)):
    """
    Halluminate sub-theme — Inspect messages from external NPC actors
    (ThreatIntelFeed, ComplianceOfficer, EndUserReporter) in the current episode.
    """
    with sess.lock:
        if role is None:
            msgs = sess.actor_registry.all_messages()
            return {"count": len(msgs), "messages": [m.model_dump() for m in msgs]}
        try:
            parsed = AgentRole(role)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        msgs = sess.actor_registry.inbox_for(parsed)
        return {"role": role, "count": len(msgs), "messages": [m.model_dump() for m in msgs]}


@app.get("/policy/current")
def policy_current(sess: SessionState = Depends(get_session)):
    """Patronus sub-theme — Current active policy version."""
    with sess.lock:
        return sess.policy_drift.current().model_dump()


@app.get("/policy/history")
def policy_history(sess: SessionState = Depends(get_session)):
    """Patronus sub-theme — Full policy-drift history for this episode."""
    with sess.lock:
        return sess.policy_drift.to_dict()


@app.get("/reward/config")
def reward_config(sess: SessionState = Depends(get_session)):
    """Mercor sub-theme — Active reward blend config (role/team/token weights)."""
    with sess.lock:
        return sess.reward_blend.model_dump()


class RewardBlendUpdate(BaseModel):
    """Optional blend override."""
    role_weight: float | None = None
    team_weight: float | None = None
    token_scale_enabled: bool | None = None
    token_scale_floor: int | None = None
    token_scale_cap: int | None = None
    token_scale_max_bonus: float | None = None


@app.post("/reward/config")
def reward_config_update(patch: RewardBlendUpdate, sess: SessionState = Depends(get_session)):
    """Patch the reward blend config. Returns the new config."""
    with sess.lock:
        fields = patch.model_dump(exclude_none=True)
        for k, v in fields.items():
            setattr(sess.reward_blend, k, v)
        return sess.reward_blend.model_dump()


class TokenBonusRequest(BaseModel):
    text: str
    content_quality: float = 0.5


@app.post("/reward/token_bonus")
def reward_token_bonus(req: TokenBonusRequest, sess: SessionState = Depends(get_session)):
    """
    Compute the Mercor token-length bonus for a given text and quality gate.
    Surfaced as an endpoint so agents / judges can preview the incentive curve.
    """
    with sess.lock:
        return explain_token_bonus(req.text, req.content_quality, sess.reward_blend)


@app.get("/experts/current")
def experts_current(sess: SessionState = Depends(get_session)):
    """Snorkel sub-theme — Active reviewing expert and their preference hint."""
    with sess.lock:
        return {
            "round": sess.curriculum_round,
            "expert": sess.current_expert.model_dump(),
            "hint": sess.expert_panel.hint_message(sess.current_expert),
        }


@app.get("/experts/panel")
def experts_panel(sess: SessionState = Depends(get_session)):
    """Snorkel sub-theme — Full expert panel roster."""
    return {"panel": [e.model_dump() for e in sess.expert_panel.all_profiles()]}


class ExpertRotateRequest(BaseModel):
    round_index: int | None = None


@app.post("/experts/rotate")
def experts_rotate(
    req: ExpertRotateRequest | None = Body(default=None),
    sess: SessionState = Depends(get_session),
):
    """
    Advance the expert rotation. If round_index is given, rotate to that round;
    otherwise increment by 1. Emulates Snorkel experts-in-the-loop curriculum.
    """
    with sess.lock:
        if req is not None and req.round_index is not None:
            sess.curriculum_round = int(req.round_index)
        else:
            sess.curriculum_round += 1
        sess.current_expert = sess.expert_panel.for_round(sess.curriculum_round)
        return {
            "round": sess.curriculum_round,
            "expert": sess.current_expert.model_dump(),
        }


class TicketOpenRequest(BaseModel):
    alert_id: str
    priority: str = "P3"
    note: str = ""


@app.post("/tickets/open")
def tickets_open(req: TicketOpenRequest, sess: SessionState = Depends(get_session)):
    """Scaler AI Labs sub-theme — Open a multi-app enterprise ticket."""
    with sess.lock:
        t = sess.ticketing.open(alert_id=req.alert_id, priority=req.priority, note=req.note)
        return t.model_dump()


@app.post("/tickets/{ticket_id}/resolve")
def tickets_resolve(ticket_id: str, note: str = "", sess: SessionState = Depends(get_session)):
    with sess.lock:
        t = sess.ticketing.resolve(ticket_id=ticket_id, note=note)
        if t is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return t.model_dump()


@app.get("/tickets")
def tickets_list(sess: SessionState = Depends(get_session)):
    """List all tickets in the current episode."""
    with sess.lock:
        return {
            "tickets": [t.model_dump() for t in sess.ticketing.all_tickets()],
            "audit": sess.ticketing.audit_summary(),
        }


@app.get("/tickets/can_disable_user")
def tickets_can_disable_user(alert_id: str, sess: SessionState = Depends(get_session)):
    """Cross-app business rule: can IAM.disable_user fire for this alert?"""
    with sess.lock:
        return {"alert_id": alert_id, "allowed": sess.ticketing.can_disable_user(alert_id)}


@app.get("/themes/coverage")
def themes_coverage():
    """
    Machine-checkable summary of which hackathon themes / sub-themes this
    env implements. Served so judges can verify coverage without reading code.
    """
    return {
        "primary_theme": "Theme #1 — Multi-Agent Interactions",
        "coverage": {
            "theme_1_multi_agent": True,
            "fleet_ai_oversight": True,
            "halluminate_multi_actor": True,
            "theme_2_long_horizon": True,
            "scale_ai_non_code_business": True,
            "mercor_token_scaled_rewards": True,
            "theme_3_1_professional": True,
            "scaler_ai_multi_app_enterprise": True,
            "patronus_schema_drift": True,
            "theme_4_self_improvement": True,
            "snorkel_experts_in_loop": True,
        },
        "evidence_endpoints": {
            "halluminate": "/actors/messages",
            "patronus": "/policy/current",
            "mercor": "/reward/token_bonus",
            "snorkel": "/experts/current",
            "scaler_ai": "/tickets",
        },
        "rlvr_rlve": {
            "rlvr_verifiers": "graders/",
            "rlve_adaptive_environment": "scenarios/red_team_generator.py",
        },
        "reward_hacking_defenses": [
            "close_case_idempotency",
            "team_f1_delta_not_sticky",
            "zero_escalation_guard",
            "over_escalation_threshold",
            "manager_judge_fallback",
            "policy_drift_active_at_semantics",
        ],
    }


# ---------------------------------------------------------------------------
# Production operations endpoints — metrics, audit trail, session admin
# ---------------------------------------------------------------------------

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text-format operational metrics (requests, episodes, steps, rewards)."""
    return PlainTextResponse(
        METRICS.render(active_sessions=len(_sessions)),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/episodes")
def list_episodes(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    List recorded episode audit trails (newest first).

    Every reset/step is recorded per episode for replay and compliance review.
    Filter with ?session_id=... to scope to one tenant.
    """
    return {"episodes": AUDIT.list_episodes(session_id=session_id, limit=limit)}


@app.get("/episodes/{episode_id}/trace")
def episode_trace(episode_id: str, format: Literal["json", "jsonl"] = Query(default="json")):
    """
    Full audit trace for one episode: every action, reward, and running total.

    ?format=jsonl returns newline-delimited JSON suitable for piping into a
    SIEM or data lake.
    """
    trace = AUDIT.get(episode_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found in the audit window.")
    if format == "jsonl":
        return PlainTextResponse(trace_to_jsonl(trace), media_type="application/x-ndjson")
    return {**trace.summary(), "events": trace.events}


@app.get("/sessions")
def sessions_list():
    """List live sessions with their episode state (operator/debug endpoint)."""
    return {"count": len(_sessions), "sessions": _sessions.list_summaries()}


@app.delete("/sessions/{session_id}")
def sessions_delete(session_id: str):
    """Drop a session and free its episode state. The default session is recreated on demand."""
    if not _sessions.drop(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Application factory (for testing)
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app


def main():
    """Entry point for the SOC-Triage-Gym server (used by [project.scripts]).

    Host/port are read from the environment so the server can be pinned to
    localhost in untrusted settings, while keeping 0.0.0.0:7860 as the default
    for container / Hugging Face Spaces deployments (which require binding all
    interfaces).
    """
    import os

    import uvicorn

    host = os.environ.get("SOC_TRIAGE_HOST", "0.0.0.0")  # noqa: S104 - deploy default
    port = int(os.environ.get("SOC_TRIAGE_PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()


