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
import threading
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from actors import build_default_registry
from actors.registry import ActorRegistry
from baseline_agent import HeuristicBaselineAgent
from graders.expert_panel import ExpertPanel
from graders.token_scaled_reward import explain as explain_token_bonus, token_scaled_bonus
from models import (
    ActorMessage,
    AgentRole,
    AlertClassification,
    EnvironmentState,
    ExpertProfile,
    PolicyVersion,
    RedTeamConfig,
    RewardBlendConfig,
    SOCAction,
    SOCObservation,
    TicketSLA,
)
from scenarios.policy_drift import PolicyDriftEngine
from scenarios.red_team_generator import RedTeamGenerator
from server.landing_ui import UI_HTML
from server.page_ui import (
    render_metadata as _render_metadata_page,
    render_tasks as _render_tasks_page,
    render_themes as _render_themes_page,
    render_state as _render_state_page,
    render_schema as _render_schema_page,
)
from server.environment import SOCEnvironment
from tools.ticketing import TicketingSystem


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_env: Optional[SOCEnvironment] = None
_env_lock = threading.Lock()
_baseline_agent = HeuristicBaselineAgent()

# v3 theme-coverage modules — deterministic, per-episode, reset in /reset.
_actor_registry: ActorRegistry = build_default_registry(seed=0)
_policy_drift: PolicyDriftEngine = PolicyDriftEngine(seed=0)
_expert_panel: ExpertPanel = ExpertPanel()
_ticketing: TicketingSystem = TicketingSystem()
_reward_blend: RewardBlendConfig = RewardBlendConfig()
_current_expert: ExpertProfile = _expert_panel.for_round(0)
_curriculum_round: int = 0
_actor_step: int = 0

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
    """Initialize environment on startup."""
    global _env
    _env = SOCEnvironment()
    yield
    # Cleanup on shutdown (nothing needed for in-memory env)


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
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: str = "phishing"
    seed: int = 42
    mode: Literal["tier1_solo", "team"] = "tier1_solo"


class GenerateScenarioRequest(BaseModel):
    """Request body for POST /generate_scenario."""
    seed: int = 42
    difficulty_floor: float = 0.5
    attack_patterns: List[str] = ["phishing", "lateral_movement", "insider_threat"]
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
    return {"status": "healthy", "version": "0.1.0", "env": "soc-triage-gym"}


@app.get("/metadata")
def metadata():
    """Environment metadata (OpenEnv runtime spec)."""
    return {
        "name": "soc-triage-gym",
        "version": "0.1.0",
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
    from models import SOCAction, SOCObservation, EnvironmentState
    return {
        "action": SOCAction.model_json_schema(),
        "observation": SOCObservation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.post("/mcp")
def mcp_endpoint(request: Optional[dict] = Body(default=None)):
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
                with _env_lock:
                    obs = _env.reset(
                        task_id=tool_args.get("task_id", "phishing"),
                        seed=tool_args.get("seed", 42),
                        mode=tool_args.get("mode", "tier1_solo"),
                    )
                    return _jsonrpc_ok({"content": [{"type": "text", "text": obs.model_dump_json()}]})

            if tool_name == "state":
                with _env_lock:
                    st = _env.state()
                    return _jsonrpc_ok({"content": [{"type": "text", "text": st.model_dump_json()}]})

            if tool_name == "submit_investigation":
                with _env_lock:
                    if _env._config is None:
                        return _jsonrpc_err(-32602, "No active episode. Call reset first.")
                    obs = _env.step(SOCAction(action_type="submit_investigation"))
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
                with _env_lock:
                    if _env._config is None:
                        return _jsonrpc_err(-32602, "No active episode. Call reset first.")
                    if tool_name == "step":
                        action_data = tool_args
                    else:
                        action_data = {"action_type": ACTION_MAP[tool_name], **tool_args}
                    action = SOCAction(**action_data)
                    obs = _env.step(action)
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
def reset(request: Optional[ResetRequest] = Body(default=None)):
    """
    Start a new episode.

    Args:
        task_id: "phishing" | "lateral_movement" | "queue_management" | "insider_threat" (default: "phishing")
        seed: RNG seed for deterministic scenario generation (default: 42)

    Returns:
        Initial SOCObservation with full alert queue.
    """
    req = request or ResetRequest()
    with _env_lock:
        try:
            obs = _env.reset(task_id=req.task_id, seed=req.seed, mode=req.mode)
            # v3 theme hooks: reset actors, policy drift, ticketing, expert rotation.
            global _actor_registry, _policy_drift, _ticketing, _current_expert, _actor_step
            _actor_registry = build_default_registry(seed=req.seed)
            _actor_registry.reset(seed=req.seed)
            _policy_drift = PolicyDriftEngine(seed=req.seed)
            max_steps = _env._config.max_steps if _env._config else 60
            _policy_drift.plan(max_steps=max_steps, drift_count=2)
            _ticketing = TicketingSystem()
            _current_expert = _expert_panel.for_round(_curriculum_round)
            _actor_step = 0
            return obs
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Error in /reset")
            raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/step", response_model=SOCObservation)
def step(action: SOCAction):
    """
    Execute an action in the current episode.

    The action_type field determines which other fields are relevant.
    See the SOCAction model for the full action schema.

    Returns:
        Updated SOCObservation with step reward, new results, and done flag.
    """
    with _env_lock:
        if _env._config is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call POST /reset first.",
            )
        try:
            obs = _env.step(action)
            # v3 theme tick: advance actors, policy drift, ticketing SLA clocks.
            global _actor_step
            _actor_step += 1
            _actor_registry.tick(
                step=_actor_step,
                ctx={"policy_version": _policy_drift.current().version},
            )
            _policy_drift.maybe_drift(step=_actor_step)
            _ticketing.tick()
            return obs
        except Exception as e:
            logger.exception("Error in /step")
            raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/state", response_model=EnvironmentState)
def state():
    """
    Get current episode metadata without consuming a step.

    Returns episode_id, task_id, step_count, max_steps, done,
    cumulative_reward, alert_count, classified_count, seed.
    """
    with _env_lock:
        try:
            return _env.state()
        except Exception as e:
            logger.exception("Error in /state")
            raise HTTPException(status_code=500, detail="Internal server error.")


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
def ui_state():
    with _env_lock:
        if _env._config is None:
            _env.reset(task_id="phishing", seed=42)
        snap = _env.state()
    payload = snap.model_dump() if hasattr(snap, "model_dump") else dict(snap)
    return _render_state_page(payload)


@app.get("/ui/schema", response_class=HTMLResponse, include_in_schema=False)
def ui_schema():
    return _render_schema_page(schema())


# ---------------------------------------------------------------------------
# REST Tool Endpoints (for LLM agents using direct REST tool calls)
# ---------------------------------------------------------------------------

def _ensure_episode():
    """Auto-start a default episode if none is active."""
    if _env._config is None:
        _env.reset(task_id="phishing", seed=42)


@app.get("/tasks")
def get_tasks():
    """List all available tasks (OpenEnv spec endpoint)."""
    return {"tasks": TASKS}


@app.post("/grader")
def grader(request: Optional[ResetRequest] = Body(default=None)):
    """
    Run the grader on the current episode state (OpenEnv spec endpoint).

    Evaluates the current investigation state against ground truth and returns
    a normalized score in [0.0, 1.0]. Does not terminate the episode.
    """
    with _env_lock:
        _ensure_episode()
        try:
            score, breakdown, feedback = _env.grade_with_breakdown()
            return {
                "score": score,
                "breakdown": breakdown,
                "feedback": feedback,
                "task_id": _env._task_id,
                "steps_used": _env._step,
                "max_steps": _env._config.max_steps if _env._config else 0,
                "done": _env._done,
            }
        except Exception as e:
            logger.exception("Error in /grader")
            raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/baseline")
def baseline(request: Optional[ResetRequest] = Body(default=None)):
    """
    Run the heuristic baseline agent on a fresh episode (OpenEnv spec endpoint).

    Resets the environment with the specified task/seed, runs the built-in
    heuristic agent to completion, and returns the final score.
    """
    req = request or ResetRequest()
    with _env_lock:
        try:
            # Reset to a fresh episode
            _env.reset(task_id=req.task_id, seed=req.seed, mode=req.mode)
            _baseline_agent.reset()
            # Run heuristic steps until done
            steps = 0
            max_steps = _env._config.max_steps if _env._config else 0
            while not _env._done and steps < max_steps:
                obs = _env._build_observation(role=_env._current_role(), reward=0.0)
                action = SOCAction(**_baseline_agent.next_action(obs.model_dump()))
                _env.step(action)
                steps += 1
            # Grade the result with breakdown
            score, breakdown, feedback = _env.grade_with_breakdown()
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
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Error in /baseline")
            raise HTTPException(status_code=500, detail="Internal server error.")


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
def generate_scenario(request: Optional[GenerateScenarioRequest] = Body(default=None)):
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
    with _env_lock:
        _env.set_generated_scenario(scenario)
    return scenario.model_dump()


@app.get("/inbox/{role}")
def inbox(role: str):
    """Debug endpoint to inspect role-filtered tickets in the current episode."""
    with _env_lock:
        _ensure_episode()
        try:
            parsed_role = AgentRole(role)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        obs = _env._build_observation(role=parsed_role, reward=0.0)
        return {"role": role, "tickets": [ticket.model_dump() for ticket in obs.tickets]}


@app.get("/api/alerts")
def list_alerts(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    List alerts in the current episode queue.

    Returns paginated alert objects with their indicators and metadata.
    Call POST /reset first to start an episode, or a default phishing
    episode will be auto-started.
    """
    with _env_lock:
        _ensure_episode()
        alerts = [a.model_dump() for a in _env._config.alerts]
        total = len(alerts)
        page = alerts[offset: offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "alerts": page,
        }


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str):
    """
    Get full details for a single alert including indicators and metadata.
    """
    with _env_lock:
        _ensure_episode()
        for alert in _env._config.alerts:
            if alert.alert_id == alert_id:
                inv = _env._investigations.get(alert_id)
                return {
                    "alert": alert.model_dump(),
                    "investigation": inv.model_dump() if inv else None,
                }
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.get("/threat-intel/ip/{ip}")
def threat_intel_ip(ip: str):
    """
    Look up threat intelligence for an IP address.

    Returns enrichment data including malicious status, reputation score,
    associated threat actors, and related indicators.
    """
    with _env_lock:
        _ensure_episode()
        db = _env._config.enrichment_db
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
def threat_intel_domain(domain: str):
    """
    Look up threat intelligence for a domain name.

    Returns enrichment data including malicious status, category,
    registrar info, and associated indicators.
    """
    with _env_lock:
        _ensure_episode()
        db = _env._config.enrichment_db
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
def threat_intel_hash(file_hash: str):
    """
    Look up threat intelligence for a file hash (MD5, SHA-1, SHA-256).

    Returns enrichment data including malware family, AV detection rate,
    and associated campaigns.
    """
    with _env_lock:
        _ensure_episode()
        db = _env._config.enrichment_db
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
    alert_id: Optional[str] = Query(default=None),
    hours: int = Query(default=24, ge=1, le=168),
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
    with _env_lock:
        _ensure_episode()
        log_db = _env._config.log_db
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
def actor_messages(role: Optional[str] = Query(default=None)):
    """
    Halluminate sub-theme — Inspect messages from external NPC actors
    (ThreatIntelFeed, ComplianceOfficer, EndUserReporter) in the current episode.
    """
    with _env_lock:
        if role is None:
            msgs = _actor_registry.all_messages()
            return {"count": len(msgs), "messages": [m.model_dump() for m in msgs]}
        try:
            parsed = AgentRole(role)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        msgs = _actor_registry.inbox_for(parsed)
        return {"role": role, "count": len(msgs), "messages": [m.model_dump() for m in msgs]}


@app.get("/policy/current")
def policy_current():
    """Patronus sub-theme — Current active policy version."""
    with _env_lock:
        return _policy_drift.current().model_dump()


@app.get("/policy/history")
def policy_history():
    """Patronus sub-theme — Full policy-drift history for this episode."""
    with _env_lock:
        return _policy_drift.to_dict()


@app.get("/reward/config")
def reward_config():
    """Mercor sub-theme — Active reward blend config (role/team/token weights)."""
    with _env_lock:
        return _reward_blend.model_dump()


class RewardBlendUpdate(BaseModel):
    """Optional blend override."""
    role_weight: Optional[float] = None
    team_weight: Optional[float] = None
    token_scale_enabled: Optional[bool] = None
    token_scale_floor: Optional[int] = None
    token_scale_cap: Optional[int] = None
    token_scale_max_bonus: Optional[float] = None


@app.post("/reward/config")
def reward_config_update(patch: RewardBlendUpdate):
    """Patch the reward blend config. Returns the new config."""
    with _env_lock:
        fields = patch.model_dump(exclude_none=True)
        for k, v in fields.items():
            setattr(_reward_blend, k, v)
        return _reward_blend.model_dump()


class TokenBonusRequest(BaseModel):
    text: str
    content_quality: float = 0.5


@app.post("/reward/token_bonus")
def reward_token_bonus(req: TokenBonusRequest):
    """
    Compute the Mercor token-length bonus for a given text and quality gate.
    Surfaced as an endpoint so agents / judges can preview the incentive curve.
    """
    with _env_lock:
        return explain_token_bonus(req.text, req.content_quality, _reward_blend)


@app.get("/experts/current")
def experts_current():
    """Snorkel sub-theme — Active reviewing expert and their preference hint."""
    with _env_lock:
        return {
            "round": _curriculum_round,
            "expert": _current_expert.model_dump(),
            "hint": _expert_panel.hint_message(_current_expert),
        }


@app.get("/experts/panel")
def experts_panel():
    """Snorkel sub-theme — Full expert panel roster."""
    return {"panel": [e.model_dump() for e in _expert_panel.all_profiles()]}


class ExpertRotateRequest(BaseModel):
    round_index: Optional[int] = None


@app.post("/experts/rotate")
def experts_rotate(req: Optional[ExpertRotateRequest] = Body(default=None)):
    """
    Advance the expert rotation. If round_index is given, rotate to that round;
    otherwise increment by 1. Emulates Snorkel experts-in-the-loop curriculum.
    """
    global _curriculum_round, _current_expert
    with _env_lock:
        if req is not None and req.round_index is not None:
            _curriculum_round = int(req.round_index)
        else:
            _curriculum_round += 1
        _current_expert = _expert_panel.for_round(_curriculum_round)
        return {
            "round": _curriculum_round,
            "expert": _current_expert.model_dump(),
        }


class TicketOpenRequest(BaseModel):
    alert_id: str
    priority: str = "P3"
    note: str = ""


@app.post("/tickets/open")
def tickets_open(req: TicketOpenRequest):
    """Scaler AI Labs sub-theme — Open a multi-app enterprise ticket."""
    with _env_lock:
        t = _ticketing.open(alert_id=req.alert_id, priority=req.priority, note=req.note)
        return t.model_dump()


@app.post("/tickets/{ticket_id}/resolve")
def tickets_resolve(ticket_id: str, note: str = ""):
    with _env_lock:
        t = _ticketing.resolve(ticket_id=ticket_id, note=note)
        if t is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return t.model_dump()


@app.get("/tickets")
def tickets_list():
    """List all tickets in the current episode."""
    with _env_lock:
        return {
            "tickets": [t.model_dump() for t in _ticketing.all_tickets()],
            "audit": _ticketing.audit_summary(),
        }


@app.get("/tickets/can_disable_user")
def tickets_can_disable_user(alert_id: str):
    """Cross-app business rule: can IAM.disable_user fire for this alert?"""
    with _env_lock:
        return {"alert_id": alert_id, "allowed": _ticketing.can_disable_user(alert_id)}


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
# Application factory (for testing)
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Return the FastAPI application instance."""
    return app


def main():
    """Entry point for the SOC-Triage-Gym server (used by [project.scripts])."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()


