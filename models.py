"""
SOC-Triage-Gym Pydantic Models
================================
All Pydantic v2 models for the SOC-Triage-Gym environment.
This module is the central contract — every other module imports from here.

Public (exposed to agent): SOCAction, SOCObservation, SOCReward, and all sub-models.
Internal (not sent to agent): GroundTruth, ScenarioConfig.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    """Severity levels for SIEM alerts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertClassification(str, Enum):
    """Classification outcomes for a security alert."""
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    BENIGN_TRUE_POSITIVE = "benign_true_positive"
    UNCLASSIFIED = "unclassified"


class IndicatorType(str, Enum):
    """Types of threat indicators that can be enriched."""
    IP = "ip"
    DOMAIN = "domain"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    URL = "url"
    USER = "user"


class LogSource(str, Enum):
    """Available SIEM log sources to query."""
    FIREWALL = "firewall"
    PROXY = "proxy"
    DNS = "dns"
    ENDPOINT = "endpoint"
    AUTH = "auth"
    EMAIL_GATEWAY = "email_gateway"
    IDS = "ids"
    CLOUD_TRAIL = "cloud_trail"


class CorrelationType(str, Enum):
    """How two alerts are correlated."""
    SOURCE_IP = "source_ip"
    DESTINATION_IP = "destination_ip"
    USER = "user"
    TECHNIQUE = "technique"
    TIME_WINDOW = "time_window"
    HOSTNAME = "hostname"
    FILE_HASH = "file_hash"
    DOMAIN = "domain"


class ResponseActionType(str, Enum):
    """Containment and response actions available to the analyst."""
    ISOLATE_ENDPOINT = "isolate_endpoint"
    DISABLE_ACCOUNT = "disable_account"
    BLOCK_IP = "block_ip"
    BLOCK_DOMAIN = "block_domain"
    QUARANTINE_FILE = "quarantine_file"
    RESET_PASSWORD = "reset_password"
    REVOKE_SESSIONS = "revoke_sessions"
    NO_ACTION = "no_action"


class ActionType(str, Enum):
    """All actions available to the SOC analyst agent."""
    # --- Tier-1 Analyst actions (original) ---
    ENRICH_INDICATOR = "enrich_indicator"
    QUERY_LOGS = "query_logs"
    CORRELATE_ALERTS = "correlate_alerts"
    CHECK_ASSET = "check_asset"
    CHECK_USER = "check_user"
    CLASSIFY_ALERT = "classify_alert"
    MAP_TECHNIQUE = "map_technique"
    RECOMMEND_ACTION = "recommend_action"
    ESCALATE = "escalate"
    SUBMIT_INVESTIGATION = "submit_investigation"
    NOOP = "noop"
    # --- Tier-1 multi-agent extension ---
    ESCALATE_TO_TIER2 = "escalate_to_tier2"
    PHASE_COMPLETE = "phase_complete"
    # --- Tier-2 Responder actions ---
    FORENSIC_TIMELINE = "forensic_timeline"
    SANDBOX_DETONATE = "sandbox_detonate"
    MEMORY_ANALYSIS = "memory_analysis"
    ISOLATE_HOST = "isolate_host"
    DISABLE_USER = "disable_user"
    BLOCK_IOC = "block_ioc"
    CLOSE_CASE = "close_case"
    # --- SOC Manager (Oversight) actions ---
    REVIEW_DECISION = "review_decision"
    OVERRIDE_CLASSIFICATION = "override_classification"
    FLAG_INCONSISTENCY = "flag_inconsistency"
    EXPLAIN_TEAM_BEHAVIOR = "explain_team_behavior"


class AgentRole(str, Enum):
    """Agent roles in the multi-tier SOC environment."""
    TIER1 = "tier1"
    TIER2 = "tier2"
    MANAGER = "manager"
    RED_TEAM = "red_team"


class EpisodeMode(str, Enum):
    """Episode operation mode."""
    TIER1_SOLO = "tier1_solo"
    TEAM = "team"


class EpisodePhase(str, Enum):
    """Current phase of a team-mode episode."""
    TRIAGE = "triage"       # Tier-1 analyst phase
    RESPONSE = "response"   # Tier-2 responder phase
    OVERSIGHT = "oversight" # Manager review phase
    COMPLETE = "complete"   # Episode finished


class TicketKind(str, Enum):
    """Type of inter-agent ticket message."""
    ESCALATION = "escalation"
    OVERRIDE = "override"
    FLAG = "flag"
    CLOSURE = "closure"
    REVIEW_REQUEST = "review_request"


# ---------------------------------------------------------------------------
# Multi-Agent Models
# ---------------------------------------------------------------------------

class TicketMessage(BaseModel):
    """Structured message passed between agent roles via the message bus."""

    model_config = ConfigDict(frozen=False)

    ticket_id: str = Field(..., description="Unique ticket identifier")
    alert_id: str = Field(..., description="Alert this ticket concerns")
    from_role: AgentRole = Field(..., description="Role that created this ticket")
    to_role: AgentRole | None = Field(None, description="Intended recipient role (None = broadcast)")
    kind: TicketKind = Field(..., description="Ticket type")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Ticket contents: reasoning, classification decision, override, evidence refs, etc."
    )
    step_created: int = Field(..., description="Episode step when ticket was created")
    resolved: bool = Field(default=False, description="Whether ticket has been acted on")


class ContainmentResult(BaseModel):
    """Result of a Tier-2 containment or forensic action."""

    model_config = ConfigDict(frozen=False)

    action_type: str = Field(..., description="The containment action performed")
    target: str = Field(..., description="Target of the action (host, user, IOC)")
    success: bool = Field(..., description="Whether the action succeeded")
    details: str = Field(..., description="Human-readable result details")
    evidence: list[str] = Field(default_factory=list, description="Forensic evidence items found")
    timeline_entries: list[str] = Field(default_factory=list, description="Forensic timeline entries")


class ManagerReviewResult(BaseModel):
    """Result of a SOC Manager oversight action."""

    model_config = ConfigDict(frozen=False)

    action_type: str = Field(..., description="The oversight action performed")
    ticket_id: str | None = Field(None, description="Ticket reviewed")
    alert_id: str | None = Field(None, description="Alert assessed")
    finding: str = Field(..., description="Manager's finding or decision")
    override_applied: bool = Field(default=False, description="Whether a classification was overridden")
    inconsistency_found: bool = Field(default=False, description="Whether an inconsistency was flagged")
    explanation: str | None = Field(None, description="Manager's explanation of team behavior")


class TeamRewardBreakdown(BaseModel):
    """Per-role reward breakdown for a team episode step."""

    model_config = ConfigDict(frozen=False)

    tier1_individual: float = Field(default=0.0, description="Tier-1 individual contribution")
    tier2_individual: float = Field(default=0.0, description="Tier-2 individual contribution")
    manager_individual: float = Field(default=0.0, description="Manager individual contribution")
    team_shared: float = Field(default=0.0, description="Shared team F1 component")
    total: float = Field(default=0.0, description="Combined reward")


class ConsistencyStats(BaseModel):
    """Historical consistency stats visible to the SOC Manager."""

    model_config = ConfigDict(frozen=False)

    tickets_total: int = Field(default=0, description="Total tickets created so far")
    tickets_resolved: int = Field(default=0, description="Tickets marked resolved")
    escalation_reviews: int = Field(default=0, description="Escalation tickets reviewed by Manager")
    closure_reviews: int = Field(default=0, description="Closure tickets reviewed by Manager")
    overrides_total: int = Field(default=0, description="Override tickets created by Manager")
    valid_flags: int = Field(default=0, description="Flags that matched ground-truth inconsistencies")
    invalid_flags: int = Field(default=0, description="Flags that did not match ground truth")
    consistency_rate: float = Field(default=0.0, description="Share of reviewed/flagged items judged consistent")


# ---------------------------------------------------------------------------
# Observation Sub-models
# ---------------------------------------------------------------------------

class AlertMeta(BaseModel):
    """Metadata for a single SIEM alert visible to the agent."""

    model_config = ConfigDict(frozen=False)

    alert_id: str = Field(..., description="Unique alert identifier")
    title: str = Field(..., description="Human-readable alert title")
    description: str = Field(..., description="Detailed alert description")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    source_system: str = Field(..., description="SIEM source system (e.g. 'Email Security', 'EDR')")
    timestamp: str = Field(..., description="ISO8601 timestamp when alert fired")
    rule_triggered: str = Field(..., description="Detection rule name that triggered this alert")
    indicators: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Threat indicators grouped by type: {ip: [...], domain: [...], file_hash: [...], email: [...], url: [...]}"
    )
    raw_log_snippet: str | None = Field(None, description="Raw log excerpt that triggered the alert")
    related_alert_ids: list[str] = Field(default_factory=list, description="Alert IDs that may be related")
    classification: AlertClassification = Field(
        default=AlertClassification.UNCLASSIFIED,
        description="Current classification (starts as unclassified)"
    )


class EnrichmentResult(BaseModel):
    """Threat intelligence enrichment result for a single indicator."""

    model_config = ConfigDict(frozen=False)

    indicator: str = Field(..., description="The indicator value that was enriched")
    indicator_type: IndicatorType = Field(..., description="Type of indicator")
    malicious: bool = Field(..., description="Whether indicator is known malicious")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    threat_score: int = Field(default=0, ge=0, le=100, description="Threat score 0-100")
    threat_type: str | None = Field(None, description="Threat category: phishing, c2, spam, malware, etc.")
    first_seen: str | None = Field(None, description="ISO8601 date first seen in threat feeds")
    last_seen: str | None = Field(None, description="ISO8601 date last seen in threat feeds")
    geo_location: str | None = Field(None, description="Country/region for IP indicators")
    whois_info: str | None = Field(None, description="WHOIS registration summary")
    associated_malware: list[str] = Field(default_factory=list, description="Associated malware family names")
    tags: list[str] = Field(default_factory=list, description="Threat intelligence tags")
    source: str = Field(default="threat_intel", description="Threat intel source name")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Additional raw intel data")


class LogEntry(BaseModel):
    """A single log event returned from a SIEM query."""

    model_config = ConfigDict(frozen=False)

    timestamp: str = Field(..., description="ISO8601 event timestamp")
    source: LogSource = Field(..., description="Log source system")
    event_type: str = Field(..., description="Event classification (e.g. 'email_received', 'process_created')")
    src_ip: str | None = Field(None, description="Source IP address")
    dst_ip: str | None = Field(None, description="Destination IP address")
    user: str | None = Field(None, description="Username associated with event")
    hostname: str | None = Field(None, description="Hostname where event occurred")
    action: str | None = Field(None, description="Action taken (allow, block, execute, etc.)")
    severity: str | None = Field(None, description="Event severity")
    details: dict[str, Any] = Field(default_factory=dict, description="Source-specific event details")
    raw: str | None = Field(None, description="Raw log line")


class CorrelatedEvent(BaseModel):
    """A correlation link found between two or more alerts."""

    model_config = ConfigDict(frozen=False)

    alert_ids: list[str] = Field(..., description="IDs of correlated alerts")
    correlation_type: CorrelationType = Field(..., description="How these alerts are correlated")
    shared_indicator: str = Field(..., description="The shared indicator value that links them")
    description: str = Field(..., description="Human-readable description of the correlation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Correlation confidence 0.0-1.0")
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this correlation is to the investigation")


class AssetInfo(BaseModel):
    """Asset inventory information for a host."""

    model_config = ConfigDict(frozen=False)

    asset_id: str = Field(..., description="Unique asset identifier")
    hostname: str = Field(..., description="Asset hostname")
    asset_type: str = Field(..., description="Asset type: workstation, server, domain_controller, etc.")
    criticality: str = Field(..., description="Business criticality: critical, high, medium, low")
    owner: str = Field(..., description="Username of asset owner/primary user")
    department: str = Field(..., description="Owning department")
    ip_address: str = Field(..., description="Primary IP address")
    os: str | None = Field(None, description="Operating system")
    patch_status: str | None = Field(None, description="Patch compliance status")
    last_scan: str | None = Field(None, description="ISO8601 date of last security scan")
    open_vulnerabilities: int = Field(default=0, description="Number of open CVEs")
    recent_activity_summary: str = Field(default="", description="Brief summary of recent activity")
    tags: list[str] = Field(default_factory=list, description="Asset tags")


class UserInfo(BaseModel):
    """User profile information from directory services."""

    model_config = ConfigDict(frozen=False)

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Login username (samaccountname)")
    display_name: str = Field(..., description="Full display name")
    email: str = Field(..., description="Email address")
    role: str = Field(..., description="Job title/role")
    department: str = Field(..., description="Department")
    access_level: str = Field(..., description="Access tier: standard, elevated, admin, service")
    is_privileged: bool = Field(default=False, description="Has privileged/admin access")
    manager: str | None = Field(None, description="Manager username")
    last_login: str | None = Field(None, description="ISO8601 datetime of last login")
    login_anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Anomaly score for login patterns")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="User risk score from UEBA")
    recent_actions: list[str] = Field(default_factory=list, description="Recent notable actions")


class InvestigationState(BaseModel):
    """
    Tracks all agent actions for a single alert during the episode.
    This is the primary data structure that graders evaluate.
    """

    model_config = ConfigDict(frozen=False)

    alert_id: str = Field(..., description="Alert being investigated")
    enriched_indicators: dict[str, EnrichmentResult] = Field(
        default_factory=dict,
        description="Indicators enriched: {indicator_value: EnrichmentResult}"
    )
    queried_sources: dict[str, list[LogEntry]] = Field(
        default_factory=dict,
        description="Log sources queried: {source_name: [LogEntry, ...]}"
    )
    correlations_found: list[CorrelatedEvent] = Field(
        default_factory=list,
        description="Correlations discovered involving this alert"
    )
    assets_looked_up: dict[str, AssetInfo] = Field(
        default_factory=dict,
        description="Assets investigated: {hostname: AssetInfo}"
    )
    users_looked_up: dict[str, UserInfo] = Field(
        default_factory=dict,
        description="Users investigated: {username: UserInfo}"
    )
    classification: AlertClassification | None = Field(
        None, description="Agent's classification decision"
    )
    classification_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Agent's confidence in classification"
    )
    mapped_techniques: list[str] = Field(
        default_factory=list,
        description="MITRE ATT&CK technique IDs mapped by agent (e.g. T1566.001)"
    )
    recommended_actions: list[ResponseActionType] = Field(
        default_factory=list,
        description="Response actions recommended by agent"
    )
    escalated: bool = Field(default=False, description="Whether alert was escalated")
    escalation_severity: str | None = Field(None, description="Escalation severity if escalated")
    escalation_justification: str | None = Field(None, description="Reason for escalation")
    evidence_timeline: list[str] = Field(
        default_factory=list,
        description="Chronological list of evidence gathered (human-readable)"
    )
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Step-by-step reward contributions"
    )


# ---------------------------------------------------------------------------
# Top-level Action Model
# ---------------------------------------------------------------------------

class SOCAction(BaseModel):
    """
    Flat action model for the SOC analyst agent.
    The action_type field determines which optional fields are relevant.

    Example actions:
        {"action_type": "enrich_indicator", "indicator": "1.2.3.4", "indicator_type": "ip"}
        {"action_type": "query_logs", "log_source": "firewall", "query_alert_id": "ALT-001"}
        {"action_type": "classify_alert", "alert_id": "ALT-001", "classification": "true_positive"}
        {"action_type": "submit_investigation"}
        {"action_type": "noop"}
    """

    model_config = ConfigDict(frozen=False)

    action_type: ActionType = Field(..., description="Type of action to perform")

    # --- enrich_indicator params ---
    indicator: str | None = Field(None, description="Indicator value to enrich (IP, domain, hash, email, URL)")
    indicator_type: IndicatorType | None = Field(None, description="Type of indicator being enriched")

    # --- query_logs params ---
    log_source: LogSource | None = Field(None, description="Log source to query")
    query_alert_id: str | None = Field(None, description="Alert ID providing context for log query")
    time_window_hours: int | None = Field(24, description="Time window for log query in hours (default 24)")

    # --- correlate_alerts params ---
    alert_id_a: str | None = Field(None, description="First alert ID for correlation check")
    alert_id_b: str | None = Field(None, description="Second alert ID for correlation check")

    # --- check_asset params ---
    hostname: str | None = Field(None, description="Hostname to look up in asset inventory")

    # --- check_user params ---
    username: str | None = Field(None, description="Username to look up in user directory")

    # --- classify_alert params ---
    alert_id: str | None = Field(None, description="Alert ID to classify")
    classification: AlertClassification | None = Field(None, description="Classification decision")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Classification confidence 0.0-1.0")

    # --- map_technique params ---
    technique_id: str | None = Field(None, description="MITRE ATT&CK technique ID (e.g. T1566.001)")

    # --- recommend_action params ---
    response_action: ResponseActionType | None = Field(None, description="Recommended containment action")
    action_target: str | None = Field(None, description="Target of the response action (IP, hostname, username, etc.)")

    # --- escalate / escalate_to_tier2 params ---
    escalation_severity: str | None = Field(None, description="Escalation severity: critical or high")
    justification: str | None = Field(None, description="Justification for escalation")

    # --- Multi-agent role field (team mode) ---
    role: AgentRole | None = Field(None, description="Role of the agent submitting this action (team mode)")

    # --- Tier-2 Responder action params ---
    target_host: str | None = Field(None, description="Target hostname for containment/forensic actions")
    target_user: str | None = Field(None, description="Target username for user-related actions")
    target_ioc: str | None = Field(None, description="IOC value to block (IP, domain, hash)")
    ioc_type: str | None = Field(None, description="IOC type: ip, domain, file_hash")

    # --- SOC Manager action params ---
    ticket_id: str | None = Field(None, description="Ticket ID to review or reference")
    new_classification: AlertClassification | None = Field(None, description="Override classification for override_classification action")
    flag_reason: str | None = Field(None, description="Reason for flagging an inconsistency")
    explanation_text: str | None = Field(None, description="Manager's explanation of team behavior")


# ---------------------------------------------------------------------------
# Top-level Observation Model
# ---------------------------------------------------------------------------

class SOCObservation(BaseModel):
    """
    Full observation returned to the agent after each step() or reset() call.
    Contains the current alert queue, investigation state, and all results
    from the most recent action.
    """

    model_config = ConfigDict(frozen=False)

    # Alert queue
    alert_queue: list[AlertMeta] = Field(
        default_factory=list,
        description="All alerts for this episode. Severity and indicators visible to agent."
    )

    # Investigation state (per-alert tracking)
    investigations: dict[str, InvestigationState] = Field(
        default_factory=dict,
        description="Per-alert investigation state keyed by alert_id"
    )

    # Results from most recent action (populated after each step)
    enrichment_results: list[EnrichmentResult] = Field(
        default_factory=list,
        description="Threat intel results from most recent enrich_indicator action"
    )
    log_results: list[LogEntry] = Field(
        default_factory=list,
        description="Log entries from most recent query_logs action"
    )
    correlated_events: list[CorrelatedEvent] = Field(
        default_factory=list,
        description="All correlations discovered so far this episode"
    )
    asset_info: AssetInfo | None = Field(None, description="Asset info from most recent check_asset action")
    user_info: UserInfo | None = Field(None, description="User info from most recent check_user action")

    # Episode metadata
    investigation_budget: int = Field(..., description="Remaining steps before forced termination")
    step: int = Field(..., description="Current step number (0-indexed)")
    done: bool = Field(..., description="True when episode has ended")
    reward: float = Field(..., description="Reward earned in this step")
    cumulative_reward: float = Field(..., description="Total reward accumulated this episode")
    message: str = Field(default="", description="Human-readable status message from environment")

    # Task context
    task_id: str | None = Field(None, description="Active task ID")
    episode_id: str | None = Field(None, description="Unique episode identifier")

    # Final normalized task score (0,1) — populated after submit_investigation
    task_score: float | None = Field(None, description="Normalized grader score in (0,1) after episode ends")

    # --- Multi-agent fields (team mode only; None in tier1_solo mode) ---
    episode_mode: EpisodeMode = Field(
        default=EpisodeMode.TIER1_SOLO,
        description="Episode operation mode: tier1_solo or team"
    )
    current_role: AgentRole | None = Field(
        None, description="Which role should act next in team mode"
    )
    current_phase: EpisodePhase | None = Field(
        None, description="Current episode phase in team mode"
    )
    phase_steps_remaining: int | None = Field(
        None, description="Steps remaining in the current phase (team mode)"
    )
    tickets: list[TicketMessage] = Field(
        default_factory=list,
        description="Role-filtered ticket messages visible to the current agent"
    )
    containment_results: list[ContainmentResult] = Field(
        default_factory=list,
        description="Results from most recent Tier-2 containment/forensic action"
    )
    manager_review_result: ManagerReviewResult | None = Field(
        None, description="Result from most recent Manager oversight action"
    )
    team_reward_breakdown: TeamRewardBreakdown | None = Field(
        None, description="Per-role reward breakdown (team mode)"
    )
    consistency_stats: ConsistencyStats | None = Field(
        None, description="Historical consistency stats visible to the Manager"
    )


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class SOCReward(BaseModel):
    """Detailed reward breakdown for a single step."""

    model_config = ConfigDict(frozen=False)

    total: float = Field(..., description="Total step reward")
    enrichment_reward: float = Field(default=0.0, description="Reward from indicator enrichment")
    log_query_reward: float = Field(default=0.0, description="Reward from log queries")
    correlation_reward: float = Field(default=0.0, description="Reward from alert correlations")
    classification_reward: float = Field(default=0.0, description="Reward from classification actions")
    response_reward: float = Field(default=0.0, description="Reward from response recommendations")
    efficiency_penalty: float = Field(default=0.0, description="Penalty for inefficient actions")
    missed_tp_penalty: float = Field(default=0.0, description="Penalty for missing true positives")
    final_grader_reward: float = Field(default=0.0, description="Final grader score contribution")
    explanation: str = Field(default="", description="Human-readable explanation of reward")


# ---------------------------------------------------------------------------
# Internal Models (NOT exposed to agent)
# ---------------------------------------------------------------------------

class GroundTruth(BaseModel):
    """
    Answer key for a scenario. Stored server-side only.
    Never serialized in the Observation returned to the agent.
    """

    model_config = ConfigDict(frozen=False)

    alert_classifications: dict[str, AlertClassification] = Field(
        default_factory=dict,
        description="Correct classification for each alert_id"
    )
    true_positive_ids: list[str] = Field(default_factory=list, description="Alert IDs that are true positives")
    false_positive_ids: list[str] = Field(default_factory=list, description="Alert IDs that are false positives")
    benign_tp_ids: list[str] = Field(default_factory=list, description="Alert IDs that are benign true positives")
    expected_techniques: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Expected MITRE ATT&CK techniques per alert_id: {alert_id: [T1566.001, ...]}"
    )
    expected_response_actions: dict[str, list[ResponseActionType]] = Field(
        default_factory=dict,
        description="Expected response actions per alert_id"
    )
    kill_chain_order: list[str] | None = Field(
        None,
        description="Ordered alert IDs forming the attack kill chain (lateral movement and queue tasks)"
    )
    relevant_log_sources: dict[str, list[LogSource]] = Field(
        default_factory=dict,
        description="Log sources that contain relevant evidence per alert_id"
    )
    relevant_indicators: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Indicator values worth enriching per alert_id"
    )
    attack_chain_ids: list[list[str]] = Field(
        default_factory=list,
        description="For queue management: list of attack chains, each a list of related alert IDs"
    )
    # --- Team-mode escalation ground truth ---
    required_escalations: list[str] = Field(
        default_factory=list,
        description="Alert IDs that Tier-1 should escalate to Tier-2 in team mode"
    )
    required_containments: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Expected Tier-2 containment actions per alert_id: {alert_id: [action_type, ...]}"
    )
    expected_manager_flags: list[str] = Field(
        default_factory=list,
        description="Alert IDs where the Manager should flag an inconsistency"
    )


class ScenarioConfig(BaseModel):
    """
    Complete scenario configuration including all simulated data.
    Stored server-side and used to answer all tool queries.
    Never sent to the agent.
    """

    model_config = ConfigDict(frozen=False)

    scenario_id: str = Field(..., description="Unique scenario identifier")
    task_id: str = Field(..., description="Task this scenario belongs to")
    seed: int = Field(..., description="RNG seed used to generate this scenario")
    description: str = Field(..., description="Human-readable scenario description")
    max_steps: int = Field(..., description="Maximum steps before episode termination")
    alerts: list[AlertMeta] = Field(..., description="All alerts presented to agent")
    enrichment_db: dict[str, EnrichmentResult] = Field(
        default_factory=dict,
        description="Threat intel database: {indicator_value: EnrichmentResult}"
    )
    log_db: dict[str, dict[str, list[LogEntry]]] = Field(
        default_factory=dict,
        description="Log database: {source_name: {alert_id: [LogEntry, ...]}}"
    )
    asset_db: dict[str, AssetInfo] = Field(
        default_factory=dict,
        description="Asset inventory: {hostname: AssetInfo}"
    )
    user_db: dict[str, UserInfo] = Field(
        default_factory=dict,
        description="User directory: {username: UserInfo}"
    )
    ground_truth: GroundTruth = Field(..., description="Answer key — never exposed to agent")
    # --- Red-Team difficulty parameters ---
    difficulty_floor: float = Field(
        default=0.5,
        description="Red-Team difficulty floor [0.0, 1.0]: higher = more obfuscated scenarios"
    )
    noise_density: float = Field(
        default=0.6,
        description="Fraction of alerts that are FP/noise [0.0, 1.0]"
    )
    ioc_freshness: float = Field(
        default=0.7,
        description="Fraction of IOCs in threat feeds [0.0=all stale, 1.0=all fresh]"
    )
    correlation_obfuscation: float = Field(
        default=0.3,
        description="How obfuscated correlation signals are [0.0=clear, 1.0=fully hidden]"
    )


# ---------------------------------------------------------------------------
# Red-Team Configuration Model
# ---------------------------------------------------------------------------

class RedTeamConfig(BaseModel):
    """Configuration for the Red-Team Generator's curriculum."""

    model_config = ConfigDict(frozen=False)

    difficulty_floor: float = Field(default=0.5, ge=0.0, le=1.0, description="Base difficulty [0,1]")
    attack_patterns: list[str] = Field(
        default_factory=lambda: ["phishing", "lateral_movement", "insider_threat"],
        description="Attack patterns to generate scenarios for"
    )
    noise_density: float = Field(default=0.6, ge=0.0, le=1.0, description="FP fraction in generated scenarios")
    ioc_freshness: float = Field(default=0.7, ge=0.0, le=1.0, description="IOC threat-feed freshness")
    correlation_obfuscation: float = Field(default=0.3, ge=0.0, le=1.0, description="Correlation signal obscurity")
    blue_team_win_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Recent blue-team win rate for curriculum adaptation")
    episode_count: int = Field(default=0, description="Total episodes generated so far")


# ---------------------------------------------------------------------------
# Environment State Model
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v3 Theme-Coverage Models (Halluminate / Patronus / Mercor / Snorkel)
# ---------------------------------------------------------------------------

class ActorKind(str, Enum):
    """External (non-learning) simulated actors the agent must coordinate with."""
    THREAT_INTEL = "threat_intel"
    COMPLIANCE = "compliance"
    END_USER = "end_user"


class ActorMessage(BaseModel):
    """
    Unsolicited message from an external NPC actor (Halluminate sub-theme).
    Surfaces in /inbox/{role} alongside TicketMessage.
    """
    model_config = ConfigDict(frozen=False)

    message_id: str = Field(..., description="Unique message identifier")
    actor: ActorKind = Field(..., description="Which external actor produced the message")
    to_role: AgentRole | None = Field(None, description="Intended recipient role (None = any)")
    subject: str = Field(..., description="Short subject line")
    body: str = Field(..., description="Full message body")
    step_created: int = Field(..., description="Episode step when message was created")
    requires_response: bool = Field(default=False, description="Whether the agent should answer back")
    ground_truth_relevant: bool = Field(
        default=False,
        description="Whether engaging with this message is rewarded (True) or a distractor (False)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Actor-specific payload")


class PolicyVersion(BaseModel):
    """
    Active policy/schema version in effect at a given step (Patronus sub-theme).
    When the policy changes mid-episode, the grader must honour whichever policy
    was active at action time.
    """
    model_config = ConfigDict(frozen=False)

    version: int = Field(default=1, description="Monotonic policy version number")
    step_activated: int = Field(default=0, description="Episode step at which this policy took effect")
    severity_threshold_high: float = Field(
        default=7.5,
        description="CVSS threshold for 'high' severity — may drift mid-episode"
    )
    field_rename_map: dict[str, str] = Field(
        default_factory=dict,
        description="Field renames in effect, e.g. {'src_ip': 'source_address'}"
    )
    admin_must_escalate: bool = Field(
        default=False,
        description="Policy toggle: all admin-account alerts must be escalated"
    )
    description: str = Field(default="v1 baseline", description="Human-readable change summary")


class RewardBlendConfig(BaseModel):
    """
    Reward blend configuration (Mercor sub-theme includes token scaling).
    Exposed via POST /reward_config so judges can see the blend.
    """
    model_config = ConfigDict(frozen=False)

    role_weight: float = Field(default=0.6, description="Weight on role-specific reward")
    team_weight: float = Field(default=0.4, description="Weight on Δteam_F1 component")
    token_scale_enabled: bool = Field(
        default=True,
        description="If True, long-form free-text actions earn a bounded length-scaled bonus"
    )
    token_scale_floor: int = Field(default=20, description="Below this many tokens, length bonus is 0")
    token_scale_cap: int = Field(default=400, description="Saturation cap — beyond this, no extra reward")
    token_scale_max_bonus: float = Field(
        default=0.10,
        description="Maximum additional reward from the token-quality scaler (Mercor)"
    )


class ExpertProfile(BaseModel):
    """A simulated expert reviewer with fixed-per-round rubric weights (Snorkel sub-theme)."""
    model_config = ConfigDict(frozen=False)

    expert_id: str = Field(..., description="Stable expert identifier")
    display_name: str = Field(..., description="Expert nickname shown to agent")
    weight_accuracy: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_reasoning: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_actionability: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_speed: float = Field(default=0.0, ge=0.0, le=1.0, description="Bonus for short step-count")
    weight_thoroughness: float = Field(default=0.0, ge=0.0, le=1.0, description="Bonus for wide tool coverage")
    personality_note: str = Field(
        default="",
        description="Free-text hint surfaced to the agent so it can infer preference"
    )


class TicketSLA(BaseModel):
    """Ticket in the enterprise ticketing app (Scaler AI Labs sub-theme)."""
    model_config = ConfigDict(frozen=False)

    ticket_id: str = Field(..., description="Unique enterprise ticket ID, e.g. TKT-00042")
    alert_id: str = Field(..., description="Source alert")
    priority: str = Field(default="P3", description="P1 (critical) to P4 (low)")
    assignee_role: AgentRole = Field(..., description="Role currently assigned")
    status: str = Field(default="open", description="open | in_progress | resolved | closed")
    sla_steps_remaining: int = Field(default=40, description="Steps before SLA breach")
    app_chain: list[str] = Field(
        default_factory=lambda: ["SIEM"],
        description="Apps touched during resolution: SIEM, EDR, IAM, TICKETING"
    )
    notes: list[str] = Field(default_factory=list, description="Append-only audit trail")


class EnvironmentState(BaseModel):
    """Current episode state returned by GET /state."""

    model_config = ConfigDict(frozen=False)

    episode_id: str | None = Field(None, description="Current episode ID")
    task_id: str | None = Field(None, description="Current task ID")
    step_count: int = Field(default=0, description="Steps taken so far")
    max_steps: int = Field(default=0, description="Maximum steps for this episode")
    done: bool = Field(default=False, description="Whether episode is complete")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")
    alert_count: int = Field(default=0, description="Number of alerts in queue")
    classified_count: int = Field(default=0, description="Number of alerts classified so far")
    seed: int | None = Field(None, description="Scenario seed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional episode metadata")
    # Multi-agent state
    episode_mode: EpisodeMode = Field(default=EpisodeMode.TIER1_SOLO, description="Episode mode")
    current_phase: EpisodePhase | None = Field(None, description="Current team phase")
    current_role: AgentRole | None = Field(None, description="Expected active role")
